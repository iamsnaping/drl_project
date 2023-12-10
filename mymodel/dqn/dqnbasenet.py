import torch
from torch import nn
from torch.nn import functional as F
from thop import profile
from thop import clever_format
from torchstat import stat



class PolicyResBlock(nn.Module):

    def __init__(self,i_put,ex_n,identity):
        super(PolicyResBlock, self).__init__()
        self.s1=nn.Sequential(nn.Linear(i_put,i_put*2),nn.LayerNorm(i_put*2),nn.GELU(),nn.Linear(i_put*2,i_put))
        self.s2=nn.Linear(identity,ex_n)
        self.norm=nn.Sequential(nn.LayerNorm(i_put+ex_n),nn.GELU())
    def forward(self,x1,x2):
        x=self.s1(x1)
        x3=self.s2(x2)
        return self.norm(torch.concat([x1+x,x3],dim=-1))

class PolicyModule(nn.Module):
    def __init__(self,input_num,ex_n,identity,num_layer):
        super(PolicyModule, self).__init__()
        self.net=nn.ModuleList()
        i_n=input_num
        for i in range(num_layer):
            self.net.append(PolicyResBlock(i_n,ex_n,identity))
            i_n=i_n+ex_n

    def forward(self,x,x2):
        for layer in self.net:
            x=layer(x,x2)
        return x

class DQNNet(nn.Module):
    # 6 20 10 2 10 3
    def __init__(self,embed_n,output_n,num_layer,device,rnn_layer=2):
        super(DQNNet, self).__init__()
        self.embedingEye=nn.Embedding(16,embed_n)
        self.embedingClick=nn.Embedding(12,embed_n)
        self.indiceEmbeding=nn.Sequential(nn.Linear(1,int(0.5*embed_n)),nn.LayerNorm(int(0.5*embed_n)),nn.Sigmoid())
        self.rnn_layer=rnn_layer
        self.rnnOut=embed_n*4
        self.rnn=torch.nn.GRU(embed_n*2+int(0.5*embed_n),embed_n*4,rnn_layer)
        self.encoder=PolicyModule(embed_n*4,embed_n,embed_n*4,num_layer)
        out_c=(num_layer+4)*embed_n
        self.value_fun=nn.Linear(out_c,1)
        self.adv_fun=nn.Linear(out_c,output_n)
        self.device=device

    # click eye seq
    def forward(self,a,b,indices):
        h0=torch.zeros((self.rnn_layer,a.shape[0],self.rnnOut)).to(self.device)
        # print(a.shape)
        a=self.embedingClick(a)
        b=self.embedingEye(b)
        # # print(a,b)
        # print(a.shape)
        indicesEmbeding=self.indiceEmbeding(indices)
        a=torch.squeeze(a,dim=1)
        b=torch.squeeze(b,dim=1)
        # print(a.shape,b.shape,indicesEmbeding.shape)
        c=torch.cat([a,b,indicesEmbeding],dim=-1)
        d=torch.permute(c,(1,0,2))
        # print(d.shape,h0.shape)
        # print(a.shape,b.shape,c.shape,d.shape)
        output,hn=self.rnn(d,h0)
        output_1=torch.permute(output[-1].unsqueeze(0),(1,0,2))
        # output_2=torch.cat([output_1,indicesEmbeding],dim=-1)
        n=self.encoder(output_1,output_1)
        return self.value_fun(n)+self.adv_fun(n)

class DQNRNNNet(nn.Module):
    def __init__(self,device,embed_n=64,rnn_layer=10):
        super(DQNRNNNet, self).__init__()
        # self.embedNum=embed_n
        self.device=device
        self.embedingEye=nn.Embedding(16,embed_n)
        self.embedingClick=nn.Embedding(12,embed_n)
        self.embedingClickN=nn.Embedding(12,embed_n)

        self.clickMLP=nn.Sequential(nn.Linear(embed_n*3,embed_n),nn.LayerNorm(embed_n),nn.GELU())
        self.clickMLPN=nn.Sequential(nn.Linear(embed_n*3,embed_n),nn.LayerNorm(embed_n),nn.GELU())
        self.positionEncoder=nn.Embedding(3,embed_n)
        self.positionEncoderN=nn.Embedding(3,embed_n)
        self.rnnLayer=rnn_layer
        
        self.hnMLP=nn.Sequential(nn.Linear(embed_n*2,embed_n),nn.LayerNorm(embed_n),nn.GELU())
        # self.rnn=torch.nn.RNN(embed_n*2,embed_n,rnn_layer,batch_first=True)
        self.rnn=torch.nn.GRU(embed_n*2,embed_n,2,batch_first=True)

        self.output=nn.Sequential(nn.LayerNorm(embed_n*2),nn.GELU(),nn.Linear(embed_n*2,13))
        self.encoder=PolicyModule(embed_n*4,embed_n,embed_n*4,rnn_layer)
        out_c=(rnn_layer+4)*embed_n
        self.value_fun=nn.Linear(out_c,1)
        self.adv_fun=nn.Linear(out_c,13)
        self.embedNum=embed_n
    
    # eyelist-> [[],[],[]] clicklist-> [] tensor->long lengths->tensor not to cuda
    # sequence first 
    # eyelist-> len->20
    # newClickList->3
    def forward(self,clickList,eyeList,newClickList,lengths):

        clickEmbed=self.embedingClick(clickList).squeeze(1)
        positionTensor=torch.tensor([0,1,2],dtype=torch.long).to(self.device)
        positionEmbed=self.positionEncoder(positionTensor)
        clickEmbed+=positionEmbed     
        clickEmbed=clickEmbed.view(len(lengths),1,-1)
        clickEncode=self.clickMLP(clickEmbed)

        newClickEmbed=self.embedingClickN(newClickList).squeeze(1)
        newPositionTensor=torch.tensor([0,1,2],dtype=torch.long).to(self.device)
        positionEmbedNew=self.positionEncoderN(newPositionTensor)
        newClickEmbed+=positionEmbedNew
        newClickEmbed=newClickEmbed.view(len(lengths),1,-1)
        newClickEncode=self.clickMLPN(newClickEmbed)


        eyeEmbed=self.embedingEye(eyeList).squeeze(1)
        clickRepeat=clickEncode.repeat(1,eyeEmbed.shape[1],1)

        state=torch.cat([eyeEmbed,clickRepeat],dim=-1).to(self.device)

        statePack=torch.nn.utils.rnn.pack_padded_sequence(state,lengths,enforce_sorted=False,batch_first=True).to(self.device)
        h0=torch.zeros(2,eyeEmbed.shape[0],self.embedNum).to(self.device)
        # c0=torch.zeros(self.rnnLayer,eyeEmbed.shape[0],self.embedNum).to(self.device)
      
        actions,hn=self.rnn(statePack,h0)
        # actions,hn=self.rnn(state,h0)
        # print(f'actionshape {actions.shape}')
        # hn=hn.permute(1,0,2)
        # actions1=actions[:,-1,:].unsqueeze(1)
        # print(actions1.shape)
        actionPad,_=torch.nn.utils.rnn.pad_packed_sequence(actions,batch_first=True)
        actions1=actionPad[[i for i in range(lengths.shape[0])],lengths-1,:].unsqueeze(1)
        # print(hn.shape)
        # hnM=hn[-1,:,:].unsqueeze(1)
        hn=torch.permute(hn,(1,0,2))
        hnM=hn.reshape((lengths.shape[0],1,-1))
        # hnM=self.hnMLP(hnM)
        actions1=torch.cat([actions1,newClickEncode,hnM],dim=-1)
        n=self.encoder(actions1,actions1)
        return self.value_fun(n)+self.adv_fun(n)
           
class REMMultiBlock(nn.Module):
    def __init__(self,inputNum,extraNum,outNum,layers,funNum) -> None:
        super(REMMultiBlock,self).__init__()
        self.encorder=PolicyModule(inputNum,extraNum,outNum,layers)
        self.valueFun=Classifier(funNum,1)
        self.advFun=Classifier(funNum,13)
        # self.valueFun=nn.Linear(funNum,1)
        # self.advFun=nn.Linear(funNum,13)

    def forward(self,actions):
        n=self.encorder(actions,actions)
        return self.valueFun(n)+self.advFun(n)

class REMNet(nn.Module):
    def __init__(self,device,embed_n=64,rnn_layer=10,remBlocskNum=5):
        super(REMNet, self).__init__()
        # self.embedNum=embed_n
        self.device=device
        self.embedingEye=nn.Embedding(16,embed_n)
        self.embedingClick=nn.Embedding(12,embed_n)
        # self.embedingClickN=nn.Embedding(12,embed_n)
        self.embedingEye.weight.requires_grad=False
        self.embedingClick.weight.requires_grad=False
 
        
        self.personMLP=nn.Sequential(nn.Linear(1,embed_n*4),nn.LayerNorm(embed_n*4),nn.GELU())

        self.clickMLP=nn.Sequential(nn.Linear(embed_n*3,embed_n),nn.LayerNorm(embed_n),nn.GELU())
        self.clickMLPN=nn.Sequential(nn.Linear(embed_n*3,embed_n),nn.LayerNorm(embed_n),nn.GELU())
        self.positionEncoder=nn.Embedding(3,embed_n)
        # self.positionEncoderN=nn.Embedding(3,embed_n)
        self.positionEncoder.weight.requires_grad=False
        # self.positionEncoderN.weight.requires_grad=False
        self.rnnLayer=rnn_layer
        
        self.hnMLP=nn.Sequential(nn.Linear(embed_n*2,embed_n),nn.LayerNorm(embed_n),nn.GELU())
        # self.rnn=torch.nn.RNN(embed_n*2,embed_n,rnn_layer,batch_first=True)
        self.rnn=torch.nn.GRU(embed_n*2,embed_n,2,batch_first=True)

        self.output=nn.Sequential(nn.LayerNorm(embed_n*2),nn.GELU(),nn.Linear(embed_n*2,13))
        self.encoder=PolicyModule(embed_n*4,embed_n,embed_n*4,rnn_layer)
        out_c=(rnn_layer+4)*embed_n
        self.value_fun=nn.Linear(out_c,1)
        self.adv_fun=nn.Linear(out_c,13)
        self.remBlocks=nn.ModuleList()
        for i in range(remBlocskNum):
            self.remBlocks.append(REMMultiBlock(embed_n*4,embed_n,embed_n*4,rnn_layer,out_c))
        self.embedNum=embed_n
    
    # eyelist-> [[],[],[]] clicklist-> [] tensor->long lengths->tensor not to cuda
    # sequence first 
    # eyelist-> len->20
    # newClickList->3
    def forward(self,clickList,eyeList,newClickList,lengths,person):
        clickEmbed=self.embedingClick(clickList).squeeze(1)
        positionTensor=torch.tensor([0,1,2],dtype=torch.long).to(self.device)
        positionEmbed=self.positionEncoder(positionTensor)
        clickEmbed+=positionEmbed     
        clickEmbed=clickEmbed.view(len(lengths),1,-1)
        clickEncode=self.clickMLP(clickEmbed)
        newClickEmbed=self.embedingClick(newClickList).squeeze(1)
        newPositionTensor=torch.tensor([0,1,2],dtype=torch.long).to(self.device)
        positionEmbedNew=self.positionEncoder(newPositionTensor)
        newClickEmbed+=positionEmbedNew
        newClickEmbed=newClickEmbed.view(len(lengths),1,-1)
        newClickEncode=self.clickMLPN(newClickEmbed)
        eyeEmbed=self.embedingEye(eyeList).squeeze(1)
        clickRepeat=clickEncode.repeat(1,eyeEmbed.shape[1],1)
        state=torch.cat([eyeEmbed,clickRepeat],dim=-1).to(self.device)
        statePack=torch.nn.utils.rnn.pack_padded_sequence(state,lengths,enforce_sorted=False,batch_first=True).to(self.device)
        h0=torch.zeros(2,eyeEmbed.shape[0],self.embedNum).to(self.device)
        actions,hn=self.rnn(statePack,h0)
        actionPad,_=torch.nn.utils.rnn.pad_packed_sequence(actions,batch_first=True)
        actions1=actionPad[[i for i in range(lengths.shape[0])],lengths-1,:].unsqueeze(1)
        hn=torch.permute(hn,(1,0,2))
        hnM=hn.reshape((lengths.shape[0],1,-1))
        actions1=torch.cat([actions1,newClickEncode,hnM],dim=-1)
        personPosition=self.personMLP(person)
        actions1+=personPosition
        ansList=[]
        for layer in self.remBlocks:
            ansList.append(layer(actions1))
        return ansList

class Classifier(nn.Module):

    def __init__(self,inputNum,outputNum) -> None:
        super(Classifier,self).__init__()
        self.s1=nn.Sequential(nn.Linear(inputNum,int(inputNum/2)),nn.LayerNorm(int(inputNum/2)),nn.GELU(),nn.Linear(int(inputNum/2),outputNum))
        self.s2=nn.Linear(inputNum,outputNum)
    

    def forward(self,x):
        return self.s1(x)+self.s2(x)

# scnene embed
class REMNet2(nn.Module):
    def __init__(self,device,embed_n=64,rnn_layer=10,remBlocskNum=5):
        super(REMNet2, self).__init__()
        # self.embedNum=embed_n
        self.device=device
        self.embedingEye=nn.Embedding(16,embed_n)
        self.embedingClick=nn.Embedding(12,embed_n)
        self.embedingEye.weight.requires_grad=False
        self.embedingClick.weight.requires_grad=False
        
        # self.personMLP=nn.Sequential(nn.Linear(1,embed_n*4),nn.LayerNorm(embed_n*4),nn.GELU())
        self.personEmbed=nn.Embedding(100,embed_n*4)
        self.sceneEmbed=nn.Embedding(10,embed_n*4)
        self.personEmbed.weight.requires_grad=False
        self.sceneEmbed.weight.requires_grad=False


        self.clickMLP=nn.Sequential(nn.Linear(embed_n*3,embed_n),nn.LayerNorm(embed_n),nn.GELU())
        self.clickMLPN=nn.Sequential(nn.Linear(embed_n*3,embed_n),nn.LayerNorm(embed_n),nn.GELU())
        self.positionEncoder=nn.Embedding(3,embed_n)
        self.positionEncoder.weight.requires_grad=False
        self.rnnLayer=rnn_layer
        
        self.rnn=torch.nn.GRU(embed_n*2,embed_n,2,batch_first=True)

        self.output=nn.Sequential(nn.LayerNorm(embed_n*2),nn.GELU(),nn.Linear(embed_n*2,13))
        self.encoder=PolicyModule(embed_n*4,embed_n,embed_n*4,rnn_layer)
        out_c=(rnn_layer+4)*embed_n
        self.remBlocks=nn.ModuleList()
        for i in range(remBlocskNum):
            self.remBlocks.append(REMMultiBlock(embed_n*4,embed_n,embed_n*4,rnn_layer,out_c))
        self.embedNum=embed_n
    
    # eyelist-> [[],[],[]] clicklist-> [] tensor->long lengths->tensor not to cuda
    # sequence first 
    # eyelist-> len->20
    # newClickList->3
    def forward(self,clickList,eyeList,newClickList,lengths,person,scene):
        clickEmbed=self.embedingClick(clickList).squeeze(1)
        positionTensor=torch.tensor([0,1,2],dtype=torch.long).to(self.device)
        positionEmbed=self.positionEncoder(positionTensor)
        clickEmbed+=positionEmbed     
        clickEmbed=clickEmbed.view(len(lengths),1,-1)
        clickEncode=self.clickMLP(clickEmbed)
        newClickEmbed=self.embedingClick(newClickList).squeeze(1)
        newPositionTensor=torch.tensor([0,1,2],dtype=torch.long).to(self.device)
        positionEmbedNew=self.positionEncoder(newPositionTensor)
        newClickEmbed+=positionEmbedNew
        newClickEmbed=newClickEmbed.view(len(lengths),1,-1)
        newClickEncode=self.clickMLPN(newClickEmbed)
        eyeEmbed=self.embedingEye(eyeList).squeeze(1)
        clickRepeat=clickEncode.repeat(1,eyeEmbed.shape[1],1)
        state=torch.cat([eyeEmbed,clickRepeat],dim=-1).to(self.device)
        statePack=torch.nn.utils.rnn.pack_padded_sequence(state,lengths,enforce_sorted=False,batch_first=True).to(self.device)
        h0=torch.zeros(2,eyeEmbed.shape[0],self.embedNum).to(self.device)
        actions,hn=self.rnn(statePack,h0)
        actionPad,_=torch.nn.utils.rnn.pad_packed_sequence(actions,batch_first=True)
        actions1=actionPad[[i for i in range(lengths.shape[0])],lengths-1,:].unsqueeze(1)
        hn=torch.permute(hn,(1,0,2))
        hnM=hn.reshape((lengths.shape[0],1,-1))
        actions1=torch.cat([actions1,newClickEncode,hnM],dim=-1)
        personPosition=self.personEmbed(person).squeeze(1)
        scenePosition=self.sceneEmbed(scene).squeeze(1)
        
        # actions1=actions1+personPosition+scenePosition
        actions1=actions1+personPosition
        ansList=[]
        for layer in self.remBlocks:
            ansList.append(layer(actions1))
        return ansList

# clickList,eyeList,newClickList,lengths,person,scene
class func(nn.Module):
    def __init__(self,net):
        super().__init__()
        self.net=net
    def forward(self,a,b,c,):
        # a=torch.randn((6,12,2),dtype=torch.float32)
        out=self.net(a,b,c)
        return out
class func2(nn.Module):
    def __init__(self,net):
        super().__init__()
        self.net=net
    def forward(self,a,b,c):
        # a=torch.randn((3,12,2),dtype=torch.float32)
        out=self.net(a,b,c)
        return out









if __name__=='__main__':
# clickList,eyeList,newClickList,lengths,person,scene
    a,b,c,d,e,f=torch.randint(0,10,(1,1,3)),torch.randint(0,10,(1,1,20)),torch.randint(0,10,(1,1,3)),\
        torch.randint(0,10,(1,)),torch.randint(0,10,(1,1,1)),torch.randint(0,10,(1,1,1))
    dqnnet=REMNet2(device='cuda:2',embed_n=128,rnn_layer=5,remBlocskNum=5)
    dqnnet.to('cuda:2')
    a=a.to('cuda:2')
    b=b.to('cuda:2')
    c=c.to('cuda:2')
    e=e.to('cuda:2')
    f=f.to('cuda:2')
    # for name,param in dqnnet.named_parameters():
    #     print(name,param.data)
    # print(dqnnet)
    flops, params = profile(dqnnet, inputs=(a,b,c,d,e,f))
    macs,params=clever_format([flops,params],'%.3f')
    print(macs,params)
    # flops, params = profile(dqnnet, inputs=(a,b,c,d,e,f))
    # macs,params=clever_format([flops,params],'%.3f')
    # print(macs,params)