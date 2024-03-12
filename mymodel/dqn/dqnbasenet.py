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
        x3=x.detach()
        for layer in self.net:
            x4=layer(x3,x2)
            x3=x4.detach()
        return x4
          
class REMMultiBlock(nn.Module):
    def __init__(self,inputNum,extraNum,outNum,layers,funNum) -> None:
        super(REMMultiBlock,self).__init__()
        self.encorder=PolicyModule(inputNum,extraNum,outNum,layers)
        self.valueFun=Classifier(funNum,1)
        self.advFun=Classifier(funNum,13)

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
        clickEmbed1=self.embedingClick(clickList).squeeze(1)
        positionTensor=torch.tensor([0,1,2],dtype=torch.long).to(self.device)
        positionEmbed=self.positionEncoder(positionTensor)
        clickEmbed=(clickEmbed1+positionEmbed).view(len(lengths),1,-1)

        clickEncode=self.clickMLP(clickEmbed)
        newClickEmbed1=self.embedingClick(newClickList).squeeze(1)
        newPositionTensor=torch.tensor([0,1,2],dtype=torch.long).to(self.device)
        positionEmbedNew=self.positionEncoder(newPositionTensor)
        newClickEmbed=(newClickEmbed1+positionEmbedNew).view(len(lengths),1,-1)

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
        actions2=torch.cat([actions1,newClickEncode,hnM],dim=-1)
        personPosition=self.personEmbed(person).squeeze(1)
        scenePosition=self.sceneEmbed(scene).squeeze(1)
        actions3=actions2+personPosition+scenePosition
        ansList=[]
        for layer in self.remBlocks:
            ansList.append(layer(actions3))
        return ansList

class REMNet2_NO_PID(nn.Module):
    def __init__(self,device,embed_n=64,rnn_layer=10,remBlocskNum=5):
        super(REMNet2_NO_PID, self).__init__()
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
        clickEmbed1=self.embedingClick(clickList).squeeze(1)
        positionTensor=torch.tensor([0,1,2],dtype=torch.long).to(self.device)
        positionEmbed=self.positionEncoder(positionTensor)
        clickEmbed=(clickEmbed1+positionEmbed).view(len(lengths),1,-1)


        clickEncode=self.clickMLP(clickEmbed)
        newClickEmbed1=self.embedingClick(newClickList).squeeze(1)
        newPositionTensor=torch.tensor([0,1,2],dtype=torch.long).to(self.device)
        positionEmbedNew=self.positionEncoder(newPositionTensor)
        newClickEmbed=(newClickEmbed1+positionEmbedNew).view(len(lengths),1,-1)

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
        actions2=torch.cat([actions1,newClickEncode,hnM],dim=-1)
        # personPosition=self.personEmbed(person).squeeze(1)
        scenePosition=self.sceneEmbed(scene).squeeze(1)
        actions3=actions2+scenePosition
        ansList=[]
        for layer in self.remBlocks:
            ansList.append(layer(actions3))
        return ansList

class REMNet2_NO_SID(nn.Module):
    def __init__(self,device,embed_n=64,rnn_layer=10,remBlocskNum=5):
        super(REMNet2_NO_SID, self).__init__()
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
        clickEmbed1=self.embedingClick(clickList).squeeze(1)
        positionTensor=torch.tensor([0,1,2],dtype=torch.long).to(self.device)
        positionEmbed=self.positionEncoder(positionTensor)
        clickEmbed=(clickEmbed1+positionEmbed).view(len(lengths),1,-1)


        clickEncode=self.clickMLP(clickEmbed)
        newClickEmbed1=self.embedingClick(newClickList).squeeze(1)
        newPositionTensor=torch.tensor([0,1,2],dtype=torch.long).to(self.device)
        positionEmbedNew=self.positionEncoder(newPositionTensor)
        newClickEmbed=(newClickEmbed1+positionEmbedNew).view(len(lengths),1,-1)

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
        actions2=torch.cat([actions1,newClickEncode,hnM],dim=-1)
        personPosition=self.personEmbed(person).squeeze(1)
        actions3=actions2+personPosition
        ansList=[]
        for layer in self.remBlocks:
            ansList.append(layer(actions3))
        return ansList

class REMNet2_NO_SPID(nn.Module):
    def __init__(self,device,embed_n=64,rnn_layer=10,remBlocskNum=5):
        super(REMNet2_NO_SPID, self).__init__()
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
        clickEmbed1=self.embedingClick(clickList).squeeze(1)
        positionTensor=torch.tensor([0,1,2],dtype=torch.long).to(self.device)
        positionEmbed=self.positionEncoder(positionTensor)
        clickEmbed=(clickEmbed1+positionEmbed).view(len(lengths),1,-1)


        clickEncode=self.clickMLP(clickEmbed)
        newClickEmbed1=self.embedingClick(newClickList).squeeze(1)
        newPositionTensor=torch.tensor([0,1,2],dtype=torch.long).to(self.device)
        positionEmbedNew=self.positionEncoder(newPositionTensor)
        newClickEmbed=(newClickEmbed1+positionEmbedNew).view(len(lengths),1,-1)

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
        actions2=torch.cat([actions1,newClickEncode,hnM],dim=-1)
        ansList=[]
        for layer in self.remBlocks:
            ansList.append(layer(actions2))
        return ansList

class REMNet2_NO_LP(nn.Module):
    def __init__(self,device,embed_n=64,rnn_layer=10,remBlocskNum=5):
        super(REMNet2_NO_LP, self).__init__()
        # self.embedNum=embed_n
        self.device=device
        self.embedingEye=nn.Embedding(16,embed_n)
        self.embedingClick=nn.Embedding(12,embed_n)
        self.embedingEye.weight.requires_grad=False
        self.embedingClick.weight.requires_grad=False
        
        # self.personMLP=nn.Sequential(nn.Linear(1,embed_n*4),nn.LayerNorm(embed_n*4),nn.GELU())
        self.personEmbed=nn.Embedding(100,embed_n*3)
        self.sceneEmbed=nn.Embedding(10,embed_n*3)
        self.personEmbed.weight.requires_grad=False
        self.sceneEmbed.weight.requires_grad=False


        self.clickMLP=nn.Sequential(nn.Linear(embed_n*3,embed_n),nn.LayerNorm(embed_n),nn.GELU())
        self.clickMLPN=nn.Sequential(nn.Linear(embed_n*3,embed_n),nn.LayerNorm(embed_n),nn.GELU())
        self.positionEncoder=nn.Embedding(3,embed_n)
        self.positionEncoder.weight.requires_grad=False
        self.rnnLayer=rnn_layer
        
        self.rnn=torch.nn.GRU(embed_n*2,embed_n,2,batch_first=True)

        out_c=(rnn_layer+3)*embed_n
        self.remBlocks=nn.ModuleList()
        for i in range(remBlocskNum):
            self.remBlocks.append(REMMultiBlock(embed_n*3,embed_n,embed_n*3,rnn_layer,out_c))
        self.embedNum=embed_n
    
    def forward(self,clickList,eyeList,newClickList,lengths,person,scene):
        clickEmbed1=self.embedingClick(clickList).squeeze(1)
        positionTensor=torch.tensor([0,1,2],dtype=torch.long).to(self.device)
        positionEmbed=self.positionEncoder(positionTensor)
        clickEmbed=(clickEmbed1+positionEmbed).view(len(lengths),1,-1)
        clickEncode=self.clickMLP(clickEmbed)
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
        actions2=torch.cat([actions1,hnM],dim=-1)
        personPosition=self.personEmbed(person).squeeze(1)
        scenePosition=self.sceneEmbed(scene).squeeze(1)
        actions3=actions2+personPosition+scenePosition
        ansList=[]
        for layer in self.remBlocks:
            ansList.append(layer(actions3))
        return ansList

class PolicyNet(nn.Module):

    def __init__(self,device,embed_n=64,rnn_layer=10,remBlocskNum=5):
        super(PolicyNet, self).__init__()
        # self.embedNum=embed_n
        self.device=device
        self.embedingEye=nn.Embedding(16,embed_n)
        self.embedingClick=nn.Embedding(12,embed_n)
        self.embedingEye.weight.requires_grad=False
        self.embedingClick.weight.requires_grad=False
        
        # self.personMLP=nn.Sequential(nn.Linear(1,embed_n*4),nn.LayerNorm(embed_n*4),nn.GELU())
        self.personEmbed=nn.Embedding(100,embed_n*3)
        self.sceneEmbed=nn.Embedding(10,embed_n*3)
        self.personEmbed.weight.requires_grad=False
        self.sceneEmbed.weight.requires_grad=False

        self.clickMLP=nn.Sequential(nn.Linear(embed_n*3,embed_n),nn.LayerNorm(embed_n),nn.GELU())
        self.clickMLPN=nn.Sequential(nn.Linear(embed_n*3,embed_n),nn.LayerNorm(embed_n),nn.GELU())
        self.positionEncoder=nn.Embedding(3,embed_n)
        self.positionEncoder.weight.requires_grad=False
        self.rnnLayer=rnn_layer
        
        self.rnn=torch.nn.GRU(embed_n*2,embed_n,2,batch_first=True)

        self.encoder=PolicyModule(embed_n*3,embed_n,embed_n*3,rnn_layer)
        out_c=(rnn_layer+3)*embed_n
        # self.output=Classifier(out_c,12)
        self.output=nn.Linear(out_c,12)
        self.embedNum=embed_n
        self.softmax=nn.Softmax(dim=-1)
    

    def forward(self,clickList,eyeList,lengths,person,scene):
        clickEmbed1=self.embedingClick(clickList).squeeze(1)
        positionTensor=torch.tensor([0,1,2],dtype=torch.long).to(self.device)
        positionEmbed=self.positionEncoder(positionTensor)
        clickEmbed=(clickEmbed1+positionEmbed).view(len(lengths),1,-1)
        clickEncode=self.clickMLP(clickEmbed)
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
        actions2=torch.cat([actions1,hnM],dim=-1)
        personPosition=self.personEmbed(person).squeeze(1)
        scenePosition=self.sceneEmbed(scene).squeeze(1)
        actions3=actions2+personPosition+scenePosition
        ans=self.output(self.encoder(actions3,actions3))
        return ans

class PolicyNet2(nn.Module):

    def __init__(self,device,embed_n=64,rnn_layer=10,remBlocskNum=5):
        super(PolicyNet2, self).__init__()
        # self.embedNum=embed_n
        self.device=device
        self.embedingEye=nn.Embedding(16,embed_n)
        self.embedingClick=nn.Embedding(12,embed_n)
        self.embedingEye.weight.requires_grad=False
        self.embedingClick.weight.requires_grad=False
        
        # self.personMLP=nn.Sequential(nn.Linear(1,embed_n*4),nn.LayerNorm(embed_n*4),nn.GELU())
        self.personEmbed=nn.Embedding(100,embed_n*3)
        self.sceneEmbed=nn.Embedding(10,embed_n*3)
        self.personEmbed.weight.requires_grad=False
        self.sceneEmbed.weight.requires_grad=False

        self.clickMLP=nn.Sequential(nn.Linear(embed_n*3,embed_n),nn.LayerNorm(embed_n),nn.GELU())
        self.clickMLPN=nn.Sequential(nn.Linear(embed_n*3,embed_n),nn.LayerNorm(embed_n),nn.GELU())
        self.positionEncoder=nn.Embedding(3,embed_n)
        self.positionEncoder.weight.requires_grad=False
        self.rnnLayer=rnn_layer
        
        self.rnn=torch.nn.GRU(embed_n*2,embed_n,2,batch_first=True)

        self.encoder=PolicyModule(embed_n*3,embed_n,embed_n*3,rnn_layer)
        out_c=(rnn_layer+3)*embed_n
        # self.output=Classifier(out_c,12)
        # self.output=nn.Linear(out_c,12)
        self.output=nn.Sequential(nn.Linear(out_c,512),nn.LayerNorm(512),nn.GELU(),nn.Linear(512,256),nn.LayerNorm(256),nn.GELU(),\
                                  nn.Linear(256,64),nn.LayerNorm(64),nn.GELU(),nn.Linear(64,12))
        self.embedNum=embed_n
        self.softmax=nn.Softmax(dim=-1)
    

    def forward(self,clickList,eyeList,lengths,person,scene):
        clickEmbed1=self.embedingClick(clickList).squeeze(1)
        positionTensor=torch.tensor([0,1,2],dtype=torch.long).to(self.device)
        positionEmbed=self.positionEncoder(positionTensor)
        clickEmbed=(clickEmbed1+positionEmbed).view(len(lengths),1,-1)
        clickEncode=self.clickMLP(clickEmbed)
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
        actions2=torch.cat([actions1,hnM],dim=-1)
        personPosition=self.personEmbed(person).squeeze(1)
        scenePosition=self.sceneEmbed(scene).squeeze(1)
        actions3=actions2+personPosition+scenePosition
        ans=self.output(self.encoder(actions3,actions3))
        return ans


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