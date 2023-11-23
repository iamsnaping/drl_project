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
    def __init__(self,input_num,ex_n,identity,num_layer):#输入特征数，额外特征数， ，残差块的数量
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
        #定义嵌入层对输入进行处理
        self.embedingEye=nn.Embedding(16,embed_n)
        self.embedingClick=nn.Embedding(12,embed_n)
        self.embedingClickN=nn.Embedding(12,embed_n)

        #定义MLP层
        self.clickMLP=nn.Sequential(nn.Linear(embed_n*3,embed_n),nn.LayerNorm(embed_n),nn.GELU())
        self.clickMLPN=nn.Sequential(nn.Linear(embed_n*3,embed_n),nn.LayerNorm(embed_n),nn.GELU())
        self.positionEncoder=nn.Embedding(3,embed_n)
        self.positionEncoderN=nn.Embedding(3,embed_n)
        self.rnnLayer=rnn_layer

        #定义RNN
        # self.rnn=torch.nn.RNN(embed_n*2,embed_n,rnn_layer,batch_first=True)
        self.rnn=torch.nn.GRU(embed_n*2,embed_n,2,batch_first=True)

        #定义输出
        self.output=nn.Sequential(nn.LayerNorm(embed_n*2),nn.GELU(),nn.Linear(embed_n*2,13))
        #对RNN输出的预测点击数据拼接后的新数据进行编码  64*2,64,64*2，残差块的数量
        self.encoder=PolicyModule(embed_n*2,embed_n,embed_n*2,rnn_layer)
        out_c=(rnn_layer+2)*embed_n
        #定义状态价值函数和动作优势函数
        self.value_fun=nn.Linear(out_c,1)
        self.adv_fun=nn.Linear(out_c,13)
        self.embedNum=embed_n

    # eyelist-> [[],[],[]] clicklist-> [] tensor->long lengths->tensor not to cuda
    # sequence first
    # eyelist-> len->20
    # newClickList->3
    def forward(self,clickList,eyeList,newClickList,lengths):
        #处理实际的点击数据
        clickEmbed=self.embedingClick(clickList).squeeze(1)#b_s,1, 3 --> b_S,1,3,64  -->b_s, 3,64
        positionTensor=torch.tensor([0,1,2],dtype=torch.long).to(self.device)
        positionEmbed=self.positionEncoder(positionTensor)# 3,64
        clickEmbed+=positionEmbed
        clickEmbed=clickEmbed.view(len(lengths),1,-1)#（ b_s，1，64*3）
        clickEncode=self.clickMLP(clickEmbed)

        #处理预测的点击数据
        newClickEmbed=self.embedingClickN(newClickList).squeeze(1)
        newPositionTensor=torch.tensor([0,1,2],dtype=torch.long).to(self.device)
        positionEmbedNew=self.positionEncoderN(newPositionTensor)
        newClickEmbed+=positionEmbedNew
        newClickEmbed=newClickEmbed.view(len(lengths),1,-1)
        newClickEncode=self.clickMLPN(newClickEmbed)

        #处理眼动数据
        eyeEmbed=self.embedingEye(eyeList).squeeze(1)

        #将经过 MLP 处理的 "click" 数据 clickEncode 在第二个维度上复制 eyeEmbed.shape[1] 次-->b_s, eyeEmbed.shape[1], 64
        clickRepeat=clickEncode.repeat(1,eyeEmbed.shape[1],1)
        state=torch.cat([eyeEmbed,clickRepeat],dim=-1).to(self.device)
        """数对输入的序列数据进行打包，将变长序列处理为压缩形式(input：要打包的输入序列，即 state
                                                lengths：包含每个序列的长度的张量。在每个时间步上有多少个有效的元素
                                                enforce_sorted = False：表示不要求输入序列是按长度降序排列的
                                        batch_first = True：表示输入数据的形状是(batch_size, sequence_length, input_size)"""
        statePack=torch.nn.utils.rnn.pack_padded_sequence(state,lengths,enforce_sorted=False,batch_first=True).to(self.device)

        #创建了一个用于初始化 RNN（循环神经网络）隐藏状态的张量 h0
        h0=torch.zeros(2,eyeEmbed.shape[0],self.embedNum).to(self.device)
        # c0=torch.zeros(self.rnnLayer,eyeEmbed.shape[0],self.embedNum).to(self.device)
        #actions 包含了 RNN 对输入序列的处理结果，hn 是 RNN 在序列处理完成后的最终隐藏状态
        actions,hn=self.rnn(statePack,h0)
        # actions,hn=self.rnn(state,h0)
        # print(f'actionshape {actions.shape}')
        # hn=hn.permute(1,0,2)
        # actions1=actions[:,-1,:].unsqueeze(1)
        # print(actions1.shape)
        #对经过 RNN 处理后的序列数据进行解压（unpack），并选择每个序列的最后一个时间步的输出
        actionPad,_=torch.nn.utils.rnn.pad_packed_sequence(actions,batch_first=True)
        actions1=actionPad[[i for i in range(lengths.shape[0])],lengths-1,:].unsqueeze(1)
        # hnL=hn[:,-1,:].unsqueeze(1)
        # hnN=torch.cat([actions1,hnL],dim=-1)
        #把RNN的输出拼上预测的点击数据
        actions1=torch.cat([actions1,newClickEncode],dim=-1)
        #对拼接后的数据进行编码
        n=self.encoder(actions1,actions1)
        return self.value_fun(n)+self.adv_fun(n)





class func(nn.Module):
    def __init__(self,net):
        super().__init__()
        self.net=net
    def forward(self,a,b,c):
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









# if __name__=='__main__':
#     # base1=func(BaseNet(38,1,10,2))
#     # base2=func2(PolicyBaseNet(2,6,30,2,5,1))
#     # a,b,c,d=torch.randn(1,1,6),torch.randn(1,1,20),torch.randn(1,1,10),torch.randn(1,1,2)
#     # a=torch.randint(0,8,(8192,1,3)).to('cuda:2')
#     # b=torch.randint(0,8,(8192,1,3)).to('cuda:2')
#
#     # c=torch.randn((8192,3,1)).to('cuda:2')
#     # dqnnet=DQNNet(32,9,4,device='cuda:2')
#     # dqnnet.to('cuda:2')
#     # # for name,param in dqnnet.named_parameters():
#     # #     print(name,param.data)
#     # # print(dqnnet)
#     # dqnnet(a,b,c)
#     # flops, params = profile(dqnnet, inputs=(a,b,c))
#     # macs,params=clever_format([flops,params],'%.3f')
#     # print(macs,params)
#     # flops, params = profile(dqnnet, inputs=(a,b,c))
#     # macs,params=clever_format([flops,params],'%.3f')
#     # print(macs,params)
#     # dqnnet=DQNNet(10,9,5)
#     # a=torch.randn((6,12,1),dtype=torch.float32)
#     # numlayers batch outputlen
#     # dqnnet(a)
#     # device='cuda:0'
#     device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#
#     dqnnet2=DQNRNNNet(device=device,embed_n=3,rnn_layer=1)
#     dqnnet2.to(device)
#     tensorList=[]
#     lengths=torch.zeros(5,dtype=torch.long)
#     tensorList2=[]
#     # print(lengths,lengths.shape)
#     k=0
#     for i in range(10,0,-1):
#         if i&1!=0:
#             continue
#         import numpy as np
#         tensorList.append(torch.arange(0,i+1,dtype=torch.long))
#         tensorList2.append(torch.arange(3,dtype=torch.long))
#         print(tensorList[-1].shape)
#         lengths[k]=i
#         k+=1
#     # eye=torch.tensor(tensorList,dtype=torch.long)
#     # eye = torch.tensor([item.detach().numpy() for item in tensorList])
#
#
#     print(lengths)
#     click=torch.tensor([0,1,2],dtype=torch.long).to(device).repeat(len(tensorList),1).squeeze(1)
#     # ans=dqnnet2(click,tensorList,lengths)
#     print(click)
#     tensorList2=torch.stack(tensorList2,dim=0).to(device).unsqueeze(1)
#     # print(tensorList2)
#     print(f't2 {tensorList2.shape}')
#     # breakpoint()
#     print(tensorList2.shape,click.shape)
#     # print(tensorList.type,tensorList2.type)
#     ans2=dqnnet2(click,tensorList2,lengths)
#     # print(ans.shape,ans)
#     # print()
#     print(ans2.shape,ans2)

    '''
    name:
    TODO:测试dqnbasenet.py模型
    time:2023-11-23
    params:
    returns:模型结构
    '''
if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    import random
    # 创建 DQNRNNNet 模型
    dqnnet2 = DQNRNNNet(device=device, embed_n=64, rnn_layer=10)
    dqnnet2.to(device)

    # 创建测试数据
    max_length = 10
    batch_size = 32
    eyeList = [torch.randint(0, 16, (random.randint(5, max_length),)) for _ in range(batch_size)]
    eyeList_padded = torch.nn.utils.rnn.pad_sequence(eyeList, batch_first=True)
    lengths = torch.tensor([len(seq) for seq in eyeList], dtype=torch.long)
    eyeList_padded = eyeList_padded.to(torch.long)
    lengths = lengths.to(torch.long)
    eyeList = eyeList_padded
    # eyeList = eyeList[0]
    # lengths = lengths[0]
    print("eyeList:", eyeList)
    print("lengths:", lengths)
    print("eyeList shpae:", eyeList.shape)
    print("lengths shape:", lengths.shape)

    clickList = torch.randint(0, 12, (32,3)).to(device)
    newClickList = torch.randint(0, 12, (32,3)).to(device)
    print("clickList:",clickList)
    print("newClickList:",newClickList )


    # 运行前向传播
    output = dqnnet2(clickList, eyeList, newClickList, lengths)
    print(output)
    # 输出模型结构
    print(dqnnet2)





