import torch
from torch import nn
from torch.nn import functional as F
from thop import profile
from thop import clever_format
from torchstat import stat



class BaseResBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(BaseResBlock,self).__init__()
        self.s1=nn.Sequential(nn.Linear(in_c,in_c+out_c),nn.LayerNorm(in_c+out_c),nn.GELU(),nn.Linear(in_c+out_c,out_c))
        self.meta=nn.Linear(in_c,out_c)
        self.norm=nn.Sequential(nn.LayerNorm(out_c),nn.GELU())
    
    def forward(self,x):
        return self.norm(self.s1(x)+self.meta(x))



class BaseResBlock2(nn.Module):
    def __init__(self, in_c, out_c):
        super(BaseResBlock2,self).__init__()
        self.s1=nn.Sequential(nn.Linear(in_c,in_c+out_c),nn.LayerNorm(in_c+out_c),nn.GELU(),nn.Linear(in_c+out_c,out_c))
        self.meta=nn.Linear(in_c,out_c)
        self.norm=nn.Sequential(nn.LayerNorm(out_c),nn.GELU())
    
    def forward(self,x):
        return self.s1(x)+self.meta(x)


class BaseModule(nn.Module):
    def __init__(self,in_n,layer_n):
        super(BaseModule,self).__init__()
        self.module_list=nn.ModuleList()
        for i in range(layer_n):
            self.module_list.append(BaseResBlock(in_n,in_n))

    def forward(self,x):
        for layer in self.module_list:
            x=layer(x)
        return x

class BaseNet(nn.Module):
    def __init__(self,in_n,ou_n,layer_n,meta_n):
        super(BaseNet,self).__init__()
        self.s1=BaseModule(in_n,layer_n)
        self.out=BaseResBlock2(in_n,ou_n)
        self.meta=BaseResBlock2(meta_n,ou_n)

    def forward(self,a,b,c,d):
        s=torch.cat([a,b,c,d],dim=-1)
        return self.out(self.s1(s))+self.meta(d)








class PolicyModule(nn.Module):
    def __init__(self, in_c, out_c, num_layers):
        super(PolicyModule, self).__init__()
        self.net = nn.ModuleList(
            [nn.Sequential(nn.Linear(in_c, out_c), nn.LayerNorm(out_c), nn.Tanh()) for i in range(num_layers)])
        self.nums_layers = num_layers

    def forward(self, x, a_meta, as_meta):
        i = 0
        for layer in self.net:
            x = layer(x)
            if i != self.nums_layers - 1:
                # x=as_meta+x+a_meta
                x = torch.concat([as_meta, x, a_meta], dim=-1)
            i += 1
        return x

class PolicyResBlock(nn.Module):

    def __init__(self,i_put,ex_n):
        super(PolicyResBlock, self).__init__()
        self.s1=nn.Sequential(nn.Linear(i_put,i_put*2),nn.LayerNorm(i_put*2),nn.GELU(),nn.Linear(i_put*2,i_put))
        self.norm=nn.Sequential(nn.LayerNorm(i_put+ex_n),nn.GELU())
    def forward(self,x1,x2):
        # print(x1.shape)
        x=self.s1(x1)
        return self.norm(torch.concat([x1+x,x2],dim=-1))

class PolicyResBlock2(nn.Module):
    def __init__(self,i_put,i_out):
        super(PolicyResBlock2, self).__init__()
        self.s1=nn.Sequential(nn.Linear(i_put,i_put+i_out),nn.LayerNorm(i_put+i_out),nn.GELU(),nn.Linear(i_put+i_out,i_out))
        self.s2=nn.Linear(i_put,i_out)
        self.norm=nn.Sequential(nn.LayerNorm(i_out))

    def forward(self,x1):
        s1=self.s1(x1)
        s2=self.s2(x1)
        return s1+s2


class PolicyModule2(nn.Module):
    def __init__(self,input_num,ex_n,num_layer):
        super(PolicyModule2, self).__init__()
        self.net=nn.ModuleList()
        i_n=input_num
        for i in range(num_layer):
            self.net.append(PolicyResBlock(i_n,ex_n))
            i_n=i_n+ex_n

    def forward(self,x,x2):
        for layer in self.net:
            x=layer(x,x2)
        return x

class PolicyBaseNet(nn.Module):
    # 6 20 10 2 10 3
    def __init__(self,action_n,state_n,n_state_n,mouse_n ,num_layer,output_n):
        super(PolicyBaseNet, self).__init__()
        encoder_nums=int(num_layer/2+0.5)
        decoder_nums=num_layer-encoder_nums

        self.encoder=PolicyModule2((action_n+state_n+mouse_n+n_state_n),(action_n+state_n+mouse_n+n_state_n),encoder_nums)
        out_c=(encoder_nums+1)*((action_n+state_n+mouse_n+n_state_n))
        self.decoder=PolicyModule2(out_c,mouse_n,decoder_nums)
        out_c=out_c+decoder_nums*mouse_n
        self.s2=PolicyResBlock2(out_c,output_n)
        self.a_meta=PolicyResBlock2(action_n,output_n)
        self.m_meta=PolicyResBlock2(mouse_n,output_n)
        self.s_meta=PolicyResBlock2(state_n,output_n)
        self.ns_meta=PolicyResBlock2(n_state_n,output_n)
        self.t_meta=PolicyResBlock2((action_n+state_n+mouse_n+n_state_n),output_n)


    def forward(self,a,b,c,d):
        # s1=self.s1(torch.concat([a,b,c,d],dim=-1))
        s1=torch.concat([a,b,c,d],dim=-1)
        n=self.encoder(s1,s1)
        n=self.decoder(n,d)
        # print(n.shape)
        return self.s2(n)+self.m_meta(d)




class PolicyBaseNet2(nn.Module):
    # 6 20 10 2 10 3
    def __init__(self,action_n,state_n,n_state_n,mouse_n ,num_layer,output_n):
        super(PolicyBaseNet2, self).__init__()
        encoder_nums=int(num_layer/2+0.5)
        decoder_nums=num_layer-encoder_nums

        self.encoder=PolicyModule2((action_n+state_n+mouse_n+n_state_n),(action_n+state_n+mouse_n+n_state_n),encoder_nums)
        out_c=(encoder_nums+1)*((action_n+state_n+mouse_n+n_state_n))
        self.decoder=PolicyModule2(out_c,mouse_n,decoder_nums)
        out_c=out_c+decoder_nums*mouse_n
        self.s2=PolicyResBlock2(out_c,output_n)
        self.a_meta=PolicyResBlock2(action_n,output_n)
        self.m_meta=PolicyResBlock2(mouse_n,output_n)
        self.s_meta=PolicyResBlock2(state_n,output_n)
        self.ns_meta=PolicyResBlock2(n_state_n,output_n)
        self.t_meta=PolicyResBlock2((action_n+state_n+mouse_n+n_state_n),output_n)


    def forward(self,a,b,c,d):
        # s1=self.s1(torch.concat([a,b,c,d],dim=-1))
        s1=torch.concat([a,b,c,d],dim=-1)
        n=self.encoder(s1,s1)
        n=self.decoder(n,d)
        # print(n.shape)
        return torch.tanh(self.s2(n)+self.m_meta(d))




class CriticResBlock2(nn.Module):

    def __init__(self,i_put,i_out):
        super(CriticResBlock2, self).__init__()
        self.s1=nn.Sequential(nn.Linear(i_put,i_put+i_out),nn.Tanh(),nn.Linear(i_put+i_out,i_out))
        self.s2=nn.Linear(i_put,i_out)
        self.norm=nn.Sequential(nn.LayerNorm(i_out))
    def forward(self,x1):
        s1=self.s1(x1)
        s2=self.s2(x1)
        return self.norm(s1+s2)

class CriticResBlock(nn.Module):
    def __init__(self,i_put,ex_n):
        super(CriticResBlock, self).__init__()
        self.s1=nn.Sequential(nn.Linear(i_put,i_put*2),nn.ReLU(),nn.Linear(i_put*2,i_put))
        self.norm=nn.Sequential(nn.LayerNorm(i_put+ex_n),nn.ReLU())
    def forward(self,x1,x2):
        x=self.s1(x1)
        return self.norm(torch.concat([x1+x,x2],dim=-1))

class CriticModule(nn.Module):
    def __init__(self,input_num,ex_n,num_layer):
        super(CriticModule, self).__init__()
        self.net=nn.ModuleList()
        i_n=input_num
        for i in range(num_layer):
            self.net.append(CriticResBlock(i_n,ex_n))
            i_n=i_n+ex_n

    def forward(self,x,x2):
        for layer in self.net:
            x=layer(x,x2)
        return x


class CriticBase(nn.Module):
    # 6 30 2 10 3
    def __init__(self,action_n,state_n,n_state_n,mouse_n ,num_layer,output_n):
        super(CriticBase, self).__init__()
        # self.s1=nn.Sequential(nn.Linear(action_n+state_n+mouse_n,n_state_n,(action_n+state_n+mouse_n+n_state_n)*2),nn.LayerNorm((action_n+state_n+mouse_n)*2),nn.ReLU())
        self.net=CriticModule((action_n+state_n+mouse_n+n_state_n),(action_n+state_n+mouse_n+n_state_n),num_layer)
        out_c=(1+num_layer)*((action_n+state_n+mouse_n+n_state_n))
        self.norm=nn.LayerNorm(action_n+state_n+mouse_n+n_state_n)
        # self.s2=nn.Sequential(nn.Linear(out_c,output_n))
        self.s2=CriticResBlock2(out_c,output_n)
        self.a_meta=nn.Sequential(nn.Linear(action_n,output_n))
        self.m_meta=nn.Sequential(nn.Linear(mouse_n,output_n))
        self.s_meta=nn.Sequential(nn.Linear(state_n,output_n))
        self.ns_meta=nn.Sequential(nn.Linear(n_state_n,output_n))
        self.t_meta=nn.Sequential(nn.Linear((action_n+state_n+mouse_n+n_state_n),output_n))


    def forward(self,a,b,c,d):
        # s1=self.s1(torch.concat([a,b,c,d],dim=-1))
        s1=torch.concat([a,b,c,d],dim=-1)
        n=self.net(s1,s1)
        return self.s2(n)+self.a_meta(a)+self.s_meta(b)+self.ns_meta(c)+self.m_meta(d)+self.t_meta(s1)



class func(nn.Module):
    def __init__(self,net):
        super().__init__()
        self.net=net
    def forward(self,a):
        a,b,c,d=torch.randn(1,1,6),torch.randn(1,1,20),torch.randn(1,1,10),torch.randn(1,1,2)
        out=self.net(a,b,c,d)
        return out
class func2(nn.Module):
    def __init__(self,net):
        super().__init__()
        self.net=net
    def forward(self,a):
        a,b,c,d=torch.randn(1,1,6),torch.randn(1,1,30),torch.randn(1,1,10),torch.randn(1,1,2)
        out=self.net(d,a,b,d)
        return out

if __name__=='__main__':
    base1=func(BaseNet(38,1,10,2))
    base2=func2(PolicyBaseNet(2,6,30,2,5,1))
    a,b,c,d=torch.randn(1,1,6),torch.randn(1,1,20),torch.randn(1,1,10),torch.randn(1,1,2)
    flops, params = profile(base1, inputs=(a,))
    macs,params=clever_format([flops,params],'%.3f')
    print(macs,params)
    flops, params = profile(base2, inputs=(a,))
    macs,params=clever_format([flops,params],'%.3f')
    print(macs,params)
