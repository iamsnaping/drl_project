import torch
from torch import nn



class ResdualBlock2(nn.Module):

    def __init__(self,input_nums,extra_n):
        super(ResdualBlock2,self).__init__()
        self.net=nn.Sequential(nn.Linear(input_nums,input_nums*2),nn.ReLU(),nn.Linear(input_nums*2,input_nums))
        self.norm=nn.LayerNorm(input_nums+extra_n)
    def forward(self,x,extra):
        x1=self.net(x)
        return self.norm(torch.concat([x+x1,extra],dim=-1))


# class ResdualBlock(nn.Module):

#     def __init__(self, input1, input2, output1):
#         super(ResdualBlock, self).__init__()
#         self.net = nn.Sequential(nn.Linear(input1, input2), nn.ReLU(), nn.Linear(input2, output1))
#         self.res = nn.Sequential(nn.Linear(input1, output1))
#         self.norm = nn.Sequential(nn.LayerNorm(output1))

#     def forward(self, x):
#         x1 = self.net(x)
#         x2 = self.res(x)
#         return self.norm(x1 + x2)


class ResdualBlock(nn.Module):

    def __init__(self, input1, output1):
        super(ResdualBlock, self).__init__()
        if output1>input1:
            self.net = nn.Sequential(nn.Linear(input1, 2*output1-input1), nn.ReLU(), nn.Linear(2*output1-input1, output1))
        else:
            self.net=nn.Sequential(nn.Linear(input1, input1*2), nn.ReLU(), nn.Linear(input1*2, output1))
        self.res = nn.Sequential(nn.Linear(input1, output1))
        self.norm = nn.Sequential(nn.LayerNorm(output1))

    def forward(self, x):
        x1 = self.net(x)
        x2 = self.res(x)
        return self.norm(x1 + x2)




class ResModule2(nn.Module):
    def __init__(self,input_nums,res_layers,extra_n):
        super(ResModule2,self).__init__()
        self.net=nn.ModuleList()
        for i in range(res_layers):
            self.net.append(ResdualBlock2(input_nums,extra_n))
            input_nums+=extra_n
    
    def forward(self,x,extra):
        t=0
        for layer in self.net:
            x=layer(x,extra)
        return x




class TiDE2(nn.Module):

    def __init__(self,y_input,a_input,x_input,x_output,res_layers,res_output,t_output):
        super(TiDE2,self).__init__()
        self.feature_projection=nn.Sequential(nn.Linear(x_input,x_output),nn.LayerNorm(x_output),nn.ReLU())
        self.res_blocks=ResModule2(x_output+y_input+a_input,
        res_layers,
        y_input+x_output+a_input)
        self.res_output=nn.Sequential(nn.Linear((y_input+x_output+a_input)*(res_layers+1)
        ,res_output),nn.LayerNorm(res_output),nn.ReLU())
        self.temporal_decoder=ResdualBlock(res_output+x_output,2*(res_output+x_output),t_output)
        self.look_back=nn.Linear(y_input,t_output)
    
    def forward(self,y,a,x):
        lb=self.look_back(y)
        fp=self.feature_projection(x)
        extra=torch.concat([y,a,fp],dim=-1)
        res=self.res_blocks(extra,extra)
        return self.temporal_decoder(torch.concat([self.res_output(res),fp],dim=-1))+lb


class ResdualBlock(nn.Module):

    def __init__(self, input1, output1):
        super(ResdualBlock, self).__init__()
        self.net = nn.Sequential(nn.Linear(input1, 3*input1-2*output1), nn.ReLU(), nn.Linear(3*input1-2*output1, output1))
        self.res = nn.Sequential(nn.Linear(input1, output1))
        self.norm = nn.Sequential(nn.LayerNorm(output1))

    def forward(self, x):
        x1 = self.net(x)
        x2 = self.res(x)
        return self.norm(x1 + x2)


class ResModule(nn.Module):

    def __init__(self, input_num,layers_num,up_num):
        super(ResModule, self).__init__()
        self.net = nn.ModuleList()
        for i in range(layers_num):
            self.net.append(ResdualBlock(input_num,input_num+up_num))
            input_num+=up_num

    def forward(self, x):
        for layers in self.net:
            x = layers(x)
        return x


class TiDE(nn.Module):

    def __init__(self,y_num,y_out,a_num,x_num,x_out,t1_output,t2_output,res_up,res_layers):
        super(TiDE, self).__init__()
        self.lb1=nn.Sequential(nn.Linear(y_num,y_out),nn.LayerNorm(y_out),nn.ReLU())
        self.lb_1=nn.Linear(y_out,t1_output)
        self.lb_2=nn.Linear(y_out,t2_output)
        self.feature_projection=ResdualBlock(x_num,x_out)
        # print(y_num+a_num+x_out)
        # print(y_num)
        # print(a_num)
        # print(x_out)
        self.res_blocks=ResModule(y_num+a_num+x_out,res_layers,res_up)
        self.t1=ResdualBlock(y_num+a_num+x_out+res_layers*res_up,t1_output)
        self.t2=ResdualBlock(y_num+a_num+x_out+res_layers*res_up,t2_output)

    def forward(self,y,a,x):
        fp = self.feature_projection(x)
        m_input=torch.concat([y,a,fp],dim=-1)
        # print(f'm {m_input.shape}')
        lb=self.lb1(y)
        lb1=self.lb_1(lb)
        lb2=self.lb_2(lb)
        res_out=self.res_blocks(m_input)
        out1=torch.sigmoid(self.t2(res_out))
        out2=torch.sigmoid(self.t1(res_out)+lb1)
        return out1,out2




if __name__=='__main__':
    net=TiDE(180,60,60,90,45,30,15,100,5)
    y=torch.zeros(1,1,180)
    a=torch.zeros(1,1,60)
    x=torch.zeros(1,1,90)
    print(net(y,a,x))

