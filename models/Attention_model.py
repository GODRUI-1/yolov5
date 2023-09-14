import torch
import torch.nn as nn
import numpy as np
import torchvision
import math

def Sigmoid(x):
    y = 1/(1+np.exp(-x))
    return y

class CA_Block(nn.Module):
    def __init__(self, channel, reduction=16):
      # reduction 是CA中设计用来减少注意力机制的参数量
        super(CA_Block, self).__init__()
        
        self.conv_1x1 = nn.Conv2d(in_channels = channel, 
                                  out_channels = channel//reduction,
                                  kernel_size=1,stride=1,bias=False)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel//reduction)
        
        self.F_h = nn.Conv2d(in_channels = channel//reduction, 
                             out_channels = channel, kernel_size = 1,stride=1,
                             bias=False)
        self.F_w = nn.Conv2d(in_channels = channel//reduction, 
                             out_channels = channel, kernel_size = 1,stride=1,
                             bias=False)
        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()
                       
        # self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        # self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        # mip = max(8, inp // groups)

        # self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        # self.bn1 = nn.BatchNorm2d(mip)
        # self.conv2 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        # self.conv3 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        # self.relu = h_swish()

    def forward(self, x):
        # x,即特征层，包含四个部分，batch_size,c,h,w 
        _,_,h,w,=x.size()
        
        # batch_size,c,h,w => batch_size,c,h,1 => batch_sizem,c,1,h
        x_h = torch.mean(x,dim = 3, keepdim = True).permute(0, 1, 3, 2)
        # batch_size,c,h,w => batch_size,c,1,w
        x_w = torch.mean(x,dim = 2, keepdim = True)
        
        # 这里是对h通道进行转置  序号为3和2的通道上进行了转置，并完成了堆叠
        # 接下来进行卷积标准化
        
        # batch_size,c,h,w => batch_size,c,1,w => batch_size, c, 1, w+h
        # batch_size,c,1,w + h => batch_size, c / r, 1, w+h
        # 接下来是卷积标准化和激活函数
        x_cat_conv_relu = self.relu(self.bn(self.conv_1x1(torch.cat((x_h, x_w), 3))))
                
        # batch_size, c / r, 1, w+h => batch_size, c / r, 1, h & batch_size, c / r, 1, w
        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h,w],3)
        
        # batch_size, c / r, 1, h => batch_size, c , h, 1
        # 这里是因为之前进行了转置， 要转回来（把h放回序号为2的通道位置）
        # 代表高方向上每个通道的特征值
        
        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0,1,3,2)))
        # batch_size, c / r, 1, w => batch_size, c , 1, w
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))
        
        # batch_size, c , h, w
        out = x * s_h.expand_as(x) * s_w.expand_as(x)
        
        # #以下是另一种代码的描述
        # identity = x
        # n,c,h,w = x.size()
        # x_h = self.pool_h(x)
        # x_w = self.pool_w(x).permute(0, 1, 3, 2)

        # y = torch.cat([x_h, x_w], dim=2)
        # y = self.conv1(y)
        # y = self.bn1(y)
        # y = self.relu(y) 
        # x_h, x_w = torch.split(y, [h, w], dim=2)
        # x_w = x_w.permute(0, 1, 3, 2)

        # x_h = self.conv2(x_h).sigmoid()
        #x_w = self.conv3(x_w).sigmoid()
        # x_h = x_h.expand(-1, -1, h, w)
        # x_w = x_w.expand(-1, -1, h, w)

        # out = identity * x_w * x_h

        return out
    

class SEnet_Block(nn.Module):
    def __init__(self, channel, ratio = 16):
        super(SEnet_Block, self).__init__()
        self.avg_pool   = nn.AdaptiveAvgPool2d(1)
        self.fc         = nn.Sequential(
            nn.Linear(channel, channel//ratio, False),
            nn.ReLU(),
            nn.Linear(channel//ratio, channel, False),
            torch.nn.Sigmoid(),
        )
    def forward(self,x):
        b, c, h, w = x.size()
        # b, c, h, w, -> b, c, 1,c1,
        avg = self.avg_pool(x).view([b, c])

        #b, c -> b, c //ratio ->b, c -> b, c, 1, 1
        fc = self.fc(avg).view([b, c, 1, 1])
        # print(fc)
        return x * fc


class CBAM_Block(nn.Module):
    def __init__(self,in_channel,reduction=16,kernel_size=7):
        super(CBAM_Block, self).__init__()
        #通道注意力机制
        self.max_pool=nn.AdaptiveMaxPool2d(output_size=1)
        self.avg_pool=nn.AdaptiveAvgPool2d(output_size=1)
        self.mlp=nn.Sequential(
            nn.Linear(in_features=in_channel,out_features=in_channel//reduction,bias=False),
            nn.ReLU(),
            nn.Linear(in_features=in_channel//reduction,out_features=in_channel,bias=False)
        )
        self.sigmoid=nn.Sigmoid()
        #空间注意力机制
        self.conv=nn.Conv2d(in_channels=2,out_channels=1,kernel_size=kernel_size ,stride=1,padding=kernel_size//2,bias=False)

    def forward(self,x):
        #通道注意力机制
        maxout=self.max_pool(x)
        maxout=self.mlp(maxout.view(maxout.size(0),-1))
        avgout=self.avg_pool(x)
        avgout=self.mlp(avgout.view(avgout.size(0),-1))
        channel_out=self.sigmoid(maxout+avgout)
        channel_out=channel_out.view(x.size(0),x.size(1),1,1)
        channel_out=channel_out*x
        #空间注意力机制
        max_out,_=torch.max(channel_out,dim=1,keepdim=True)
        mean_out=torch.mean(channel_out,dim=1,keepdim=True)
        out=torch.cat((max_out,mean_out),dim=1)
        out=self.sigmoid(self.conv(out))
        out=out*channel_out
        return out

class ECA_Block(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(ECA_Block, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


# model = CBAM_Block(channel=512)
# print(model)
# input = torch.ones([2,512,26,26])
# outputs = model(input)
# x = torch.randn(1,1024,32,32)
# net = CBAM_Block(1024)
# y = net.forward(x)
# print(y.shape)