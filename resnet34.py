import torch
import torch.nn as nn
class residual_block(nn.Module):
    def __init__(self,in_channel,out_channel,stride=1,shortcut=None):
        super(residual_block,self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=(3,3),stride=stride,padding=(1,1),bias=False),
            nn.BatchNorm2d(num_features=out_channel),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channel,out_channels=out_channel,kernel_size=(3,3),stride=1,padding=(1,1),bias=False),
            nn.BatchNorm2d(num_features=out_channel)
        )
        self.shortcut=shortcut
    def forward(self,x):
        shortcut=x
        x=self.conv(x)
        if self.shortcut is not None:
            shortcut=self.shortcut(shortcut)
        x=x+shortcut
        x=torch.relu(x)
        return x

class residual_part(nn.Module):
    def __init__(self,in_channel,out_channel,num_block=3,stride=1):
        super(residual_part,self).__init__()
        self.blocks=nn.ModuleList()
        self.num_block=num_block
        self.shortcut=nn.Sequential(
            nn.Conv2d(in_channel,out_channel,kernel_size=(1,1),stride=stride,padding=(0,0),bias=False),
            nn.BatchNorm2d(out_channel)
        )
        self.blocks.append(residual_block(in_channel,out_channel,stride,self.shortcut))
        for i in range(self.num_block-1):
            self.blocks.append(residual_block(out_channel,out_channel))
    def forward(self,x):
        for i in range(self.num_block):
            x=self.blocks[i](x)
        return x

class resnet34(nn.Module):
    def __init__(self,in_channel=3):
        super(resnet34,self).__init__()
        self.pre=nn.Sequential(
            nn.Conv2d(in_channels=in_channel,out_channels=64,kernel_size=(7,7),stride=(2,2),padding=(3,3),bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3,3),stride=(2,2),padding=(1,1))
        )
        in_channel=64
        self.hidden_channel=[64,128,256,512]
        self.hidden_num_block=[3,4,6,3]
        self.parts=nn.ModuleList()
        self.num_part=4
        for i in range(self.num_part):
            self.parts.append(residual_part(in_channel=in_channel,out_channel=self.hidden_channel[i],num_block=self.hidden_num_block[i],stride=(1 if i==0 else 2)))
            in_channel=self.hidden_channel[i]
        self.pool=nn.AdaptiveAvgPool2d(1)
    def forward(self,x):
        x=self.pre(x)
        for i in range(self.num_part):
            x=self.parts[i](x)
        x=self.pool(x)
        x=x.view(x.size(0),-1)
        return x
    
if __name__=="__main__":
    model=resnet34(in_channel=4)
    x=torch.randn(5,4,84,84) # -> torch.Size([5, 512, 84//16, 84//16])
    y=model(x)
    print(y.shape) # torch.Size([5, 512])