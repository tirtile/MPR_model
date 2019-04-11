import torch 
from torch import nn
import torch.nn.functional as F

class DWConv(nn.Module):
    def __init__(self, inc, ouc, kernel, stride, padding):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(inc, inc, 3, stride, padding, groups=inc)
        self.bn1 = nn.BatchNorm2d(inc)
        self.relu1 = nn.PReLU(inc)

        self.pwconv = nn.Conv2d(inc, ouc, 1, 1, 0)
        self.bn2 = nn.BatchNorm2d(ouc)
        self.relu2 = nn.PReLU(ouc)

    def forward(self, x):
        x = self.dwconv(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pwconv(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x


class DilateBlock(nn.Module):
    def __init__(self, in_c=16):
        super(DilateBlock, self).__init__()
  
        self.conv1 = DWConv(in_c, in_c, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(kernel_size=1, stride=(2, 2))
        self.pool2 = nn.MaxPool2d(kernel_size=1, stride=(2, 2))
        self.pool3 = nn.MaxPool2d(kernel_size=1, stride=(2, 2))
        self.pool4 = nn.MaxPool2d(kernel_size=1, stride=(2, 2))

        self.sconv1 = DWConv(in_c,in_c,3,1,1)
        self.sconv2 = DWConv(in_c,in_c,3,1,1)
        self.sconv3 = DWConv(in_c,in_c,3,1,1)
        self.sconv4 = DWConv(in_c,in_c,3,1,1)

        self.compress1 = nn.Conv2d(in_c*2, in_c, 1, 1, 0)
        self.compress2 = nn.Conv2d(in_c*2, in_c, 1, 1, 0)
        self.compress3 = nn.Conv2d(in_c*2, in_c, 1, 1, 0)        
        self.compress = nn.Conv2d(in_c*4, in_c, 1, 1, 0)

    def forward(self, x):

        x = self.conv1(x)
        c1 = self.pool1(x)
        c2 = self.pool2(x[:,:,1:,:])
        c3 = self.pool3(x[:,:,:,1:])
        c4 = self.pool4(x[:,:,1:,1:])

        c1 = self.compress(torch.cat((c1, c2, c3, c4),1))

        sc1 = self.sconv1(c1)
        c2 = torch.cat((sc1, c2), 1)
        c2 = self.compress1(c2)
        sc2 = self.sconv2(c2)
        c3 = torch.cat((sc2, c3), 1)
        c3 = self.compress2(c3)
        sc3 = self.sconv3(c3)
        c4 = torch.cat((sc3, c4), 1)
        c4 = self.compress3(c4)
        sc4 = self.sconv4(c4)

        return sc4 

class LCA(nn.Module):
    def __init__(self, classnum):
        super(LCA, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.PReLU(16)
        self.conv2 = DWConv(16,16,3,1,1)

        self.block1 = DilateBlock(16)
        self.block2 = DilateBlock(16)
        self.block3 = DilateBlock(16)
        self.avg_pool = nn.MaxPool2d(kernel_size=8, stride=(2, 2))
        self.conv1x1 = nn.Conv2d(16, 512, 1, 1, 0)
        self.conv1x12 = nn.Conv2d(512, classnum, 1, 1, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        x = self.avg_pool(x)
        x = self.conv1x1(x)
        x = self.conv1x12(x)
        x = x.view(x.size(0), -1)
        return F.log_softmax(x)
