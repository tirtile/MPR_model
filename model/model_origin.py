import torch 
from torch import nn
import torch.nn.functional as F


class DilateBlock(nn.Module):
    def __init__(self, in_c=16):
        super(DilateBlock, self).__init__()
  
        self.conv1 = nn.Conv2d(in_c,in_c,3,1,1)
        self.bn1 = nn.BatchNorm2d(in_c)
        self.relu1 = nn.PReLU(in_c)

        self.pool1 = nn.MaxPool2d(kernel_size=1, stride=(2, 2))
        self.pool2 = nn.MaxPool2d(kernel_size=1, stride=(2, 2))
        self.pool3 = nn.MaxPool2d(kernel_size=1, stride=(2, 2))
        self.pool4 = nn.MaxPool2d(kernel_size=1, stride=(2, 2))

        self.sconv1 = nn.Conv2d(in_c,in_c,3,1,1)
        self.sconv2 = nn.Conv2d(in_c,in_c,3,1,1)
        self.sconv3 = nn.Conv2d(in_c,in_c,3,1,1)
        self.sconv4 = nn.Conv2d(in_c,in_c,3,1,1)
        
        self.conv = nn.Conv2d(in_c*4, in_c*4, 3, 1, 1)
        self.compress = nn.Conv2d(in_c*4, in_c, 1, 1, 0)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        c1 = self.pool1(x)
        c2 = self.pool2(x[:,:,1:,:])
        c3 = self.pool3(x[:,:,:,1:])
        c4 = self.pool4(x[:,:,1:,1:])

        sc1 = self.sconv1(c1)
        sc2 = self.sconv2(c2)
        sc3 = self.sconv3(c3)
        sc4 = self.sconv4(c4)
        
        sc = torch.cat((sc1, sc2, sc3, sc4), 1)

        sc = self.conv(sc)
        shuffle_sc = self.compress(sc)
        return shuffle_sc 

class LCA(nn.Module):
    def __init__(self, classnum):
        super(LCA, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.PReLU(16)

        self.conv2 = nn.Conv2d(16, 16, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.PReLU(16)

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
        x = self.bn2(x)
        x = self.relu2(x)
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        x = self.avg_pool(x)
        x = self.conv1x1(x)
        x = self.conv1x12(x)
        x = x.view(x.size(0), -1)

        return F.log_softmax(x)
