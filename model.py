import torch
import torch.nn as nn
import torch.nn.functional as F

class Self_Attention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Self_Attention, self).__init__()
        self.query = nn.Conv2d(in_channels, out_channels, 1)
        self.key = nn.Conv2d(in_channels, out_channels, 1)
        self.value = nn.Conv2d(in_channels, out_channels, 1)
        self.final = nn.Conv2d(out_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()
        query = self.query(x).view(B, -1, H*W)
        key   = self.key(x).view(B, -1, H*W)
        value = self.value(x).view(B, -1, H*W)

        attn = torch.bmm(query.permute(0,2,1), key)
        attn = F.softmax(attn, dim=-1)

        out = torch.bmm(value, attn.permute(0,2,1))
        out = out.view(B, -1, H, W)
        out = self.final(out)
        return self.gamma * out + x

class CNNWithAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.att1  = Self_Attention(32, 8)
        self.pool1 = nn.MaxPool2d(2,2)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.att2  = Self_Attention(64, 16)
        self.pool2 = nn.MaxPool2d(2,2)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.att3  = Self_Attention(128, 32)
        self.pool3 = nn.MaxPool2d(2,2)

        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.att4  = Self_Attention(256, 64)
        self.pool4 = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.att1(x)

        x = self.pool2(F.relu(self.conv2(x)))
        x = self.att2(x)

        x = self.pool3(F.relu(self.conv3(x)))
        x = self.att3(x)

        x = self.pool4(F.relu(self.conv4(x)))
        x = self.att4(x)

        x = x.view(x.size(0), -1)
        return self.fc(x)
