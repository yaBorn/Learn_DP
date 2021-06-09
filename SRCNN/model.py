"""
    SRCNN 模型结构
"""
import torch.nn as nn
import torch.nn.functional as F


class SRCNN(nn.Module):
    def __init__(self, channels_number=1):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(channels_number, 64, 9, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(64, 32, 1, stride=1, padding=0, bias=False)  # 25=33+1-9
        self.conv3 = nn.Conv2d(32, channels_number, 5, stride=1, padding=0, bias=False)  # 21=25+1-5

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = self.conv3(out)
        return out