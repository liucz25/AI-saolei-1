# 导入必要的库
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import sys
import time
import random
import collections
from tqdm import * # 用于显示进度条

# 策略模型，给定状态生成各个动作的概率
class PolicyModel(nn.Module):
    def __init__(self, grid_size, output_dim):
        super(PolicyModel, self).__init__()
        self.grid_size = grid_size
        
        # 定义卷积层
        self.conv = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=1), # 使用1x1卷积升维
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 4, kernel_size=1), # 使用1x1卷积降维
            nn.BatchNorm2d(4),
            nn.ReLU()
        )
        
        # 使用全连接层构建一个简单的神经网络，ReLU作为激活函数
        # 最后加一个Softmax层，使得输出可以看作是概率分布
        self.fc = nn.Sequential(
            nn.Flatten(), # 将特征图展开为一维向量
            nn.Linear(grid_size * grid_size * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Softmax(dim = 1)
        )

    # 定义前向传播，输出动作概率分布
    def forward(self, x):
        x = x.view(-1, 3, self.grid_size, self.grid_size)
        x = self.conv(x)
        action_prob = self.fc(x)
        return action_prob

# 价值模型，给定状态估计价值
class ValueModel(nn.Module):
    def __init__(self, grid_size):
        super(ValueModel, self).__init__()
        self.grid_size = grid_size
        
        # 定义卷积层
        self.conv = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=1), # 使用1x1卷积升维
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 4, kernel_size=1), # 使用1x1卷积降维
            nn.BatchNorm2d(4),
            nn.ReLU()
        )
        
        # 网络结构和策略模型类似，输出维度为动作空间的维度
        self.fc = nn.Sequential(
            nn.Flatten(), # 将特征图展开为一维向量
            nn.Linear(grid_size * grid_size * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    # 定义前向传播，输出价值估计
    def forward(self, x):
        x = x.view(-1, 3, self.grid_size, self.grid_size)
        x = self.conv(x)
        value = self.fc(x)
        return value