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

# 定义简单神经网络
class Net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Net, self).__init__()
        self.input_dim = input_dim*input_dim*3 # 网络的输入维度
        self.output_dim = output_dim # 网络的输出维度
        
        # 定义一个仅包含全连接层的网络，激活函数使用ReLU函数
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, self.output_dim)
        )
    
    # 定义前向传播，输出动作Q值
    def forward(self, state):
        action = self.fc(state)
        return action
    
# 经验回放缓冲区
class ReplayBuffer:
    # 构造函数，max_size是缓冲区的最大容量
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = collections.deque(maxlen = self.max_size)  # 用collections的队列存储，先进先出

    # 添加experience（五元组）到缓冲区
    def add(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    # 从buffer中随机采样，数量为batch_size
    def sample(self, batch_size):
        # print(batch_size)
        batch = random.sample(self.buffer, batch_size)
        # b=self.buffer.view(-1)
        # c=RandomSampler(b,num_samples=5)
        # d=[i for i in c]
        state, action, reward, next_state, done = zip(*batch)
        # state, action, reward, next_state, done = d
        return state, action, reward, next_state, done
    
    # 返回缓冲区数据数量
    def __len__(self):
        return len(self.buffer)