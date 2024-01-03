from dqnBase import Net,ReplayBuffer
import torch
import torch.nn.functional as F
import numpy as np


# 定义DQN类
class DQN:
    # 构造函数，参数包含环境，学习率，折扣因子，经验回放缓冲区大小，目标网络更新频率
    def __init__(self, env, learning_rate=0.001, gamma=0.99, buffer_size=10000, T=10):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.replay_buffer = ReplayBuffer(max_size=buffer_size)
        self.T = T

        # 判断可用的设备是 CPU 还是 GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 定义Q网络和目标网络，模型结构是一样的
        # print(env.observation_space, env.action_space.n)
        # print(env.observation_space.shape)
        # print(env.observation_space.shape[0])
        # print(env.observation_space.shape[1])
        # # print(env.observation_space.shape[2])
        self.model = Net(env.observation_space.shape[0], env.action_space.n).to(self.device)
        # self.model = Net(2,3).to(self.device)

        self.target_model = Net(env.observation_space.shape[0], env.action_space.n).to(self.device)

        # 初始化时，令目标网络的参数就等于Q网络的参数
        for param, target_param in zip(self.model.parameters(), self.target_model.parameters()):
            target_param.data.copy_(param)

        # 定义Adam优化器
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # 记录模型更新的次数，用于决定何时更新目标模型
        self.update_count = 0
    
    # 根据epsilon-greedy策略选择动作
    def choose_action(self, state, epsilon=0.1):
        if np.random.rand() < epsilon: # 以epsilon的概率随机选择一个动作
            return np.random.randint(self.env.action_space.n)
        else: # 否则选择模型认为最优的动作
            state = torch.FloatTensor(np.array([state])).to(self.device)
            action = self.model(state).argmax().item()
            return action
    
    # 计算损失函数，参数batch为随机采样的一批数据
    def compute_loss(self, batch):
        # 取出数据，并将其转换为numpy数组
        # 然后进一步转换为tensor，并将数据转移到指定计算资源设备上
        states, actions, rewards, next_states, dones = batch
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.tensor(np.array(actions),dtype=torch.int64).view(-1, 1).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).view(-1, 1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).view(-1, 1).to(self.device)

        # 计算当前Q值，即Q网络对当前状态动作样本对的Q值估计
        curr_Q = self.model(states).gather(1, actions)
        # 计算目标网络对下一状态的Q值估计
        next_Q = self.target_model(next_states)
        # 选择下一状态中最大的Q值
        max_next_Q = torch.max(next_Q, 1)[0].view(-1, 1)
        # 计算期望的Q值，若达到终止状态则直接是reward
        expected_Q = rewards + (1 - dones) * self.gamma * max_next_Q
        
        # 计算当前Q值和期望Q值之间的均方误差，返回结果
        loss = torch.mean(F.mse_loss(curr_Q, expected_Q))
        return loss
    
    # 模型更新，参数为批次大小
    def update(self, batch_size):
        # 从经验回放缓冲区中随机采样
        batch = self.replay_buffer.sample(batch_size)
        # 计算这部分数据的损失
        loss = self.compute_loss(batch)

        # 梯度清零、反向传播、更新参数
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 每隔一段时间，更新目标网络的参数
        if self.update_count % self.T == 0:
            for param, target_param in zip(self.model.parameters(), self.target_model.parameters()):
                target_param.data.copy_(param)
        # 记录模型更新的次数
        self.update_count += 1