from ppo import PPO
import sys
import numpy as np
from snakeEnv import SnakeEnv
from tqdm import * # 用于显示进度条
def train():
    # 定义超参数
    max_episodes = 10000 # 训练episode数
    max_steps = 200 # 每个回合的最大步数

    env = SnakeEnv()
    # 创建PPO对象
    agent = PPO(env)
    # 定义保存每个回合奖励的列表
    episode_rewards = []

    # 开始循环，tqdm用于显示进度条并评估任务时间开销
    for episode in tqdm(range(max_episodes), file=sys.stdout):
        # 重置环境并获取初始状态
        state, _ = env.reset()
        # 当前回合的奖励
        episode_reward = 0
        # 记录每个episode的信息
        buffer = []

        # 循环进行每一步操作
        for step in range(max_steps):
            # 根据当前状态选择动作
            action = agent.choose_action(state)
            # 执行动作，获取新的信息
            next_state, reward, terminated, truncated, info = env.step(action)
            # 判断是否达到终止状态
            done = terminated or truncated
            
            # 将这个五元组加到buffer中
            buffer.append((state, action, reward, next_state, done))
            # 累计奖励
            episode_reward += reward
            
            # 更新当前状态
            state = next_state

            if done:
                break
        
        # 更新策略
        agent.update(buffer)
        # 记录当前回合奖励值
        episode_rewards.append(episode_reward)
        
        # 打印中间值
        if episode % (max_episodes // 10) == 0:
            tqdm.write("Episode " + str(episode) + ": " + str(episode_reward))

if __name__=="__main__":
    train()