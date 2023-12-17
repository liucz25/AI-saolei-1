# 导入必要的库
import gym
from gym import spaces
import numpy as np
import random

# 创建贪食蛇SnakeEnv类，继承gym.Env
class SnakeEnv(gym.Env):
    # 构造函数，参数为grid_size
    def __init__(self, grid_size = 16):
        super(SnakeEnv, self).__init__()
        
        # 保存网格大小
        self.grid_size = grid_size
        # 蛇的初始位置，设为中心点
        self.snake = [(self.grid_size // 2, self.grid_size // 2)]

        # 定义动作空间，离散的四个值，上下左右
        self.action_space = spaces.Discrete(4)
        # 定义观测空间，是一个grid*grid*3的三维空间，值域为0到1
        self.observation_space = spaces.Box(low=0, high=1, shape=(grid_size, grid_size, 3), dtype=np.uint8)
        
        # 初始化食物变量
        self.food = None
        # 蛇行动的总步数
        self.steps = 0
        # 蛇未吃到食物的步数，可以理解为饥饿度
        self.hungry = 0
        # 上一步的动作，初始化一个特殊值即可
        self.last_action = -10
        # 场上食物的数量
        self.num_food = 10
        # 游戏最大步数
        self.max_steps = 200
    
    # 环境重置函数
    def reset(self):
        # 蛇回到初始位置
        self.snake = [(self.grid_size // 2, self.grid_size // 2)]
        # 食物重新生成
        self.food = self._generate_food()
        # 步数和饥饿度归零
        self.steps = 0
        self.hungry = 0
        # 返回当前状态和一个dict，对齐gym环境中的reset返回值
        return self._get_state(), {}

    # 动作执行函数
    def step(self, action):
        # 确定蛇头的位置
        head = self.snake[0]
        
        # 如果蛇试图做和上一步相反的动作，那么保持上一步的动作不变
        if abs(action - self.last_action) == 2:
            action = self.last_action
        
        # 根据动作决定蛇头的新位置
        if action == 0:   # 上
            new_head = (head[0]-1, head[1])
        elif action == 1: # 右
            new_head = (head[0], head[1]+1)
        elif action == 2: # 下
            new_head = (head[0]+1, head[1])
        else:              # 左
            new_head = (head[0], head[1]-1)
        
        # 记录这一步的动作，并且步数+1
        self.last_action = action
        self.steps += 1

        # 如果蛇头的新位置在蛇身上，游戏结束，返回-100的奖励
        if new_head in self.snake:
            return self._get_state(), -100, True, self.steps>=self.max_steps, {}
        # 如果蛇头的新位置出界了，游戏结束，返回-200的奖励
        if self._is_out_of_bounds(new_head):
            return self._get_state(), -200, True, self.steps>=self.max_steps, {}
        elif new_head in self.food: # 如果新蛇头位置有食物
            # 在蛇体数组的最前面插入一节蛇头的新位置
            self.snake.insert(0, new_head)
            # 重新生成食物
            self.food = self._generate_food()
            # 饥饿度归零
            self.hungry = 0
            # 游戏继续，返回50的奖励
            return self._get_state(), 50, False, self.steps>=self.max_steps, {}
        else: # 没有食物，且游戏也没有触发结束条件
            # 在蛇体数组的最前面插入一节蛇头的新位置
            self.snake.insert(0, new_head)
            # 去掉最后一节
            self.snake.pop()
            
            # 默认奖励为0，饥饿度增加，如果饥饿度达到20，则每步给予一个负奖励
            r = 0
            self.hungry += 1
            if self.hungry >= 20:
                r -= (1 + (self.hungry - 20) / 100)
            
            # 返回对应结果
            return self._get_state(), r, False, self.steps>=self.max_steps, {}
    
    # 从动作空间随机选取一个动作
    def sample(self):
        return self.action_space.sample()
    
    # 渲染游戏界面，返回图像数组
    def render(self, mode='rgb_array'):
        # 初始化画布
        img = np.zeros((self.grid_size, self.grid_size, 3))

        # 绘制蛇体，蛇头是红色，蛇身是蓝色
        for i, s in enumerate(self.snake):
            if i == 0:
                img[s] = [1, 0, 0]
            else:
                img[s] = [0, 0, 1]

        # 食物设为绿色
        for f in self.food:
            img[f] = [0, 1, 0]

        # 返回图像
        return img

    # state直接就是图像数组
    def _get_state(self):
        return self.render()
    
    # 判断某个位置是否出界
    def _is_out_of_bounds(self, position):
        x, y = position
        return x < 0 or y < 0 or x >= self.grid_size or y >= self.grid_size

    # 生成食物
    def _generate_food(self):
        foods = []
        while True:
            # 每次生成一个食物
            food = (random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1))
            # 既不能在蛇身上，也不能在已有的食物上
            if food not in self.snake and food not in foods:
                foods.append(food)
            # 生成数量足够的食物则返回
            if len(foods) >= self.num_food:
                return foods