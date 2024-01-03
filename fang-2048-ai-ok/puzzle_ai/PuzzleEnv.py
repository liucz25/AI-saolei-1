# 导入必要的库
import gym
from gym import spaces
import numpy as np
import random

import logic
import constants as c
def gen():
    return random.randint(0, c.GRID_LEN - 1)
# 创建贪食蛇SnakeEnv类，继承gym.Env
class PuzzleEnv(gym.Env):
    # 构造函数，参数为grid_size
    def __init__(self):
        super(PuzzleEnv, self).__init__()
        
        # 保存网格大小
        self.grid_size = 4
        # 蛇的初始位置，设为中心点
        self.snake = [(self.grid_size // 2, self.grid_size // 2)]

        # 定义动作空间，离散的四个值，上下左右
        self.action_space = spaces.Discrete(4)
        # 定义观测空间，是一个grid*grid*3的三维空间，值域为0到1
        self.observation_space = spaces.Box(low=0, high=1, shape=(16,1), dtype=np.uint8)
        # print(self.action_space)
        # print(self.observation_space)

        #初始化
        self.steps = 0
                # 游戏最大步数
        self.max_steps = 2000
        self.grid_cells = []
        self.init_grid()
        self.matrix = logic.new_game(c.GRID_LEN)
        self.history_matrixs = []
        self.update_grid_cells()
        

    def init_grid(self):
        for i in range(c.GRID_LEN):
            grid_row = []
            for j in range(c.GRID_LEN):
                cell =""
                grid_row.append(cell)
            self.grid_cells.append(grid_row)

    def update_grid_cells(self):
        for i in range(c.GRID_LEN):
            for j in range(c.GRID_LEN):
                new_number = self.matrix[i][j]
                if new_number == 0:
                    self.grid_cells[i][j]=""
                else:
                    self.grid_cells[i][j]=str(new_number)
    def generate_next(self):
        index = (gen(), gen())
        while self.matrix[index[0]][index[1]] != 0:
            index = (gen(), gen())
        self.matrix[index[0]][index[1]] = 2

    # 环境重置函数
    def reset(self):
        self.grid_cells = []
        self.init_grid()
        self.matrix = logic.new_game(c.GRID_LEN)
        self.history_matrixs = []
        # 返回当前状态和一个dict，对齐gym环境中的reset返回值
        return self._get_state(), {}
    
    def doneDo(self,done):
        if done:
                self.matrix = logic.add_two(self.matrix)
                # record last move
                self.history_matrixs.append(self.matrix)
                self.update_grid_cells()



    # 动作执行函数
    def step(self, action):

        
        # 根据动作决定执行函数
        if action == 0:   # 上
            self.matrix, done = logic.up(self.matrix)
            self.doneDo(done)
            
        elif action == 1: # 右
            self.matrix, done = logic.right(self.matrix)
            self.doneDo(done)
        elif action == 2: # 下
            self.matrix, done = logic.down(self.matrix)
            self.doneDo(done)
        else:              # 左
           self.matrix, done = logic.left(self.matrix)
           self.doneDo(done)
        
        # 记录这一步的动作，并且步数+1
        self.last_action = action
        self.steps += 1


        # if logic.game_state(self.matrix) == 'win':
        #     self.grid_cells[1][1]="You Win!"
        # 如果输了游戏，游戏结束，返回-100的奖励
        if logic.game_state(self.matrix) == 'lose':
            return self._get_state(), -100, True, self.steps>=self.max_steps, {}
            self.grid_cells[1][1]="You Lose!"
        # 如果输赢了游戏，游戏结束，返回1000的奖励
        elif logic.game_state(self.matrix) == 'win':
            return self._get_state(), 1000, True, self.steps>=self.max_steps, {}
            self.grid_cells[1][1]="You Win!"
        elif logic.game_state(self.matrix) == 'not over1':
            return self._get_state(), 2, True, self.steps>=self.max_steps, {}
        elif logic.game_state(self.matrix) == 'not over2':
            return self._get_state(), 4, True, self.steps>=self.max_steps, {}
        elif logic.game_state(self.matrix) == 'not over3':
            return self._get_state(), 6, True, self.steps>=self.max_steps, {}
        elif logic.game_state(self.matrix) == 'not over4':
            return self._get_state(), 8, True, self.steps>=self.max_steps, {}
        else:
            return self._get_state(), 1, True, self.steps>=self.max_steps, {}
    
    # 从动作空间随机选取一个动作
    def sample(self):
        return self.action_space.sample()
    
    # 渲染游戏界面，返回图像数组
    def render(self, mode='rgb_array'):
        # print(self.matrix)
        # print(self.grid_cells)
        return self.matrix

    # state直接就是图像数组
    def _get_state(self):
        return self.render()
    

