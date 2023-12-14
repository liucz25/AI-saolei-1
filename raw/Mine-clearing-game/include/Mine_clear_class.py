'''
未考虑的情况
1.标记与打开同一个格子，这个需要互斥
2.先标记，标记之后，打开别的格子连带将标记的打开
'''
# _*_ coding : utf-8 _*_
from enum import Enum
import random

# 坐标说明，本文件内坐标均是“矩阵坐标”即在矩阵中(x,y)表示x行y列。

class GridState(Enum):  # 格子状态
    normal = 1 	 	# 初始化的状态，未被打开，未被标记
    opened = 2    	# 已打开的格子
    flag = 3    	# 标记为旗子
    ask = 4     	# 标记为问号
class GameStae(Enum):  # 游戏状态
    ready = 1,		# 不稳定，会立即被start状态取代
    start = 2,
    lose = 3,
    win = 4

class Grid:  # 一个小格子
    def __init__(self, x=0, y=0, is_mine=False):
        self.x = x
        self.y = y
        self.is_mine = is_mine
        self.around_mine_count = 0  # around_mine_count
        self.state = GridState.normal

    def __repr__(self):
        re_str = "该格子的矩阵坐标： (" + str(self.y)+','+str(self.x)+')'+'\n'
        re_str += "该格子的状态： " + str(self.state)+'\n'
        re_str += "该格子是否是地雷：" + str(self.is_mine)+'\n'
        re_str += "该格子周围地雷数：" + str(self.around_mine_count)
        return re_str

    def get_x(self): return self.x
    def set_x(self, x): self.x = x

    def get_y(self): return self.y
    def set_y(self, y): self.y = y

    def get_is_mine(self): return self.is_mine
    def set_is_mine(self, is_mine): self.is_mine = is_mine

    def get_around_mine_count(self): return self.around_mine_count
    def set_around_mine_count(self, around_mine_count): self.around_mine_count = around_mine_count

    def get_state(self): return self.state
    def set_state(self, state): self.state = state


class ChessBoard:  # 游戏棋盘和一些影响棋盘的操作
    def __init__(self, WIDTH=10, HEIGHT=10, MINE_COUNT=10):
        # 生成一个棋盘
        self.WIDTH = WIDTH
        self.HEIGHT = HEIGHT
        self.MINE_COUNT = MINE_COUNT
        self.LEFT_FLAG_COUNT = MINE_COUNT
        self.grids = [[Grid(i, j) for i in range(WIDTH)]for j in range(HEIGHT)]
        # 放置地雷
        # random.sample 返回一个MINE_COUNT长的列表，内存放k个随机的唯一的元素。
        # “//”代表取整
        for i in random.sample(range(WIDTH * HEIGHT), MINE_COUNT):
            x = i // WIDTH
            y = i % WIDTH
            self.grids[x][y].set_is_mine(True)
            # 计算周围地雷数，即地雷周围每一个格子的around mine count数加一
            for m in range(max(0, x-1), min(WIDTH, x+2)):
                for n in range(max(0, y-1), min(HEIGHT, y+2)):
                    self.grids[m][n].set_around_mine_count(self.grids[m][n].get_around_mine_count()+1)

    def get_all_grids(self): return self.grids
    def get_one_grid(self, x, y): return self.grids[x][y]

    # 后续需要依据游戏模式（简单、困难之类）才能更改这些变量。
    def get_WIDTH(self): return self.WIDTH
    def get_HEIGHT(self): return self.HEIGHT
    def get_MINE_COUNT(self): return self.MINE_COUNT

    def opened_grid(self, x, y):  # 打开格子
        self.grids[x][y].set_state(GridState.opened)
        if self.grids[x][y].get_is_mine() == True:  # 踩到雷了
            return False

        # 没踩到，打开周围3*3格子
        if self.grids[x][y].get_around_mine_count() == 0:
            for i in range(max(0, x-1),  min(self.WIDTH, x+2)):
                for j in range(max(0, y-1), min(self.HEIGHT, y+2)):
                    if self.grids[i][j].get_state() == GridState.normal:
                        self.opened_grid(i, j)
        
        return True

    def mark_grid(self, x, y):  # 标记格子
        if (self.grids[x][y].get_state() == GridState.normal):
            if self.LEFT_FLAG_COUNT <= 0:
                return
            else:
                self.grids[x][y].set_state(GridState.flag)
                self.LEFT_FLAG_COUNT -= 1
        elif (self.grids[x][y].get_state() == GridState.flag):
            self.grids[x][y].set_state(GridState.ask)
            self.LEFT_FLAG_COUNT = min(self.LEFT_FLAG_COUNT + 1, self.MINE_COUNT)
        elif (self.grids[x][y].get_state() == GridState.ask):
            self.grids[x][y].set_state(GridState.normal)
        # 对open状态啥也不干

    # 赢的方式1.打开所有非雷的格子
    # 赢的方式2.标记所有雷的格子
    # （逻辑判断均在主程序中实现）
    def get_opened_grid_count(self):  # 统计所有打开的格子数
        res_count = 0
        for i in range(self.WIDTH):
            for j in range(self.HEIGHT):
                if self.grids[i][j].get_state() == GridState.opened:
                    res_count += 1
        return res_count
    
    def get_flag_mine_count(self):  # 统计所有标记对的格子数
        # \ 是下一行接到这一行的意思，可以用括号代替。
        res_count = 0
        for i in range(self.WIDTH):
            for j in range(self.HEIGHT):
                if self.grids[i][j].get_state() == GridState.flag \
                    and self.grids[i][j].get_is_mine() == True:
                    res_count += 1
        return res_count
    
    def get_left_flag_count(self): return self.LEFT_FLAG_COUNT
    
