'''
未考虑的情况
1.标记与打开同一个格子，这个需要互斥
2.先标记，标记之后，打开别的格子连带将标记的打开

利用规则扫雷成功了：
利用规则，实验性的，添加标记格子更多信息，例如 周围未打开格子数，周围标记格子数，差值，等
第一步，找到周围剩余未开格子数，等于，格子标记数（周围雷数），把这些标记为雷
更新数据
第二步
找到，已经展示出来的数字网格，找到已经满足条件，周围标记为雷的格子数 等于 格子标记数（周围雷数），，并在这些表格中，继续找到，差值不为空的，打开这些格子

重复，第1，第2步

可以解决大部分问题，不是所有

下一步是找更复杂的规则，这个应该需要很清晰的思路，应该能找一些，不过应该找不齐

下一步还可以，利用强化学习，找找规则，
    这个方式，需要继续改，第一步，把这个扫雷变成环境，类似动作空间，状态空间的东西，
    然后仿照，贪吃蛇强化学习的方式，完善强化学习框架
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
        self.around_unopen_grid_count=0
        self.around_unopen_grid_list=[]
        self.around_mark_grid_count=0
        self.around_mark_grid_list=[]
        self.unopen_mark_grid_list=[]

    def __repr__(self):
        re_str = "该格子的矩阵坐标： (" + str(self.y)+','+str(self.x)+')'+'\n'
        re_str += "该格子的状态： " + str(self.state)+'\n'
        re_str += "该格子是否是地雷：" + str(self.is_mine)+'\n'
        re_str += "该格子周围地雷数：" + str(self.around_mine_count)+'\n'
        re_str += "该格子周围 未打开 格子数：" + str(self.around_unopen_grid_count)+'\n'
        re_str += "该格子周围 未打开 格子列表：" + str(self.around_unopen_grid_list)+'\n'
        re_str += "该格子周围标记地雷数：" + str(self.around_mark_grid_count)+'\n'
        re_str += "该格子周围标记地雷列表：" + str(self.around_mark_grid_list)+'\n'
        re_str +='\n'        
        re_str += "该格子周围标记地雷于打开格子差值数：" + str(self.unopen_mark_grid_list)+'\n'
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
        self.open_update_grid_state()
        # self.mark_grid_as_mine()
        
        return True

    def mark_grid(self, x, y):  # 标记格子
        if (self.grids[x][y].get_state() == GridState.normal):
            if self.LEFT_FLAG_COUNT <= 0:
                return
            else:
                self.grids[x][y].set_state(GridState.flag)
                self.LEFT_FLAG_COUNT -= 1
                
                # #更新周围格子状态
                # self.mark_update_grid_state(x,y)
                # self.find_diff_from_unopen_mark(x,y)

        # elif (self.grids[x][y].get_state() == GridState.flag):
        #     self.grids[x][y].set_state(GridState.ask)
        #     self.LEFT_FLAG_COUNT = min(self.LEFT_FLAG_COUNT + 1, self.MINE_COUNT)
                            
        #     #更新周围格子状态
        #     self.mark_update_grid_state(x,y)
        #     self.find_diff_from_unopen_mark(x,y)
        # elif (self.grids[x][y].get_state() == GridState.ask):
        #     self.grids[x][y].set_state(GridState.normal)
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

    def set_unopen_grid_count_list(self,x,y):
        self.grids[x][y].around_unopen_grid_count=0
        self.grids[x][y].around_unopen_grid_list=[]
        #单个网格计算周围打开网格个数及列表
        for i in range(max(0, x-1),  min(self.WIDTH, x+2)):
            for j in range(max(0, y-1), min(self.HEIGHT, y+2)):
                if (self.grids[i][j].get_state() == GridState.normal):
                    # print(x,y,i,j)
                    self.grids[x][y].around_unopen_grid_count+=1
                    self.grids[x][y].around_unopen_grid_list.append([i,j])
                elif (self.grids[i][j].get_state() == GridState.flag):
                    # print(x,y,i,j)
                    self.grids[x][y].around_unopen_grid_count+=1
                    self.grids[x][y].around_unopen_grid_list.append([i,j])


    def open_update_grid_state(self):
        #遍历网格，挨个计算
        for i in range(self.WIDTH):
            for j in range(self.HEIGHT):
                self.set_unopen_grid_count_list(i,j)


    def mark_update_grid_state(self,x,y):

        #根据被标记网格，把周围环绕的表格，遍历一下

        self.grids[x][y].around_mark_grid_count=0
        self.grids[x][y].around_mark_grid_list=[]
        for i in range(max(0, x-1),  min(self.WIDTH, x+2)):
            for j in range(max(0, y-1), min(self.HEIGHT, y+2)):
                # print("s")
                if self.grids[i][j].get_state() == GridState.flag:
                    self.grids[x][y].around_mark_grid_count+=1
                    self.grids[x][y].around_mark_grid_list.append([i,j]) 

    def mark_update_all_grid_state(self):
         for i in range(self.WIDTH):
            for j in range(self.HEIGHT):
                self.mark_update_grid_state(i,j)
  
    
    def find_diff_from_unopen_mark_sigle(self,x,y):
        # result=[[x,y] for [x,y] in self.grids[x][y].around_unopen_grid_list if [x,y] not in self.grids[x][y].around_mark_grid_list]
        res=[]
        for i1,j1 in self.grids[x][y].around_unopen_grid_list:
            res.append([i1,j1])
        # print(res)
        for i,j in self.grids[x][y].around_mark_grid_list:
            if self.grids[i][j].get_state() == GridState.flag:
                if [i,j] in res:
                    res.remove([i,j])

        return res

    def find_diff_from_unopen_mark(self):
        self.open_update_grid_state()
        self.mark_update_all_grid_state()
        for i in range(self.WIDTH):
            for j in range(self.HEIGHT):
                self.grids[i][j].unopen_mark_grid_list=self.find_diff_from_unopen_mark_sigle(i,j)

    def mark_grid_as_mine(self):
         for i in range(self.WIDTH):
            for j in range(self.HEIGHT):
                if self.grids[i][j].state==GridState.opened and self.grids[i][j].around_mine_count!=0:
                    if self.grids[i][j].around_mine_count== self.grids[i][j].around_unopen_grid_count:
                        # print()
                        for x,y in self.grids[i][j].around_unopen_grid_list:
                            # print (x,y)
                            if self.grids[x][y].state==GridState.normal:
                                self.mark_grid(x,y)

    def after_mark_open_grid(self):

        self.find_diff_from_unopen_mark()
        #思路
        #找所有已经打开的数字格
        #在这些格中找，数字大小已经被满足的格子
        #在这个更小的范围中找，最后一项差值属性不为空的 格子（就是要被打开的格子），

        want_open_list=[]
        for i in range(self.WIDTH):
            for j in range(self.HEIGHT):
                if self.grids[i][j].state==GridState.opened and self.grids[i][j].around_mine_count>0:#已经打开的数字格
                    if  self.grids[i][j].around_mark_grid_count>=self.grids[i][j].around_mine_count:#数字大小已经被满足的格子
                        # print(self.grids[i][j])
                        if len(self.grids[i][j].unopen_mark_grid_list)>0:#差值属性不为空的
                            # print(self.grids[i][j]) 
                            for [x1,y1] in self.grids[i][j].unopen_mark_grid_list:
                                if [x1,y1] not in want_open_list:
                                    want_open_list.append([x1,y1])
        print(want_open_list)
        for x2,y2 in want_open_list:
            self.opened_grid(x2,y2)
                            #     if self.grids[x1][y1].state!=GridState.flag:
                            #         print(x1,y1)
                            #     # self.opened_grid(x1,y1)