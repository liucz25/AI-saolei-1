from tkinter import Frame, Label, CENTER
import random
import logic
import constants as c
import numpy as np

def gen():
    return random.randint(0, c.GRID_LEN - 1)

class Puzzle(Frame):

    def __init__(self):
        Frame.__init__(self)
                #初始化
        self.steps = 0
                # 游戏最大步数
        self.max_steps = 2000

        self.grid()
        self.master.title('2048')
        self.master.bind("<Key>", self.key_down)

        self.commands = {
            c.KEY_UP: logic.up,
            c.KEY_DOWN: logic.down,
            c.KEY_LEFT: logic.left,
            c.KEY_RIGHT: logic.right,
            c.KEY_UP_ALT1: logic.up,
            c.KEY_DOWN_ALT1: logic.down,
            c.KEY_LEFT_ALT1: logic.left,
            c.KEY_RIGHT_ALT1: logic.right,
            c.KEY_UP_ALT2: logic.up,
            c.KEY_DOWN_ALT2: logic.down,
            c.KEY_LEFT_ALT2: logic.left,
            c.KEY_RIGHT_ALT2: logic.right,
        }

        self.grid_cells = []
        self.init_grid()
        self.matrix = logic.new_game(c.GRID_LEN)
        self.history_matrixs = []
        self.update_grid_cells()

        # self.mainloop()
    def init_grid(self):
        background = Frame(self, bg=c.BACKGROUND_COLOR_GAME,width=c.SIZE, height=c.SIZE)
        background.grid()

        for i in range(c.GRID_LEN):
            grid_row = []
            for j in range(c.GRID_LEN):
                cell = Frame(
                    background,
                    bg=c.BACKGROUND_COLOR_CELL_EMPTY,
                    width=c.SIZE / c.GRID_LEN,
                    height=c.SIZE / c.GRID_LEN
                )
                cell.grid(
                    row=i,
                    column=j,
                    padx=c.GRID_PADDING,
                    pady=c.GRID_PADDING
                )
                t = Label(
                    master=cell,
                    text="",
                    bg=c.BACKGROUND_COLOR_CELL_EMPTY,
                    justify=CENTER,
                    font=c.FONT,
                    width=5,
                    height=2)
                t.grid()
                grid_row.append(t)
            self.grid_cells.append(grid_row)

    def update_grid_cells(self):
        for i in range(c.GRID_LEN):
            for j in range(c.GRID_LEN):
                new_number = self.matrix[i][j]
                if new_number == 0:
                    self.grid_cells[i][j].configure(text="",bg=c.BACKGROUND_COLOR_CELL_EMPTY)
                else:
                    self.grid_cells[i][j].configure(
                        text=str(new_number),
                        bg=c.BACKGROUND_COLOR_DICT[new_number],
                        fg=c.CELL_COLOR_DICT[new_number]
                    )
        self.update_idletasks()

    def key_down(self, event):
        key = event.keysym
        print(event)
        if key == c.KEY_QUIT: exit()
        if key == c.KEY_BACK and len(self.history_matrixs) > 1:
            self.matrix = self.history_matrixs.pop()
            self.update_grid_cells()
            print('back on step total step:', len(self.history_matrixs))
        elif key in self.commands:
            self.matrix, done = self.commands[key](self.matrix)
            if done:
                self.matrix = logic.add_two(self.matrix)
                # record last move
                self.history_matrixs.append(self.matrix)
                self.update_grid_cells()
                if logic.game_state(self.matrix) == 'win':
                    self.grid_cells[1][1].configure(text="You", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
                    self.grid_cells[1][2].configure(text="Win!", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
                if logic.game_state(self.matrix) == 'lose':
                    self.grid_cells[1][1].configure(text="You", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
                    self.grid_cells[1][2].configure(text="Lose!", bg=c.BACKGROUND_COLOR_CELL_EMPTY)

    def generate_next(self):
        index = (gen(), gen())
        while self.matrix[index[0]][index[1]] != 0:
            index = (gen(), gen())
        self.matrix[index[0]][index[1]] = 2
    def doneDo(self,done):
        if done:
                self.matrix = logic.add_two(self.matrix)
                # record last move
                self.history_matrixs.append(self.matrix)
                self.update_grid_cells()


    def handle_action(self,action):
  
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

class Game:
    def __init__(self,Width=640,Height=480):
        self.puzzle = Puzzle()
        self.reset()        
        self.input_n = 16
        self.output_n = 3
    
    def reset(self):
        # init game state
        self.score = 0

        self.puzzle = Puzzle()

        self.total_step = 0
        self.reward = 0
    def play_step(self, action):

        game_over = False
        self.reward = 0


                     
        self.total_step += 1

        self.puzzle.handle_action(action)
        if self.total_step> 100:
            game_over=True
        if logic.game_state(self.puzzle.matrix) == 'lose':
            self.reward=-100
        # 如果输赢了游戏，游戏结束，返回1000的奖励
        elif logic.game_state(self.puzzle.matrix) == 'win':
            self.reward=1000
            game_over=True
        elif logic.game_state(self.puzzle.matrix) == 'not over1':
            self.reward=2
        elif logic.game_state(self.puzzle.matrix) == 'not over2':
            self.reward=4
        elif logic.game_state(self.puzzle.matrix) == 'not over3':
            self.reward=6
        elif logic.game_state(self.puzzle.matrix) == 'not over4':
            self.reward=8
        else:
            self.reward=1
        self.score=self.get_score(self.puzzle.matrix)
        # print(self.score)
        self.puzzle.update_grid_cells()
 
        return self.reward, game_over, self.score
        
    def get_score(self,matrix):
        s=0
        for i in matrix:
            for j in i:
                s+= j
        return s        

    

    def get_state(self):
        state = self.puzzle.matrix
        # print(np.array(state, dtype=int).flatten())

       
        return np.array(state, dtype=int).flatten()