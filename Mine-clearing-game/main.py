import pygame
import time
from pygame.locals import *
from include.Mine_clear_class import *

# 基础屏幕大小、格子块大小等全局信息。
FIRST_SCREEN_WIDTH = 720
FIRST_SCREEN_HEIGHT = 720
SIZE = 40 # 一个正方形格子的大小

WIDTH = 10 # 长宽多少个格子
HEIGHT = 10
MINE_COUNT = 10     # 地雷数

def Load_image():
    # 把图片和笑脸大小声明为全局变量
    global face_size
    global first_bac,img_normal,img_flag,img_ask,img_mine,img_explode,img_wrong_mine,img_face_lose,img_face_normal,img_face_win
    global img_mine_count
    # first background
    first_bac = pygame.image.load('./data/pictures/first_bac.png').convert() # 加载并渲染
    first_bac = pygame.transform.smoothscale(first_bac, (FIRST_SCREEN_WIDTH, FIRST_SCREEN_HEIGHT))  # 平滑缩放（这里可以用学过的最近邻？或双线性）
    # open 状态 around_mines_count
    img_amc0 = pygame.image.load('./data/pictures/0.bmp').convert()
    img_amc0 = pygame.transform.smoothscale(img_amc0, (SIZE, SIZE))
    img_amc1 = pygame.image.load('./data/pictures/1.bmp').convert()
    img_amc1 = pygame.transform.smoothscale(img_amc1, (SIZE, SIZE))
    img_amc2 = pygame.image.load('./data/pictures/2.bmp').convert()
    img_amc2 = pygame.transform.smoothscale(img_amc2, (SIZE, SIZE))
    img_amc3 = pygame.image.load('./data/pictures/3.bmp').convert()
    img_amc3 = pygame.transform.smoothscale(img_amc3, (SIZE, SIZE))
    img_amc4 = pygame.image.load('./data/pictures/4.bmp').convert()
    img_amc4 = pygame.transform.smoothscale(img_amc4, (SIZE, SIZE))
    img_amc5 = pygame.image.load('./data/pictures/5.bmp').convert()
    img_amc5 = pygame.transform.smoothscale(img_amc5, (SIZE, SIZE))
    img_amc6 = pygame.image.load('./data/pictures/6.bmp').convert()
    img_amc6 = pygame.transform.smoothscale(img_amc6, (SIZE, SIZE))
    img_amc7 = pygame.image.load('./data/pictures/7.bmp').convert()
    img_amc7 = pygame.transform.smoothscale(img_amc7, (SIZE, SIZE))
    img_amc8 = pygame.image.load('./data/pictures/8.bmp').convert()
    img_amc8 = pygame.transform.smoothscale(img_amc8, (SIZE, SIZE))
    img_mine_count = { # 字典数据结构
        0: img_amc0,
        1: img_amc1,
        2: img_amc2,
        3: img_amc3,
        4: img_amc4,
        5: img_amc5,
        6: img_amc6,
        7: img_amc7,
        8: img_amc8
    }
    # 格子状态图：normal, flag, ask
    img_normal = pygame.image.load('./data/pictures/normal.bmp').convert()
    img_normal = pygame.transform.smoothscale(img_normal, (SIZE, SIZE))
    img_flag = pygame.image.load('./data/pictures/flag.bmp').convert()
    img_flag = pygame.transform.smoothscale(img_flag, (SIZE, SIZE))
    img_ask = pygame.image.load('./data/pictures/ask.bmp').convert()
    img_ask = pygame.transform.smoothscale(img_ask, (SIZE, SIZE))
    # 雷的状态
    img_mine = pygame.image.load('./data/pictures/mine.bmp').convert() # normal->mine , ask也默认这个
    img_mine = pygame.transform.smoothscale(img_mine, (SIZE, SIZE))
    img_explode = pygame.image.load('./data/pictures/explode.bmp').convert() # open->mine
    img_explode = pygame.transform.smoothscale(img_explode, (SIZE, SIZE))
    img_wrong_mine = pygame.image.load('./data/pictures/wrong_mine.bmp').convert() # flag->mine
    img_wrong_mine = pygame.transform.smoothscale(img_wrong_mine, (SIZE, SIZE))
    # 经典笑脸 ready和start , lose , win
    face_size = int(SIZE * 1.5) # 笑脸大小
    img_face_lose = pygame.image.load('./data/pictures/face_lose.bmp').convert()
    img_face_lose = pygame.transform.smoothscale(img_face_lose, (face_size, face_size))
    img_face_normal = pygame.image.load('./data/pictures/face_normal.bmp').convert()
    img_face_normal = pygame.transform.smoothscale(img_face_normal, (face_size, face_size))
    img_face_win = pygame.image.load('./data/pictures/face_win.bmp').convert()
    img_face_win = pygame.transform.smoothscale(img_face_win, (face_size, face_size))

def Load_font():
    # first 一层；second 二层
    global first_font, second_font
    first_font_size = FIRST_SCREEN_WIDTH//10
    # 'C:/Windows/Fonts/xxx.ttf'  # （系统自带字体文件夹，xxx是其名字）
    first_font_path = './data/font/云峰静龙行书.ttf'
    # pygame.font.Font参数(字体位置, 字体大小)
    first_font = pygame.font.Font(first_font_path, first_font_size)

    second_font_path = './data/font/kaiti.ttf'
    second_font = pygame.font.Font(second_font_path, SIZE*2)  # 得分的字体

def Set_clock():
    global clock, FPS
    FPS = 60
    # 获取一个Clock对象，限制游戏的运行速度（clock.tick(FPS)这样用）
    clock = pygame.time.Clock() # 类似于sleep
def Set_colors():
    global pink_color, white_color
    pink_color = (245, 140, 190)  # (定义RGB三元组)这个还可以在调淡色一点
    white_color = (255, 255, 255)
def My_init():
    Load_image()
    Load_font()
    Set_clock()
    Set_colors()

def Print_text(screen, font, x, y, text, fcolor=(0, 0, 0)):
    # font.render参数 (文本,抗锯齿否,字体颜色,背景颜色)
    imgText = font.render(text, True, fcolor)
    screen.blit(imgText, (x, y))

def Set_easy_mode():
    # easy无需改参数，使用默认的
    global WIDTH, HEIGHT, MINE_COUNT
    game_screen = pygame.display.set_mode((WIDTH * SIZE, (HEIGHT + 2) * SIZE)) # 因为要放笑脸，所以长一点
    return game_screen
def Set_medium_mode():
    global WIDTH, HEIGHT, MINE_COUNT
    WIDTH = WIDTH*2
    HEIGHT = HEIGHT*2
    MINE_COUNT = MINE_COUNT*2
    game_screen = pygame.display.set_mode((WIDTH * SIZE, (HEIGHT + 2) * SIZE)) # 因为要放笑脸，所以长一点
    return game_screen
def Set_hard_mode():
    global WIDTH, HEIGHT, MINE_COUNT
    WIDTH = WIDTH*3
    HEIGHT = HEIGHT*3
    MINE_COUNT = MINE_COUNT*3
    game_screen = pygame.display.set_mode((WIDTH * SIZE, (HEIGHT + 2) * SIZE)) # 因为要放笑脸，所以长一点
    return game_screen

def matrixTOscreen(matrix_x, matrix_y):
    screen_x = matrix_y * SIZE
    screen_y = (matrix_x+2) * SIZE
    return screen_x, screen_y
def screenTOmatrix(screen_x, screen_y):
    matrix_x = screen_y // SIZE - 2
    matrix_y = screen_x // SIZE
    return matrix_x, matrix_y

def Secend_game_interface(screen):
    global WIDTH, HEIGHT, MINE_COUNT
    # 定义出游戏实例，定义游戏状态
    chess_board = ChessBoard(WIDTH, HEIGHT, MINE_COUNT)
    game_state = GameStae.ready

    face_pos = ((WIDTH*SIZE - face_size) // 2, (SIZE * 2 - face_size) // 2) # 笑脸位置
    (num_width, num_height) = second_font.size("999")

    start_time = time.time()
    duration_time = 0 # 游戏时长
    # screen.blit(first_bac, (0, 0)) # 载入游戏开始界面
    while True:
        # 大概顺序是操作，判断输赢，图片展示
        # 获取事件，包括鼠标和键盘
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # 需要回调参数
                WIDTH = 10 # 长宽多少个格子
                HEIGHT = 10
                MINE_COUNT = 10     # 地雷数
                print("成功退出")
                return
            elif event.type == pygame.MOUSEBUTTONDOWN: # 鼠标按下(任何一键)
                screen_x, screen_y = event.pos # 获取鼠标点击位置的pygame坐标
                matrix_x, matrix_y = screenTOmatrix(screen_x, screen_y) # 格子的矩阵坐标
                grid = chess_board.get_one_grid(matrix_x, matrix_y)

                mouse_l, mouse_m, mouse_r = pygame.mouse.get_pressed()# 检测具体哪个键(bool型)（左，滚轮，右）
                # 打破ready状态，进入start，而且放在这里是个按下左键就开始的意思。
                # 但是不跳出本次点击动作，可以顺带执行打开操作
                if game_state == GameStae.ready:
                    game_state = GameStae.start
                    start_time = time.time()
                if mouse_l == True:
                    # 优先级强判定
                    if face_pos[0] <= screen_x <= face_pos[0] + face_size \
                        and face_pos[1] <= screen_y <= face_pos[1] + face_size:
                        # 开始或重新开始游戏
                        del chess_board
                        chess_board = ChessBoard(WIDTH, HEIGHT, MINE_COUNT)
                        game_state = GameStae.ready
                        continue
                    if grid.get_state() == GridState.normal \
                        and (game_state == GameStae.ready or game_state == GameStae.start):
                        lose_flag = chess_board.opened_grid(matrix_x, matrix_y)
                        if(lose_flag == False):
                            game_state = GameStae.lose
                            break # 跳出事件循环
                        else:
                            if(chess_board.get_opened_grid_count() == WIDTH*HEIGHT - MINE_COUNT):
                                # 赢的条件1.打开所有非雷的格子
                                game_state = GameStae.win
                                break # 跳出事件循环

                if mouse_r == True \
                    and (game_state == GameStae.ready or game_state == GameStae.start):
                    chess_board.mark_grid(matrix_x,matrix_y)
                    if(chess_board.get_flag_mine_count() == MINE_COUNT):
                        # 赢的条件2.标记完全部地雷
                        game_state = GameStae.win
                        break # 跳出事件循环

                if mouse_m == True:
                    chess_board.mark_update_all_grid_state()
                    chess_board.mark_grid_as_mine()
                    chess_board.after_mark_open_grid()
                    # 显示调试信息
                    print(grid)
            # elif event.type == pygame.KEYUP:
            #     print(event.type)
            #     screen_x, screen_y = event.pos # 获取鼠标点击位置的pygame坐标
            #     matrix_x, matrix_y = screenTOmatrix(screen_x, screen_y) # 格子的矩阵坐标
            #     grid = chess_board.get_one_grid(matrix_x, matrix_y)
            #     chess_board.find_diff_from_unopen_mark(matrix_x,matrix_y)
            # print("down"+str(pygame.KEYDOWN))
            # print("up"+ str(pygame.KEYUP))
        
        # 下面是各种游戏状态的显示逻辑了
        screen.fill(white_color) # 这句出现了很多次可能多余了
        if game_state == GameStae.ready:
            duration_time = int(0) # 游戏时长
        if game_state == GameStae.start:
            duration_time = int(time.time()- start_time) # 游戏时长
        Print_text(screen, second_font, int((1.5*WIDTH*SIZE-num_width)*0.5), int((2*SIZE-num_height)*0.5), '%03d' % (duration_time), pink_color)

        left_flags_count = chess_board.get_left_flag_count()
        Print_text(screen, second_font, int((0.5*WIDTH*SIZE-num_width)*0.5), int((2*SIZE-num_height)*0.5), '%03d' % (left_flags_count), pink_color)

        if game_state == GameStae.ready:
            screen.blit(img_face_normal, face_pos)
            for i in range(WIDTH):
                for j in range(WIDTH):
                    pos = matrixTOscreen(i, j)
                    screen.blit(img_normal, pos)

        if game_state == GameStae.start:
            screen.blit(img_face_normal, face_pos)
            
            for i in range(WIDTH):
                for j in range(WIDTH):
                    # 开始的时候无需关心地雷此时的状态
                    grid = chess_board.get_one_grid(i, j)
                    pos = matrixTOscreen(i, j)

                    if grid.get_state() == GridState.normal:
                        screen.blit(img_normal, pos)
                    elif grid.get_state() == GridState.opened:
                        screen.blit(img_mine_count[grid.get_around_mine_count()], pos)
                    elif grid.get_state() == GridState.flag:
                        screen.blit(img_flag, pos)
                    elif grid.get_state() == GridState.ask:
                        screen.blit(img_ask, pos)
                    else:
                        print(game_state)
                        print(grid)
                        exit("something error unknown grid state")

        elif game_state == GameStae.lose:
            screen.blit(img_face_lose, face_pos)
            for i in range(WIDTH):
                for j in range(WIDTH):
                    # 这里需要加上对地雷的考虑了
                    grid = chess_board.get_one_grid(i, j)
                    pos = matrixTOscreen(i, j)

                    if grid.get_state() == GridState.normal and grid.get_is_mine() == False:
                        screen.blit(img_normal, pos)
                    elif grid.get_state() == GridState.normal and grid.get_is_mine() == True:
                        screen.blit(img_mine, pos)
                    elif grid.get_state() == GridState.opened and grid.get_is_mine() == False:
                        screen.blit(img_mine_count[grid.get_around_mine_count()], pos)
                    elif grid.get_state() == GridState.opened and grid.get_is_mine() == True:
                        screen.blit(img_explode, pos)
                    elif grid.get_state() == GridState.flag and grid.get_is_mine() == False :#标记错了
                        screen.blit(img_wrong_mine, pos)
                    elif grid.get_state() == GridState.flag and grid.get_is_mine() == True :#标记对了
                        screen.blit(img_flag, pos)
                    elif grid.get_state() == GridState.ask and grid.get_is_mine() == False:
                        screen.blit(img_ask, pos)
                    elif grid.get_state() == GridState.ask and grid.get_is_mine() == True:
                        screen.blit(img_mine, pos)
                    else:
                        print(game_state)
                        print(grid)
                        exit("something error unknown grid state")

        elif game_state == GameStae.win:
            screen.blit(img_face_normal, face_pos)
            for i in range(WIDTH):
                for j in range(WIDTH):
                    # 这里需要加上对地雷的考虑了
                    grid = chess_board.get_one_grid(i, j)
                    pos = matrixTOscreen(i, j)

                    if grid.get_state() == GridState.normal:
                        screen.blit(img_normal, pos)
                    elif grid.get_state() == GridState.opened:
                        screen.blit(img_mine_count[grid.get_around_mine_count()], pos)
                    elif grid.get_state() == GridState.flag:
                        screen.blit(img_flag, pos)
                    elif grid.get_state() == GridState.ask:
                        screen.blit(img_ask, pos)
                    else:
                        print(game_state)
                        print(grid)
                        exit("something error unknown grid state")

        # else:
            # do nothing

        pygame.display.update()
        clock.tick(FPS)
    
# 简单介绍一下模块内各种函数的作用，具体的可以用时现查
# 把各种加载操作放到别出去，看代码的时候可以直接缩小他们。
if __name__ == "__main__":  # main
    pygame.init()  # 检查硬件调用接口、基础功能是否有问题（模糊一点就说初始化）
    # pygame.display.set_mode参数(分辨率, 显示模式=0, 颜色位数=0, 不知道=0, 不知道=0)
    menu_screen = pygame.display.set_mode((FIRST_SCREEN_WIDTH, FIRST_SCREEN_HEIGHT))
    My_init() # 自定义的初始化函数
    pygame.display.set_caption("扫雷") # 设置当前窗口标题
    # screen.blit参数(图像/字,位置,不知道,不知道)
    # 第三个参数好像与动态图有关，可能其他程序会用得到。
    menu_screen.blit(first_bac, (0, 0)) # 载入游戏开始界面
    # 之后要用这些参数定位鼠标动作，所以单起一个“稍短”的名字
    (text_width, text_height) = first_font.size("游戏难度")
    text_x = int((FIRST_SCREEN_WIDTH-text_width)/2)
    base_text_y = int(FIRST_SCREEN_HEIGHT/6)
    Print_text(menu_screen, first_font, text_x, base_text_y, "游戏难度", pink_color)
    Print_text(menu_screen, first_font, text_x, base_text_y*2, "简单模式", pink_color)
    Print_text(menu_screen, first_font, text_x, base_text_y*3, "中单模式", pink_color)
    Print_text(menu_screen, first_font, text_x, base_text_y*4, "困单模式", pink_color)
    reset_mode = False # 不会挑来跳去，便用一个bool变量代替。
    while True:
        # 获取事件，包括鼠标和键盘
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit("ok")
            # 暂时还想不到窗口最大化的好办法
            elif event.type == pygame.MOUSEBUTTONDOWN: # 鼠标按下(任何一键)
                screen_x, screen_y = event.pos # 获取鼠标点击位置的pygame坐标
                mouse_l, mouse_m, mouse_r = pygame.mouse.get_pressed()# 检测具体哪个键(bool型)（左，滚轮，右）
                if mouse_l == True:
                    # \ 表示承接上一行，因为一行写不开了。
                    if text_x <= screen_x <= text_x + text_width \
                        and base_text_y*2 <= screen_y <= base_text_y*2 + text_height:
                        game_screen = Set_easy_mode()
                        Secend_game_interface(game_screen)
                        # 退出来之后需要重新刷新屏幕。
                        reset_mode = True
                        # 这里要是有goto跳来跳去岂不美哉。但怕用多了自己也掌控不了它跳到哪里去了>_<

                    elif text_x <= screen_x <= text_x + text_width \
                        and base_text_y*3 <= screen_y <= base_text_y*3 + text_height:
                        game_screen = Set_medium_mode()
                        Secend_game_interface(game_screen)
                        reset_mode = True

                    elif text_x <= screen_x <= text_x + text_width \
                        and base_text_y*4 <= screen_y <= base_text_y*4 + text_height:
                        game_screen = Set_hard_mode()
                        Secend_game_interface(game_screen)
                        reset_mode = True

            if reset_mode == True:
                menu_screen = pygame.display.set_mode((FIRST_SCREEN_WIDTH, FIRST_SCREEN_HEIGHT))
                menu_screen.blit(first_bac, (0, 0)) # 载入游戏开始界面
                Print_text(menu_screen, first_font, text_x, base_text_y, "游戏难度", pink_color)
                Print_text(menu_screen, first_font, text_x, base_text_y*2, "简单模式", pink_color)
                Print_text(menu_screen, first_font, text_x, base_text_y*3, "中单模式", pink_color)
                Print_text(menu_screen, first_font, text_x, base_text_y*4, "困单模式", pink_color)
                reset_mode = False

        pygame.display.update()
        clock.tick(FPS)


# 此时一个正常扫雷该有的功能都具备了
# 接下来可以实现一些打趣的功能
# 1.将随机点开的第一个一直变为雷，直接输掉游戏
# 2.点完第一个，直接赢得游戏
# 这也好改可以直接改变输赢条件的逻辑。
