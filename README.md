# python-games

无



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





项目简介：

别人的项目改的

添加了上边描述的部分，常规玩法，

注意，左键打开格子，右键标记为雷，只能标记，不能取消

鼠标中键是，代码操作部位，利用规则，上边描述部分，自动扫雷，能解决一部分情况



下一步要尝试做的事也是如上所述
