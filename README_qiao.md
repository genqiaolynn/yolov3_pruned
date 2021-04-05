这个项目是基于https://github.com/eriklindernoren/PyTorch-YOLOv3 修改的
1. 支持基础训练: 将sr设置成False就可
2. 支持稀疏训练的三种策略:
设置sr=True   prune=0是正常剪枝，剪枝率比较低
    sr=True   prune=1是极限剪枝
2021年3.31上午10:56 现在支持正常剪枝，剪枝率比较低的那个