这个项目是基于https://github.com/eriklindernoren/PyTorch-YOLOv3 修改的
* 支持基础训练: 将sr设置成False就可
* 支持稀疏训练的三种策略:
设置sr=True   prune=0是正常剪枝，剪枝率比较低
    sr=True   prune=1是极限剪枝
2021年3.31上午10:56 现在支持各种剪枝

```angular2html
prune.py 支持通道剪枝，剪枝率低的那个
```

```angular2html
slim_prune.py 通道剪枝策略3，需要微调，是我试验最好的
```

```angular2html
layer_prune.py 层剪枝，这个可以砍16个，需要微调
```

```angular2html
layer_channel_prune.py 这个是在前面基础上做的，这个里的参数都是前面调好的参数放过来的，经验值，需要微调，微调之后可以升点
```
