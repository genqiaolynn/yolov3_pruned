# -*- coding:utf-8 -*-
__author__ = 'lynn'
__date__ = '2021/3/26 15:59'


import torch
from pathlib import Path
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from utils.utils import xywh2xyxy


def write_cfg(cfg_path, model_defs):
    with open(cfg_path, 'w') as f:
        for model_def in model_defs:
            f.write(f'[{model_def["type"]}]\n')
            for key, value in model_def.items():
                if key != 'type':
                    f.write(f'{key} = {value}\n')
            f.write('\n')
    return cfg_path


# 这是最保守的一种剪枝方案，但也是实现了高剪枝率
def parse_moudle_defs(module_defs):
    '''
    剪枝原则:
    1. 上采样前面一个CBL不剪
    2. 三个单独的卷积层不剪
    3. 残差单元和残差大组件中的取舍规则是: shortcut层中的起始末尾层不剪
    '''
    CBL_idx = []
    Conv_idx = []
    ignore_idx = set()

    for i, module_def in enumerate(module_defs):
        if module_def['type'] == 'convolutional':
            if module_def['batch_normalize'] == '1':
                CBL_idx.append(i)
            else:
                Conv_idx.append(i)  # 三个单独的卷积
            if module_defs[i+1]['type'] == 'maxpool' and module_defs[i+2]['type'] == 'route':
                # spp前一个CBL不剪，区分tiny和spp
                ignore_idx.add(i)

        # 上采样前的一个CBL不剪
        elif module_def['type'] == 'upsample':
            ignore_idx.add(i - 1)   # upsample前一个CBL不剪
        elif module_def['type'] == 'shortcut':
            ignore_idx.add(i - 1)   # 直连的层
            identity_idx = (i + int(module_def['from']))
            if module_defs[identity_idx]['type'] == 'convolutional':
                ignore_idx.add(identity_idx)
            elif module_defs[identity_idx]['type'] == 'shortcut':
                ignore_idx.add(identity_idx - 1)
    # print(ignore_idx)
    prune_idx = [idx for idx in CBL_idx if idx not in ignore_idx]
    return CBL_idx, Conv_idx, prune_idx


def parse_moudle_defs1(module_defs):
    CBL_idx = []
    Conv_idx = []
    shortcut_idx = dict()
    shortcut_all = set()
    ignore_idx = set()
    for i, module_def in enumerate(module_defs):
        if module_def['type'] == 'convolutional':
            if module_def['batch_normalize'] == '1':
                CBL_idx.append(i)
            else:
                Conv_idx.append(i)  # 三个单独的卷积
            if module_defs[i+1]['type'] == 'maxpool' and module_defs[i+2]['type'] == 'route':
                # spp前一个CBL不剪，区分tiny和spp
                ignore_idx.add(i)

        # 上采样前的一个CBL不剪
        elif module_def['type'] == 'upsample':
            ignore_idx.add(i - 1)   # upsample前一个CBL不剪

        elif module_def['type'] == 'shortcut':
            shortcut_all.add(i - 1)
            identity_idx = i + int(module_def['from'])
            if module_defs[identity_idx]['type'] == 'convolutional':   # CBL
                shortcut_all.add(i)
                shortcut_idx[i - 1] = identity_idx

            elif module_defs[identity_idx]['type'] == 'shortcut':      # 残差大组件中才会出现,残差单元后面那个
                shortcut_all.add(i - 1)
                shortcut_idx[i - 1] = identity_idx - 1

    prune_idx = [idx for idx in CBL_idx if idx not in ignore_idx]
    return CBL_idx, Conv_idx, prune_idx, shortcut_idx, shortcut_all


def parse_moudle_defs_layers(module_defs):
    CBL_idx = []
    Conv_idx = []
    shortcut_idx = []
    for idx, module_def in enumerate(module_defs):
        if module_def['type'] == 'convolutional':
            if module_def['batch_normalize'] == '1':
                CBL_idx.append(idx)
            else:
                Conv_idx.append(idx)
        elif module_def['type'] == 'shortcut':
            shortcut_idx.append(idx - 1)
    return CBL_idx, Conv_idx, shortcut_idx


def updateBN(module_list, s, prune_idx, idx2masks):
    for idx in prune_idx:
        bn_module = module_list[idx][1]   # bn
        bn_module.weight.grad.data.add_(s * torch.sign(bn_module.weight.data))   # torch.sign 的取值范围: [-1, 0, 1]
    if idx2masks:
        # s = s if epoch < 0.5 * epochs else s * 0.01
        for idx in idx2masks:
            bn_module = module_list[idx][1]
            # 85%通道的保持s    (100-85)%的衰减100倍--> 0.01 * s
            # bn_module.weight.grad.data.sub_(0.5 * s * torch.sign(bn_module.weight.data) * (1 - idx2masks[idx].to(device)))
            bn_module.weight.grad.data.sub_(0.99 * s * torch.sign(bn_module.weight.data) * idx2masks[idx])
            # 这句代码的理解:
            # 在torch的张量运算中:sub_是减运算
            # 执行if idx2mask的时候所有通道都是按照原力度压缩了，这里面要去掉衰减部分的通道。
            # idx2mask里面存放的是全部的bn能不能衰减的通道，能剪掉就是0，保留的就是1
            # 这行代码的意思就是全部的通道-0.99 *weight*idx2maks = 0.01衰减的通道 这时候根85%没关系，因为都是1
            # 网络中只剩0.01 * weight(能衰减的通道)
            # 梯度更新时sub_的公式: weight - lr*gradient


def obtain_bn_mask(bn_module, thresh):
    # 获取bn层剪枝的掩膜，1表示保留，0表示剪掉
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    thresh = thresh.to(device)
    # ge(a, b)相当于 a>= b  在bn_module中找出former_bn,返回通道是否需要剪枝的通道状态
    mask = bn_module.weight.data.abs().ge(thresh).float()
    return mask


def get_masks(model, prune_idx, percent):
    bn_weight = gather_bn_weights(model.module_list, prune_idx)
    # weight从小到大排序，前85%按照原力度压缩，剩下的15%进行s衰减100倍
    sorted_bn = torch.sort(bn_weight)[0]
    thresh_index = int(len(sorted_bn) * percent)  # thresh_index  yuzhi index
    thresh = sorted_bn[thresh_index]  # yuzhi

    filter_mask = []
    for idx in prune_idx:
        bn_module = model.module_list[idx][1]
        mask = obtain_bn_mask(bn_module, thresh)   # 这里返回的是85%中需要剪枝的状态
        filter_mask.append(mask.clone())
    idx2masks = {idx: mask for idx, mask in zip(prune_idx, filter_mask)}
    # 这里是全部的bn需要剪枝的状态 0:剪掉 1:保留
    return idx2masks


def gather_bn_weights(module_list, prune_idx):
    size_list = [module_list[idx][1].weight.data.shape[0] for idx in prune_idx]   # 这个list里只有一个元数，因为prune的length=1
    bn_weight = torch.zeros(sum(size_list))   # 设置这么大的矩阵放bn的weight
    index = 0   # zip 是一个元祖
    for idx, size in zip(prune_idx, size_list):
        bn_weight[index:(index + size)] = module_list[idx][1].weight.data.abs().clone()
        index += size
    return bn_weight


def plot_images(imgs, targets, paths=None, fname='images.jpg'):
    # Plots training images overlaid with targets
    imgs = imgs.cpu().numpy()
    targets = targets.cpu().numpy()
    # targets = targets[targets[:, 1] == 21]  # plot only one class

    fig = plt.figure(figsize=(10, 10))
    bs, _, h, w = imgs.shape  # batch size, _, height, width
    bs = min(bs, 16)  # limit plot to 16 images
    ns = np.ceil(bs ** 0.5)  # number of subplots

    for i in range(bs):
        boxes = xywh2xyxy(targets[targets[:, 0] == i, 2:6]).T
        boxes[[0, 2]] *= w
        boxes[[1, 3]] *= h
        plt.subplot(ns, ns, i + 1).imshow(imgs[i].transpose(1, 2, 0))
        plt.plot(boxes[[0, 2, 2, 0, 0]], boxes[[1, 1, 3, 3, 1]], '.-')
        plt.axis('off')
        if paths is not None:
            s = Path(paths[i]).name
            plt.title(s[:min(len(s), 40)], fontdict={'size': 8})  # limit to 40 characters
    fig.tight_layout()
    fig.savefig(fname, dpi=200)
    plt.close()


def save_weights(self, path='model.weights', cutoff=-1):
    # Converts a PyTorch model to Darket format (*.pt to *.weights)
    # Note: Does not work if model.fuse() is applied
    with open(path, 'wb') as f:
        # Write Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        # self.version.tofile(f)  # (int32) version info: major, minor, revision
        # self.seen.tofile(f)  # (int64) number of images seen during training

        # Iterate through layers
        for i, (mdef, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if mdef['type'] == 'convolutional':
                conv_layer = module[0]
                # If batch norm, load bn first
                if mdef['batch_normalize']:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(f)
                    bn_layer.weight.data.cpu().numpy().tofile(f)
                    bn_layer.running_mean.data.cpu().numpy().tofile(f)
                    bn_layer.running_var.data.cpu().numpy().tofile(f)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(f)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(f)


def save_darknet_weights(self, path, cutoff=-1):
    """
        @:param path    - path of the new weights file
        @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
    """
    fp = open(path, "wb")
    self.header_info[3] = self.seen
    self.header_info.tofile(fp)

    # Iterate through layers
    for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
        if module_def["type"] == "convolutional":
            conv_layer = module[0]
            # If batch norm, load bn first
            if module_def["batch_normalize"]:
                bn_layer = module[1]
                bn_layer.bias.data.cpu().numpy().tofile(fp)
                bn_layer.weight.data.cpu().numpy().tofile(fp)
                bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                bn_layer.running_var.data.cpu().numpy().tofile(fp)
            # Load conv bias
            else:
                conv_layer.bias.data.cpu().numpy().tofile(fp)
            # Load conv weights
            conv_layer.weight.data.cpu().numpy().tofile(fp)

    fp.close()
