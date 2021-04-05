# -*- coding:utf-8 -*-
__author__ = 'lynn'
__date__ = '2021/3/30 15:52'


import torch, time, traceback
from models import Darknet
import numpy as np
from copy import deepcopy
from test import evaluate
from utils.utils import load_classes
from prune.prune_tools import parse_moudle_defs1, gather_bn_weights, obtain_bn_mask, write_cfg, save_darknet_weights
import torch.nn.functional as F
from terminaltables import AsciiTable


# 先以全局阈值找出各卷积层的mask，然后对于每组shortcut，它将相连的各卷积层的剪枝mask取并集
# 用merge后的mask进行剪枝，这样对每一个相关层都做了考虑。同时它还对每一层的保留通道做了限制
# 对激活偏置层添加了处理，降低剪枝时的精度损失
# TODO 原理
# 1. 先以全局阈值找出各卷积层的mask
# 2. 然后对于每组shortcut
# 3. 它将相连的各卷积层的剪枝mask取并集
# 4. 用merge后的mask进行剪枝，这样对每一个相关层都做了考虑
# 5. 同时它还对每一个层的保留通道做了限制，实验中它的剪枝效果最好


hyp0 = {
    'img_size': 1120,
    'batch_size': 1,
    'conf_thresh': 0.5,
    'iou_thresh': 0.5,
    'nms_thresh': 0.5,
    'global_percent': 0.5,   # 这个值是我稀疏训练剪枝不掉点的阈值，也就是通道剪枝的策略1，最保守的那种剪枝方案得到的阈值
    'layer_keep': 0.1,
    'cfg_path': 'config/yolov3.cfg',
    'weight_path': 'checkpoints/yolov3_ckpt_245_s.pth',
    'train_path': 'data/math_blank/train.txt',
    'test_path': 'data/math_blank/test.txt',
    'prune_num': 16,  # 剪掉16块的意思
}
print(hyp0)

classes = load_classes('data/math_blank/class.names')


def prune_and_eval(model, sorted_bn, percent, bn_weights, prune_idx):
    # 这个percent是个理论剪枝率，这个{1 - remain_num / len(sorted_bn):.4f}是个实际剪枝率
    model_copy = deepcopy(model)
    thresh_index = int(len(sorted_bn) * percent)
    thresh = bn_weights[thresh_index]
    print(f'gamma value less than {thresh:.4f} are set to zero')
    remain_num = 0
    for idx in prune_idx:
        bn_module = model_copy.module_list[idx][1]
        mask = obtain_bn_mask(bn_module, thresh)
        remain_num += int(mask.sum())
        bn_module.weight.data.mul_(mask)   # weight = weight * mask
    print('test current model')

    # with torch.no_grad():
    #     precision, recall, AP1, f1, ap_class = evaluate(
    #         model_copy,
    #         path=hyp0['test_path'],
    #         iou_thres=hyp0['iou_thresh'],
    #         conf_thres=hyp0['conf_thresh'],
    #         nms_thres=hyp0['nms_thresh'],
    #         img_size=hyp0['img_size'],
    #         batch_size=hyp0['batch_size'],
    #     )
    # print(f'number of channels has been reduced from {remain_num / len(sorted_bn):.4f}')
    # print(f'prund ratio {1 - remain_num / len(sorted_bn):.4f}')
    # print(f'mAP of the pruned model is {AP1.mean():.4f}')
    return thresh


def get_filters_mask_channel(model, threshold, CBL_idx, prune_idx):
    # CBL_idx 是全部的CBL索引(72)    prune_idx不包括upsample前一个CBL(70)
    # 1. 先以全局阈值找出各卷积层的mask
    # 2. 然后对于每组shortcut
    # 3. 它将相连的各卷积层的剪枝mask取并集
    # 4. 用merge后的mask进行剪枝，这样对每一个相关层都做了考虑
    # 5. 同时它还对每一个层的保留通道做了限制，实验中它的剪枝效果最好
    # return: 返回原模型每个conv层要保留的数据，以及对应的掩膜

    # TODO layer_keep: 0.01(不清楚原因)   global_percent: 0.9    global_percent按照理论剪枝率设置
    pruned = 0
    total = 0
    num_filters = []
    filters_mask = []
    for idx in CBL_idx:
        bn_module = model.module_list[idx][1]
        if idx in prune_idx:
            weight_copy = bn_module.weight.data.abs().clone()
            channels = weight_copy.shape[0]   # bn的输入通道数量
            min_channel_num = int(channels * hyp0['layer_keep']) if int(channels * hyp0['layer_keep']) > 0 else 1
            mask = weight_copy.gt(threshold).float()   # 各卷积层的mask,这里应该是0,1这样的数，shape大小不改变
            # mask_sum = int(mask.sum())
            # print(mask_sum)

            if int(torch.sum(mask)) < min_channel_num:
                _, sort_bn_weight = torch.sort(weight_copy, descending=True)   # 从大到小排序
                mask[sort_bn_weight[:min_channel_num]] = 1.   # 一维张量的切割
            remain = int(mask.sum())
            pruned = pruned + mask.shape[0] - remain    # 总的剪掉剩下的
            print(f'layer index: {idx:>3d} \t total channel: {mask.shape[0]:>4d} \t remaining channnel {remain:>4d}')
        else:
            mask = torch.ones(bn_module.weight.data.shape)
            remain = mask.shape[0]
        total = total + mask.shape[0]
        num_filters.append(remain)
        filters_mask.append(mask.clone())
    pruned_ratio = pruned / total
    print(f'pruned channels: {pruned:4d} \t pruned ratio: {pruned_ratio:4f}')
    return num_filters, filters_mask


def update_activation(i, pruned_model, activation, CBL_idx):
    # 这里的i是针对整个网络而言的，所以i可以直接+1
    next_id = i + 1
    # 判断下一层是不是卷积层，是的话就更新，将beta值移植进去
    if pruned_model.module_defs[next_id]['type'] == 'convolutional':
        next_conv = pruned_model.module_list[next_id][0]   # next_conv的torch.size [in_channel, out_channel, 3, 3]
        conv_sum = next_conv.weight.data.sum(dim=(2, 3))
        # matmul方法中如果第一个参数或者第二个参数是1维的,它会提升该参数为矩阵(根据另一个参数维数,给该参数增加一个为1的维数).
        # 矩阵相乘之后会将为1的维数去掉,所以这里可以不用reshape扩维以及缩维
        # 即(64,32).matmul(32) -> (64,32).matmul(32,1) -> (64,1) -> (64)
        # offset = conv_sum.matmul(activation.reshape(-1, 1)).reshape(-1)
        # 如果这一层是卷积层，则offset就是卷积和*激活函数，不管有没有BN，所以bn的判断不在这里
        offset = conv_sum.matmul(activation)   # bias = conv2[relu(beta1*(1-mask))]
        if next_id in CBL_idx:  # 这里判断这个next_id是不是含有bn
            next_bn = pruned_model.module_list[next_id][1]
            next_bn.running_mean.data.sub_(offset)   # means - conv2(beta1*(1-mask))
        else:
            next_conv.bias.data.add_(offset)          # bias + conv2(beta1*(1-mask))


def beta2next(model, prune_idx, CBL_idx, CBL2mask):
    # 先科普下bn: https://blog.csdn.net/hjimce/article/details/50866313
    """
    该方法主要是将待剪枝层bn的β值做一下处理,然后用它的下一层的conv层的bn中的mean或者conv层中的bias来吸收这个处理值
    同时把把被移植conv中的γ与β置为0
    例: 第n层 y = ReLu{BN1[CONV1(x)]}   第n+1层 z = ReLu{BN2[CONV2(y)]} 假设如果有BN和ReLu的话
    -> y = ReLU(γ1 * [(x - mean1) / std1] + β1)
    -> z = ReLU(γ2 * [(y - mean2) / std1] + β2)
    第n层剪枝后(将待剪枝通道的γ置为0)y就分为了两部分,保留下来的γ与β 与γ置为0的β 设该层的剪枝掩膜为mask 1保留 0剪掉
    -> y = ReLU(γ1*mask * [(x - mean1) / std1] + β1*mask) + ReLU(0*(1-mask) * [(x - mean1) / std1] + β1*(1-mask))
    -> y = ReLU(γ1*mask * [(x - mean1) / std1] + β1*mask) + ReLU(β1*(1-mask))
    ->   = y' + ReLU(β1*(1-mask))

    所以第n+1层 z = ReLU(γ2 * [(CONV2(y') + CONV2(ReLU(β1*(1-mask))) - mean2) / std2] + β2)
    带入上面的式子我们可以发现,在保证第二层采用同样的计算方式和结果不变的情况下:令 mean2' = mean2 - CONV2(ReLU(β1*(1-mask)))
    -> z = ReLU(γ2 * [CONV2(y') - mean2'] / std2 + β2)

    同理,如果第n+1层是无bn的conv层的话,z = CONV2(y') + bias  令 bias' = bias+CONV2(ReLU(β1*(1-mask)))
    -> z = CONV2(y') + bias'
    :param model:       原始稀疏化训练后的模型
    :param prune_idx:   待剪枝的conv层索引,根据剪枝方式不同,可剪枝的层也不同
    :param CBL_idx:     有bn层的conv索引,YOLO层前一层除外
    :param CBLidx2mask: CBL_idx中conv层对应的剪枝掩膜 1保留 0剪掉
    :return: 处理后的剪枝模型
    """
    activations = []
    pruned_model = deepcopy(model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for i, model_def in enumerate(pruned_model.module_defs):   # 这个i是针对网络中全部的module
        if model_def['type'] == 'convolutional':
            # 这里对激活值进行一次初始化,以防当前层不被剪枝
            activation = torch.zeros(int(model_def['filters'])).to(device)     # 激活的通道和conv的通道一样多
            if i in prune_idx:
                mask = torch.from_numpy(CBL2mask[i]).to(device)
                bn_module = pruned_model.module_list[i][1]
                # 将当前gamma置为0,CBL是一个组合，需要一起处理
                bn_module.weight.data.mul_(mask)
                activation = F.leaky_relu((1-mask) * bn_module.bias.data, 0.1)   # 激活函数中不能剪枝的置为0.1
                # 将当前层的beta值移植到下一层中, update_activation的操作
                update_activation(i, pruned_model, activation, CBL_idx)
                # 下一层吸收了当前bn层中的beta之后，将当前层beta=0，防止对以后产生影响
                bn_module.bias.data.mul_(mask)
            activations.append(activation)

        elif model_def['type'] == 'shortcut':
            activ1 = activations[i - 1]
            from_layer = int(model_def['from'])
            activ2 = activations[from_layer + i]
            activation = activ1 + activ2
            update_activation(i, pruned_model, activation, CBL_idx)
            activations.append(activation)

        elif model_def['type'] == 'route':
            # spp不参与剪枝，其中的route不用更新，仅占位
            from_layers = [int(s) for s in model_def['layers'].split(',')]
            activation = None
            if len(from_layers) == 1:
                activation = activations[i + from_layers[0]]
                update_activation(i, pruned_model, activation, CBL_idx)
            elif len(from_layers) == 2:
                actv1 = activations[i + from_layers[0]]
                actv2 = activations[from_layers[1]]
                activation = torch.cat((actv1, actv2))
                update_activation(i, pruned_model, activation, CBL_idx)
            activations.append(activation)

        # 因为 YOLOv3中 upsample下一层一定是 rout层所以这里没有传递β过去
        elif model_def['type'] == 'upsample':
            # num_ = torch.zeros(int(model.module_defs[i - 1]['filters']))
            activations.append(torch.zeros(int(model.module_defs[i - 1]['filters'])).to(device))
            # activations.append(activations[i - 1])

        elif model_def['type'] == 'yolo':
            activations.append(None)

        elif model_def['type'] == 'maxpool':  # 区分spp和tiny
            # route层不参加剪枝，只占位
            if pruned_model.module_defs[i+1]['type'] == 'route':
                activations.append(None)
            else:
                activation = activations[i - 1]
                update_activation(i, pruned_model, activation, CBL_idx)
                activations.append(activation)
    return pruned_model


def beta2next_(model, prune_idx, CBL_idx, CBL2mask):
    # 先科普下bn: https://blog.csdn.net/hjimce/article/details/50866313
    """
    该方法主要是将待剪枝层bn的β值做一下处理,然后用它的下一层的conv层的bn中的mean或者conv层中的bias来吸收这个处理值
    同时把把被移植conv中的γ与β置为0
    例: 第n层 y = ReLu{BN1[CONV1(x)]}   第n+1层 z = ReLu{BN2[CONV2(y)]} 假设如果有BN和ReLu的话
    -> y = ReLU(γ1 * [(x - mean1) / std1] + β1)
    -> z = ReLU(γ2 * [(y - mean2) / std1] + β2)
    第n层剪枝后(将待剪枝通道的γ置为0)y就分为了两部分,保留下来的γ与β 与γ置为0的β 设该层的剪枝掩膜为mask 1保留 0剪掉
    -> y = ReLU(γ1*mask * [(x - mean1) / std1] + β1*mask) + ReLU(0*(1-mask) * [(x - mean1) / std1] + β1*(1-mask))
    -> y = ReLU(γ1*mask * [(x - mean1) / std1] + β1*mask) + ReLU(β1*(1-mask))
    ->   = y' + ReLU(β1*(1-mask))

    所以第n+1层 z = ReLU(γ2 * [(CONV2(y') + CONV2(ReLU(β1*(1-mask))) - mean2) / std2] + β2)
    带入上面的式子我们可以发现,在保证第二层采用同样的计算方式和结果不变的情况下:令 mean2' = mean2 - CONV2(ReLU(β1*(1-mask)))
    -> z = ReLU(γ2 * [CONV2(y') - mean2'] / std2 + β2)

    同理,如果第n+1层是无bn的conv层的话,z = CONV2(y') + bias  令 bias' = bias+CONV2(ReLU(β1*(1-mask)))
    -> z = CONV2(y') + bias'
    :param model:       原始稀疏化训练后的模型
    :param prune_idx:   待剪枝的conv层索引,根据剪枝方式不同,可剪枝的层也不同
    :param CBL_idx:     有bn层的conv索引,YOLO层前一层除外
    :param CBLidx2mask: CBL_idx中conv层对应的剪枝掩膜 1保留 0剪掉
    :return: 处理后的剪枝模型
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pruned_model = deepcopy(model)
    activations = []
    for i, model_def in enumerate(model.module_defs):

        if model_def['type'] == 'convolutional':
            activation = torch.zeros(int(model_def['filters'])).to(device)
            if i in prune_idx:
                mask = torch.from_numpy(CBL2mask[i]).to(device)
                bn_module = pruned_model.module_list[i][1]
                bn_module.weight.data.mul_(mask)
                if model_def['activation'] == 'leaky':
                    activation = F.leaky_relu((1 - mask) * bn_module.bias.data, 0.1)
                elif model_def['activation'] == 'mish':
                    activation = (1 - mask) * bn_module.bias.data.mul(F.softplus(bn_module.bias.data).tanh())
                update_activation(i, pruned_model, activation, CBL_idx)
                bn_module.bias.data.mul_(mask)
            activations.append(activation)

        elif model_def['type'] == 'shortcut':
            actv1 = activations[i - 1]
            from_layer = int(model_def['from'])
            actv2 = activations[i + from_layer]
            activation = actv1 + actv2
            update_activation(i, pruned_model, activation, CBL_idx)
            activations.append(activation)

        elif model_def['type'] == 'route':
            #  spp不参与剪枝，其中的route不用更新，仅占位
            from_layers = [int(s) for s in model_def['layers'].split(',')]
            activation = None
            if len(from_layers) == 1:
                activation = activations[i + from_layers[0] if from_layers[0] < 0 else from_layers[0]]
                update_activation(i, pruned_model, activation, CBL_idx)
            elif len(from_layers) == 2:
                actv1 = activations[i + from_layers[0]]
                actv2 = activations[i + from_layers[1] if from_layers[1] < 0 else from_layers[1]]
                activation = torch.cat((actv1, actv2))
                update_activation(i, pruned_model, activation, CBL_idx)
            activations.append(activation)

        elif model_def['type'] == 'upsample':
            # activation = torch.zeros(int(model.module_defs[i - 1]['filters'])).cuda()
            activations.append(activations[i - 1])

        elif model_def['type'] == 'yolo':
            activations.append(None)

        elif model_def['type'] == 'maxpool':  # 区分spp和tiny
            if model.module_defs[i + 1]['type'] == 'route':
                activations.append(None)
            else:
                activation = activations[i - 1]
                update_activation(i, pruned_model, activation, CBL_idx)
                activations.append(activation)
    return pruned_model


def get_input_mask1(loose_model, idx, CBL2mask):
    model_defs = loose_model.module_defs
    # 获取每层conv的输入mask, 也就是上一层的输出mask
    # 第一层是固定的，是图像的输入mask
    if idx == 0:
        return np.ones(3)
    if model_defs[idx - 1]['type'] == 'convolutional':
        return CBL2mask[idx - 1]
    elif model_defs[idx - 1]['type'] == 'shortcut':
        return CBL2mask[idx - 2]
    elif model_defs[idx - 1]['type'] == 'route':
        # yolov3中有4个route层, route=-4是从后往前第4层，就是5个CBL那一层,这个我觉得就是只将索引向前退，不负责接
        # route = -1, 36 这个就是正常跳跃连接
        route_in_index = []
        # 区分spp  spp中 layers=-1,-3,-5,-6
        # 这种情况,layer_i会依次输出-1, -3, -5, -6这几个值
        for layer_i in model_defs[idx - 1]['layers'].split(','):
            if int(layer_i) < 0:
                route_in_index.append(idx - 1 + int(layer_i))   # 在现在索引的基础上往前退
            else:
                route_in_index.append(int(layer_i))  #
        if len(route_in_index) == 1:
            return CBL2mask[route_in_index[0]]
        elif len(route_in_index) == 2:
            # 这里不是平白无故剪去1的, 是因为rout层中layers有两个值实际所代表的层都是shortcut或upsmale层, 故需要进行-1
            return np.concatenate([CBL2mask[in_idx - 1] for in_idx in route_in_index])
        else:
            print('route is broken, layers = 0 or layers > n')


def get_input_mask(model_defs, idx, CBL2mask):
    # 获取每层conv的输入mask, 也就是上一层的输出mask
    # 第一层是固定的，是图像的输入mask
    if idx == 0:
        return np.ones(3)
    if model_defs[idx - 1]['type'] == 'convolutional':
        return CBL2mask[idx - 1]
    elif model_defs[idx - 1]['type'] == 'shortcut':
        return CBL2mask[idx - 2]
    elif model_defs[idx - 1]['type'] == 'route':
        # yolov3中有4个route层, route=-4是从后往前第4层，就是5个CBL那一层,这个我觉得就是只将索引向前退，不负责接
        # route = -1, 36 这个就是正常跳跃连接
        route_in_index = []
        # 区分spp  spp中 layers=-1,-3,-5,-6
        # 这种情况,layer_i会依次输出-1, -3, -5, -6这几个值
        for layer_i in model_defs[idx - 1]['layers'].split(','):
            if int(layer_i) < 0:
                route_in_index.append(idx - 1 + int(layer_i))   # 在现在索引的基础上往前退
            else:
                route_in_index.append(int(layer_i))  #
        if len(route_in_index) == 1:
            return CBL2mask[route_in_index[0]]
        elif len(route_in_index) == 2:
            # 这里不是平白无故剪去1的, 是因为rout层中layers有两个值实际所代表的层都是shortcut或upsmale层, 故需要进行-1
            return np.concatenate([CBL2mask[in_idx - 1] for in_idx in route_in_index])
        else:
            print('route is broken, layers = 0 or layers > n')


def init_weights_from_loose_model1(compact_model, loose_model, CBL_idx, Conv_idx, CBL2mask):
    # 将稀疏化的模型中保留下来的conv层，bn层的权重移植到compact_model中
    # compact_model   剪枝后的模型，已经初始化
    # loose_mode      移植了beta的稀疏化之后的模型
    # CBL_idx         有bn层的conv索引
    # Conv_idx        无bn层的conv索引
    for idx in CBL_idx:
        compact_CBL = compact_model.module_list[idx]
        loose_CBL = loose_model.module_list[idx]
        # BN中保留的通道索引
        # out_channel_index = np.argwhere(CBL2mask[idx]).tolist()  # out_channel_index --> list
        # out_channel_index = np.argwhere(CBL2mask[idx])[0, :].tolist()  # out_channel_index --> list  raw

        if np.argwhere(CBL2mask[idx]).shape[1] == 1:
            out_channel_index = np.argwhere(CBL2mask[idx])[:, 0].tolist()  # out_channel_index --> list  raw
        elif np.argwhere(CBL2mask[idx]).shape[0] == 1:
            out_channel_index = np.argwhere(CBL2mask[idx])[0, :].tolist()
        else:
            out_channel_index = np.argwhere(CBL2mask[idx]).tolist()

        # out_channel_index = np.argwhere(CBL2mask[idx])[0, :].tolist()

        compact_bn, loose_bn = compact_CBL[1], loose_CBL[1]
        # get compact_bn weight, bias, mean, var
        compact_bn.weight.data = loose_bn.weight.data[out_channel_index].clone()
        compact_bn.bias.data = loose_bn.bias.data[out_channel_index].clone()
        compact_bn.running_mean.data = loose_bn.running_mean.data[out_channel_index].clone()
        compact_bn.running_var.data = loose_bn.running_var.data[out_channel_index].clone()

        # get loose_model 中每一层的conv的输入mask
        input_mask = get_input_mask(loose_model, idx, CBL2mask)

        # 计算出输入通道的mask中非0的索引,即保留的索引,然后得出剪枝过的输入(上一层输出即这一层输入),输入通道的更改完成
        # 将其存为中间值后,再通过本层输出通道的mask得出最终剪切后的conv层的权重,输出通道的更改完成。整个conv层完成剪枝
        # 卷积的通道顺序 [out_channel, in_]
        in_channel_idx = np.argwhere(input_mask)[:, 0].tolist()   # 剪枝过后保留的通道
        # np.argwhere(input_mask)[:, 0] --> shape: (3, 1)
        # conv.weight.data里存储数据的方式是: torch.Size([64, 32, 3, 3])  -->
        # [out_channel, input_channel, kernel_size, kenel_size]

        compact_conv, loose_conv = compact_CBL[0], loose_CBL[0]
        tmp = loose_conv.weight.data[:, in_channel_idx, :, :].clone()
        compact_conv.weight.data = tmp[out_channel_index, :, :, :].clone()

    for idx in Conv_idx:
        # Conv_idx: 81, 93, 105 YOLO层前一层的conv只需要更改输入通道即可，输出通道是股东的3*(classes + 5)
        compact_CBL = compact_model.module_list[idx]
        loose_CBL = loose_model.module_list[idx]
        compact_conv, loose_conv = compact_CBL[0], loose_CBL[0]

        input_mask = get_input_mask(loose_model, idx, CBL2mask)
        in_channel_idx = np.argwhere(input_mask)[:, 0].tolist()   # 剪枝过后保留的通道
        compact_conv.weight.data = loose_conv.weight.data[:, in_channel_idx, :, :].clone()
        compact_conv.bias.data = loose_conv.bias.data.clone()


def init_weights_from_loose_model(compact_model, loose_model, CBL_idx, Conv_idx, CBLidx2mask):
    for idx in CBL_idx:
        compact_CBL = compact_model.module_list[idx]
        loose_CBL = loose_model.module_list[idx]
        out_channel_idx = np.argwhere(CBLidx2mask[idx])[:, 0].tolist()

        compact_bn, loose_bn         = compact_CBL[1], loose_CBL[1]
        compact_bn.weight.data       = loose_bn.weight.data[out_channel_idx].clone()
        compact_bn.bias.data         = loose_bn.bias.data[out_channel_idx].clone()
        compact_bn.running_mean.data = loose_bn.running_mean.data[out_channel_idx].clone()
        compact_bn.running_var.data  = loose_bn.running_var.data[out_channel_idx].clone()

        input_mask = get_input_mask(loose_model.module_defs, idx, CBLidx2mask)
        in_channel_idx = np.argwhere(input_mask)[:, 0].tolist()
        compact_conv, loose_conv = compact_CBL[0], loose_CBL[0]
        tmp = loose_conv.weight.data[:, in_channel_idx, :, :].clone()
        compact_conv.weight.data = tmp[out_channel_idx, :, :, :].clone()

    for idx in Conv_idx:
        compact_conv = compact_model.module_list[idx][0]
        loose_conv = loose_model.module_list[idx][0]

        input_mask = get_input_mask(loose_model.module_defs, idx, CBLidx2mask)
        in_channel_idx = np.argwhere(input_mask)[:, 0].tolist()
        compact_conv.weight.data = loose_conv.weight.data[:, in_channel_idx, :, :].clone()
        compact_conv.bias.data   = loose_conv.bias.data.clone()


def obtain_avg_forward_time(input, model, repeat=2):
    # 重复200次算均值
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    start_time = time.time()
    with torch.no_grad():
        for i in range(repeat):
            output = model(input).to(device)
    cost_time = (time.time() - start_time) / repeat
    return cost_time, output


def merge_mask_(model, CBL2mask, CBL2filters):
    # 对于每组shortcut, 它将相连的各卷积层的剪枝mask取并集

    # 取shortcut的首尾，也就是直连的CBL
    # range(start, stop, step)
    # 0: len() - 1
    for idx in range(len(model.module_defs)-1, -1, -1):
        module_type = model.module_defs[idx]['type']
        if module_type == 'shortcut':
            if model.module_defs[idx]['is_access']:   # model.module_defs[layer_i]['is_access'] = False  这个值是False
                continue
            # --------------- mask -------------
            merge_masks = []
            layer_i = idx
            while module_type == 'shortcut':
                model.module_defs[layer_i]['is_access'] = True   # 将这个标志设为True
                if model.module_defs[layer_i - 1]['type'] == 'convolutional':
                    bn = model.module_defs[layer_i - 1]['batch_normalize']
                    if bn:
                        merge_masks.append(CBL2mask[layer_i - 1].unsqueeze(0))
                layer_i = int(model.module_defs[layer_i]['from']) + layer_i    # 跳跃连接层，往前退4
                module_type = model.module_defs[layer_i]['type']
                if module_type == 'convolutional':    # 负责下采样的那个CBL
                    bn = int(model.module_defs[layer_i]['batch_normalize'])
                    if bn:
                        merge_masks.append(CBL2mask[layer_i].unsqueeze(0))
            if len(merge_masks) > 1:
                # 每组shortcut中的相连的各卷积层的mask取并集
                merge_masks = torch.cat(merge_masks, 0)
                merge_masks = (torch.sum(merge_masks, dim=0) > 0).float()    # 这里求和， 大于1的为1，小于1的为0
            else:
                merge_masks = merge_masks[0].float()

            # ----------------filters-----------
            module_type = 'shortcut'
            layer_i = idx
            while module_type == 'shortcut':
                if model.module_defs[layer_i - 1]['type'] == 'convolutional':
                    bn = model.module_defs[layer_i - 1]['batch_normalize']
                    if bn:
                        CBL2mask[layer_i - 1] = merge_masks
                        CBL2filters[layer_i - 1] = int(torch.sum(merge_masks).item())
                layer_i = int(model.module_defs[layer_i]['from']) + layer_i    # 跳跃连接层，往前退4
                module_type = model.module_defs[layer_i]['type']
                if module_type == 'convolutional':
                    bn = int(model.module_defs[layer_i]['batch_normalize'])
                    if bn:
                        CBL2mask[layer_i] = merge_masks
                        CBL2filters[layer_i] = int(torch.sum(merge_masks).item())


def merge_mask(model, CBLidx2mask, CBLidx2filters):
    for i in range(len(model.module_defs) - 1, -1, -1):
        mtype = model.module_defs[i]['type']
        if mtype == 'shortcut':
            if model.module_defs[i]['is_access']:
                continue

            Merge_masks = []
            layer_i = i
            while mtype == 'shortcut':
                model.module_defs[layer_i]['is_access'] = True

                if model.module_defs[layer_i - 1]['type'] == 'convolutional':
                    bn = int(model.module_defs[layer_i - 1]['batch_normalize'])
                    if bn:
                        Merge_masks.append(CBLidx2mask[layer_i - 1].unsqueeze(0))

                layer_i = int(model.module_defs[layer_i]['from']) + layer_i
                mtype = model.module_defs[layer_i]['type']

                if mtype == 'convolutional':
                    bn = int(model.module_defs[layer_i]['batch_normalize'])
                    if bn:
                        Merge_masks.append(CBLidx2mask[layer_i].unsqueeze(0))

            if len(Merge_masks) > 1:
                Merge_masks = torch.cat(Merge_masks, 0)
                merge_mask = (torch.sum(Merge_masks, dim=0) > 0).float()
            else:
                merge_mask = Merge_masks[0].float()

            layer_i = i
            mtype = 'shortcut'
            while mtype == 'shortcut':

                if model.module_defs[layer_i - 1]['type'] == 'convolutional':
                    bn = int(model.module_defs[layer_i - 1]['batch_normalize'])
                    if bn:
                        CBLidx2mask[layer_i - 1] = merge_mask
                        CBLidx2filters[layer_i - 1] = int(torch.sum(merge_mask).item())

                layer_i = int(model.module_defs[layer_i]['from']) + layer_i
                mtype = model.module_defs[layer_i]['type']

                if mtype == 'convolutional':
                    bn = int(model.module_defs[layer_i]['batch_normalize'])
                    if bn:
                        CBLidx2mask[layer_i] = merge_mask
                        CBLidx2filters[layer_i] = int(torch.sum(merge_mask).item())


def slim_prune_demo():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Darknet(hyp0['cfg_path']).to(device)  # 训练的时候才需要初始化，这里就需要使用网络里的数据，所以不必再初始化
    # 训练的时候才需要初始化，这里就需要使用网络里的数据，所以不必再初始化
    model.load_state_dict(torch.load(hyp0['weight_path'], map_location="cpu"))

    # precision, recall, AP, f1, ap_class = evaluate(
    #     model,
    #     path=hyp0['test_path'],
    #     iou_thres=hyp0['iou_thresh'],
    #     conf_thres=hyp0['conf_thresh'],
    #     nms_thres=hyp0['nms_thresh'],
    #     img_size=hyp0['img_size'],
    #     batch_size=hyp0['batch_size'],
    # )
    # 剪枝前的参数总量
    # before_parameters = sum([param.nelement() for param in model.parameters()])   # 62002578
    # print(f'稀疏化训练后模型mAP:{AP.mean():.4f}')

    CBL_idx, Conv_idx, prune_idx, shortcut_idx, shortcut_all = parse_moudle_defs1(model.module_defs)
    bn_weights = gather_bn_weights(model.module_list, prune_idx)

    # gamma_list = []
    # for idx in prune_idx:
    #     bn_module = model.module_list[idx][1].weight.data.abs().max().item()
    #     gamma_list.append(bn_module)
    # high_thresh = min(gamma_list)    # TODO 所有bn层中最大的gamma中最小的那个数,取这个值可以保证不会整层剪掉
    # print(f'剪枝γ最小的阈值1{high_thresh:.4f}')

    sorted_bn = torch.sort(bn_weights)[0]   # 0是tensor 1是index
    # print(sorted_bn)

    # 固定阈值得到的 threshold
    threshold_index = int(len(bn_weights) * hyp0['global_percent'])   # 理论的剪枝率
    threshold = sorted_bn[threshold_index].to(device)
    print(f'剪枝γ最小的阈值2{threshold:.4f}')

    # 虽然上面已经能看到剪枝后的效果，但是没有生成剪枝后的模型结构，因此下面的代码是为了生成新的模型结构并拷贝旧模型参数到新模型
    num_filters, filters_mask = get_filters_mask_channel(model, threshold, CBL_idx, prune_idx)

    # CBL2mask --> upsample前一个CBL, yolo前一个conv, shortcut层中的起始末尾层
    # 1保留   0剪掉
    CBL2mask = {idx: mask for idx, mask in zip(CBL_idx, filters_mask)}
    CBL2filters = {idx: filters for idx, filters in zip(CBL_idx, num_filters)}

    # 对于每组shortcut, 它将相连的各卷积层的剪枝mask取并集
    for i in model.module_defs:
        if i['type'] == 'shortcut':
            i['is_access'] = False
    print('merge the mask of layers connected to shortcut!')

    merge_mask(model, CBL2mask, CBL2filters)
    # 将torch转成numpy格式
    for i in CBL2mask:
        CBL2mask[i] = CBL2mask[i].clone().cpu().numpy()
    # 到这里已经gamma设置为0了，接下来处理beta

    # 处理bn层的β
    # 将稀疏化训练后的模型中待剪枝层的bn中的β参数移植到下后面的层,并返回移植后的模型
    pruned_model = beta2next_(model, prune_idx, CBL_idx, CBL2mask)
    # pruned_model = beta2next(model, prune_idx, CBL_idx, CBL2mask)

    # precision, recall, AP2, f1, ap_class = evaluate(
    #     pruned_model,
    #     path=hyp0['test_path'],
    #     iou_thres=hyp0['iou_thresh'],
    #     conf_thres=hyp0['conf_thresh'],
    #     nms_thres=hyp0['nms_thresh'],
    #     img_size=hyp0['img_size'],
    #     batch_size=hyp0['batch_size'],
    # )
    # print(f'剪枝层移植β之后的mAP{AP2.mean():.4f}')

    for i in model.module_defs:
        if i['type'] == 'shortcut':
            i.pop('is_access')

    # 重新建一份和原始cfg一模一样的网络配置文件，并更改剪枝层的卷积核个数
    compact_model_defs = deepcopy(model.module_defs)   # 复制一份cfg文件，根据旧的复制
    for idx, nums in zip(CBL_idx, num_filters):   # num_filters 这个表示这一层中保留的通道个数
        assert compact_model_defs[idx]['type'] == 'convolutional'
        compact_model_defs[idx]['filters'] = str(CBL2filters[idx])
    # 通过剪枝后的cfg文件初始化一个新模型，并计算模型的参数量
    compact_model = Darknet([model.hyperparams.copy()] + compact_model_defs).to(device)
    after_parameters = sum([param.nelement() for param in compact_model.parameters()])
    # 将pruned_model中的部分权重移植到刚刚初始化后的compact_model，简单来说就是将compact_model中conv层多余的通道剪掉

    try:
        init_weights_from_loose_model(compact_model, pruned_model, CBL_idx, Conv_idx, CBL2mask)
    except Exception as e:
        traceback.print_exc()
    random_input = torch.rand((1, 3, hyp0['img_size'], hyp0['img_size'])).to(device)
    # 测试两个模型前向传播的时间，理论上loose_model和pruned_model应该完全一致
    before_time, pruned_output = obtain_avg_forward_time(random_input, pruned_model)
    after_time, compact_output = obtain_avg_forward_time(random_input, compact_model)
    differ = (pruned_output - compact_output).abs().gt(0.01).sum().item()   # 差值大于0.01的和  理论上这个值为0
    if differ > 0:
        print(f'剪枝过程中出现了问题')
    else:
        print(f'剪枝没问题')

    # precision, recall, AP3, f1, ap_class = evaluate(
    #     pruned_model,
    #     path=hyp0['test_path'],
    #     iou_thres=hyp0['iou_thresh'],
    #     conf_thres=hyp0['conf_thresh'],
    #     nms_thres=hyp0['nms_thresh'],
    #     img_size=hyp0['img_size'],
    #     batch_size=hyp0['batch_size'],
    # )
    # print(f'剪枝层移植β之后的mAP{AP3.mean():.4f}')
    # 比较剪枝前后参数数量的变化，指标的变化
    # metric_table = [
    #     ["Metric", "Before", "After"],
    #     ["mAP", f'{AP.mean():.4f}', f'{AP3.mean():.4f}'],
    #     ["Parameters", f"{before_parameters}", f"{after_parameters}"],
    #     ["Inference", f'{before_time:.4f}', f'{after_time:.4f}']
    # ]
    # print(AsciiTable(metric_table).table)

    # 生成剪枝后的cfg文件并保存模型
    # cfg_path = f'config/yolov3_channel_strategy3_{hyp0["global_percent"]}_mAP_{AP3.mean():.4f}.cfg'
    # write_cfg(cfg_path, model.module_defs)
    # print(f'保持好剪枝后的cfg文件，地址: {cfg_path}')
    # weight_path = f'checkpoints/channel_strategy3_{hyp0["global_percent"]}_mAP_{AP3.mean():.4f}.pth'
    # torch.save(compact_model.state_dict(), weight_path)
    # print(f'保持好剪枝后的cfg文件，地址: {weight_path}')

    cfg_path = 'config/pruned.cfg'
    write_cfg(cfg_path, [model.hyperparams.copy()] + compact_model_defs)
    weight_path = 'checkpoints/pruned.pth'
    if weight_path.endswith('.pth'):
        weight_path = weight_path.replace('.pth', '.weights')
    save_darknet_weights(compact_model, path=weight_path)
    print(f'保持好剪枝后的cfg文件，地址: {weight_path}')


if __name__ == '__main__':
    slim_prune_demo()
