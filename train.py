from __future__ import division

from models import *
from utils.logger import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from test import evaluate
from terminaltables import AsciiTable
import os, sys, cv2
import time
import datetime
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
from prune.prune_tools import updateBN, parse_moudle_defs, gather_bn_weights, plot_images, parse_moudle_defs1, get_masks
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '2'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=250, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--data_config", type=str, default="data/math_blank/class.data", help="path to data config file")

    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--pretrained_weights", type=str, default='weights/yolov3.weights',
                        help="if specified starts from checkpoint model")

    # TODO finetune---stadge
    # parser.add_argument("--model_def", type=str, default="config/channel_layer_16_0.7_shortcut_0.0001.cfg",
    #                     help="path to model definition file")
    # parser.add_argument("--pretrained_weights", type=str, default='checkpoints/channel_layer_16_0.7_shortcut_0.0001.weights',
    #                     help="if specified starts from checkpoint model")

    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=1120, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=5, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=5, help="interval evaluations on validation set")
    parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    parser.add_argument("--multiscale_training", default=False, help="allow for multi-scale training")

    # TODO train version
    # parser.add_argument('--sparsity-regularization', '-sr', dest='sr', action='store_true',
    #                     help='train with channel sparsity regularization')
    # parser.add_argument('--s', type=float, default=0.001, help='scale sparse rate')
    # parser.add_argument('--prune', type=int, default=1, help='0:nomal prune 1:other prune ')

    # TODO debug version
    parser.add_argument('--sparsity-regularization', '-sr', dest='sr', default=True, action='store_true',
                        help='train with channel sparsity regularization')

    parser.add_argument('--s', type=float, default=0.0001, help='scale sparse rate')   # O.3
    parser.add_argument('--prune', type=int, default=1, help='0:nomal prune 1:other prune ')   # 1 --> strategy3
    # TODO end debug version

    opt = parser.parse_args()
    print(opt)
    logger = Logger("logs")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Get data configuration
    data_config = parse_data_config(opt.data_config)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    model.apply(weights_init_normal)

    # If specified we start from checkpoint
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(opt.pretrained_weights))
        else:
            model.load_darknet_weights(opt.pretrained_weights)

    # Get dataloader
    dataset = ListDataset(train_path, img_size=opt.img_size, augment=True, multiscale=opt.multiscale_training)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    optimizer = torch.optim.Adam(model.parameters())

    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]

    if opt.sr:
        if opt.prune == 0:
            # 通道剪枝策略1，shortcut层中的起始末尾层
            CBL_idx, Conv_idx, prune_idx = parse_moudle_defs(model.module_defs)  # TODO 剪枝策略1
            print('normal sparse training')
        elif opt.prune == 1:
            CBL_idx, _, prune_idx, shortcut_idx, _ = parse_moudle_defs1(model.module_defs)  # TODO 剪枝策略3
            print('shortcut sparse training')

    # tensorboard
    tb_writer = SummaryWriter()

    for epoch in range(opt.epochs):
        model.train()
        if opt.sr:
        # TODO bn可视化
            for idx in prune_idx:
                bn_weights = gather_bn_weights(model.module_list, [idx])
                tb_writer.add_histogram('bn_weight/hist', bn_weights.numpy(), epoch, bins='doane')

        start_time = time.time()
        for batch_i, (paths, imgs, targets) in enumerate(dataloader):
            batches_done = len(dataloader) * epoch + batch_i

            # TODO plot images  100次保存一次结果
            if batches_done == 0:
                fname = 'train_batch%g.jpg' % batch_i
                plot_images(imgs=imgs, targets=targets, paths=paths, fname=fname)
                tb_writer.add_image(fname, cv2.imread(fname)[:, :, ::-1], dataformats='HWC')

            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)

            # TODO   可视化的是哪一张图像
            # tb_writer.add_image(fname, cv2.imread(fname)[:, :, ::-1], dataformats='HWC')  #
            loss, outputs = model(imgs, targets)
            loss.backward()

            # TODO 稀疏化训练
            if opt.sr and opt.prune == 0 and epoch > opt.epochs * 0.5:
                # idx2masks 是一个字典
                idx2masks = get_masks(model, prune_idx, 0.85)
            elif opt.sr and opt.prune == 1 and epoch > opt.epochs * 0.5:
                idx2masks = get_masks(model, prune_idx, 0.85)
            else:
                idx2masks = None
            if opt.sr:
                updateBN(model.module_list, opt.s, prune_idx, idx2masks)

            if batches_done % opt.gradient_accumulations:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

            if (batch_i + 1) % 100 == 0:    # 隔100次show一次结果
                # ----------------
                #   Log progress
                # ----------------

                log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, opt.epochs, batch_i, len(dataloader))
                metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

                # Log metrics at each YOLO layer
                for i, metric in enumerate(metrics):
                    formats = {m: "%.6f" for m in metrics}
                    formats["grid_size"] = "%2d"
                    formats["cls_acc"] = "%.2f%%"
                    row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
                    metric_table += [[metric, *row_metrics]]

                    # Tensorboard logging
                    tensorboard_log = []
                    for j, yolo in enumerate(model.yolo_layers):
                        for name, metric in yolo.metrics.items():
                            if name != "grid_size":
                                tensorboard_log += [(f"{name}_{j+1}", metric)]
                    tensorboard_log += [("loss", loss.item())]
                    logger.list_of_scalars_summary(tensorboard_log, batches_done)

                log_str += AsciiTable(metric_table).table
                log_str += f"\nTotal loss {loss.item()}"

                # Determine approximate time left for epoch
                epoch_batches_left = len(dataloader) - (batch_i + 1)
                time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
                log_str += f"\n---- ETA {time_left}"
                print(log_str)

            model.seen += imgs.size(0)

        if epoch % opt.evaluation_interval == 0:
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            precision, recall, AP, f1, ap_class = evaluate(
                model,
                path=valid_path,
                iou_thres=0.5,
                conf_thres=0.5,
                nms_thres=0.5,
                img_size=opt.img_size,
                batch_size=opt.batch_size,
            )
            evaluation_metrics = [
                ("val_precision", precision.mean()),
                ("val_recall", recall.mean()),
                ("val_mAP", AP.mean()),
                ("val_f1", f1.mean()),
            ]
            logger.list_of_scalars_summary(evaluation_metrics, epoch)

            # Print class APs and mAP
            ap_table = [["Index", "Class name", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
            print(f"---- mAP {AP.mean()}")

        if epoch % opt.checkpoint_interval == 0:
            torch.save(model.state_dict(), f"checkpoints/yolov3_ckpt_%d.pth" % epoch)

    # end epoch
    for idx in prune_idx:
        bn_weights = gather_bn_weights(model.module_list, [idx])
        tb_writer.add_histogram('after_train_perlayer_bn_weights/hist', bn_weights.numpy(), idx, bins='doane')

    torch.cuda.empty_cache()
