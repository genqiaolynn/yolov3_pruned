from __future__ import division
# -*- coding:utf-8 -*-
__author__ = 'lynn'
__date__ = '2021/1/18 18:09'

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse

from PIL import Image

import torch, cv2
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
import xml.etree.ElementTree as ET


def create_xml(obj_name, tree, xmin, ymin, xmax, ymax):
    root = tree.getroot()

    pobject = ET.SubElement(root, 'object', {})
    pname = ET.SubElement(pobject, 'name')
    pname.text = obj_name
    ppose = ET.SubElement(pobject, 'pose')
    ppose.text = 'Unspecified'
    ptruncated = ET.SubElement(pobject, 'truncated')
    ptruncated.text = '0'
    pdifficult = ET.SubElement(pobject, 'difficult')
    pdifficult.text = '0'
    # add bndbox
    pbndbox = ET.SubElement(pobject, 'bndbox')
    pxmin = ET.SubElement(pbndbox, 'xmin')
    pxmin.text = str(xmin)

    pymin = ET.SubElement(pbndbox, 'ymin')
    pymin.text = str(ymin)

    pxmax = ET.SubElement(pbndbox, 'xmax')
    pxmax.text = str(xmax)

    pymax = ET.SubElement(pbndbox, 'ymax')
    pymax.text = str(ymax)

    return tree


def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/math_blank/qqq", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="checkpoints/yolov3_ckpt_10.pth", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/math_blank/class.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=1120, help="size of each image dimension")
    # parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)

    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        # model.load_state_dict(torch.load(opt.weights_path), map_location=lambda storage, loc: storage)
        model.load_state_dict(torch.load(opt.weights_path, map_location="cpu"))

    model.eval()  # Set in evaluation mode

    dataloader = DataLoader(
        ImageFolder(opt.image_folder, img_size=opt.img_size),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
    )

    classes = load_classes(opt.class_path)  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index

    print("\nPerforming object detection:")
    prev_time = time.time()
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))

        # Get detections
        with torch.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)
            # detections = py_cpu_nms(detections, opt.nms_thres)

        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))

        # Save image and detections
        imgs.extend(img_paths)
        img_detections.extend(detections)

    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 60)]

    print("\nSaving images:")
    # Iterate through images and save plot of detections
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

        print("(%d) Image: '%s'" % (img_i, path))

        # Create plot
        img = np.array(Image.open(path))
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        # Draw bounding boxes and labels of detections
        if detections is not None:
            # Rescale boxes to original image
            detections = rescale_boxes(detections, opt.img_size, img.shape[:2])
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_preds)
            xml_template = './exam_info/000000-template.xml'
            tree = ET.parse(xml_template)
            root = tree.getroot()
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                cls = classes[int(cls_pred)]
                create_xml(cls + '_' + str(float(cls_conf.numpy())), tree, int(x1.numpy()),
                           int(y1.numpy()), int(x2.numpy()), int(y2.numpy()))
            filename = path.split('/')[-1]
            tree.write('output/{}'.format(filename).replace('.jpg', '.xml'))
            cv2.imwrite('output/{}'.format(filename), img)

        #         print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))
        #
        #         box_w = x2 - x1
        #         box_h = y2 - y1
        #
        #         color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
        #         # Create a Rectangle patch
        #         bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
        #         # Add the bbox to the plot
        #         ax.add_patch(bbox)
        #         # Add label
        #         plt.text(
        #             x1,
        #             y1,
        #             s=classes[int(cls_pred)],
        #             color="white",
        #             verticalalignment="top",
        #             bbox={"color": color, "pad": 0},
        #         )
        #
        # # Save generated image with detections
        # plt.axis("off")
        # plt.gca().xaxis.set_major_locator(NullLocator())
        # plt.gca().yaxis.set_major_locator(NullLocator())
        # filename = path.split('/')[-1].split(".")[0]
        # plt.savefig(f"output/{filename}.png", bbox_inches="tight", pad_inches=0.0)
        # plt.close()