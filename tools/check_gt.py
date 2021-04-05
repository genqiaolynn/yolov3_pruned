# -*- coding:utf-8 -*-
__author__ = 'lynn'
__date__ = '2021/1/28 12:14'


import os
import xml.etree.ElementTree as ET
import numpy as np


img_path = r'F:\yolov3_\PyTorch_YOLOv3_raw\data\math_blank\VOC2007\JPEGImages\3.jpg'
xml_path = r'F:\yolov3_\PyTorch_YOLOv3_raw\data\math_blank\VOC2007\Annotations\3.xml'


# xml_list = os.listdir(xml_path)
sheet_list = []
# for ele in xml_list:
# xml_path0 = os.path.join(xml_path, ele)
tree = ET.parse(xml_path)
root = tree.getroot()
obj1 = root.findall('object')
for obj in obj1:
    sheet_dict = {}
    bounding_box_dict = {}
    name = obj.find('name').text
    bndbox = obj.find('bndbox')
    xmin = bndbox.find('xmin').text
    ymin = bndbox.find('ymin').text
    xmax = bndbox.find('xmax').text
    ymax = bndbox.find('ymax').text
    bounding_box_dict['xmin'] = xmin
    bounding_box_dict['ymin'] = ymin
    bounding_box_dict['xmax'] = xmax
    bounding_box_dict['ymax'] = ymax

    sheet_dict['class_name'] = name
    sheet_dict['bounding_box'] = bounding_box_dict
    sheet_list.append(sheet_dict)

print(sheet_list)
choice_s_list = [ele for ele in sheet_list if ele['class_name'] == 'choice_s']
choice_m_list = [ele for ele in sheet_list if ele['class_name'] == 'choice_m']
iou_list = []
for ele in choice_m_list:
    choice_m_xmin = ele['bounding_box']['xmin']
    choice_m_xmax = ele['bounding_box']['xmax']
    choice_m_ymin = ele['bounding_box']['ymin']
    choice_m_ymax = ele['bounding_box']['ymax']
    for choice_s in choice_s_list:
        choice_s_bbox = choice_s['bounding_box']
        xmin_d = int(int(choice_s_bbox['xmin']) + int(choice_s_bbox['xmax'])) / 2
        ymin_d = int(int(choice_s_bbox['ymin']) + int(choice_s_bbox['ymax'])) / 2
        if int(choice_m_xmin) < int(xmin_d) < int(choice_m_xmax) and int(choice_m_ymin) < int(ymin_d) < int(choice_m_ymax):
            choice_m_np = np.array(list(map(int, [choice_m_xmin, choice_m_ymin, choice_m_xmax, choice_m_ymax])))
            choice_s_np = np.array(list(map(int, [choice_s_bbox['xmin'], choice_s_bbox['ymin'], choice_s_bbox['xmax'], choice_s_bbox['ymax']])))

            new_xmin = np.maximum(choice_m_np[0], choice_s_np[0])
            new_ymin = np.maximum(choice_m_np[1], choice_s_np[1])
            new_xmax = np.minimum(choice_m_np[2], choice_s_np[2])
            new_ymax = np.minimum(choice_m_np[3], choice_s_np[3])

            inter_area = (new_xmax - new_xmin) * (new_ymax - new_ymin)
            union_area = (choice_m_np[2] - choice_m_np[0]) * (choice_m_np[3] - choice_m_np[1]) + \
                         (choice_s_np[2] - choice_s_np[0]) * (choice_s_np[3] - choice_s_np[1]) - inter_area
            iou = inter_area / union_area
            iou_list.append(iou)
print(iou_list)




# 看同一个choice_m里面的choice_s和choice_n占比例多少，计算IOU值