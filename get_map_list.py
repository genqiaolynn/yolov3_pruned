# -*- coding:utf-8 -*-
__author__ = 'lynn'
__date__ = '2021/4/4 15:04'


import os, re


# 0.7580283567340425

# path1 = r'F:\yolov3_\PyTorch_YOLOv3_raw\nohup.out'
path1 = r'F:\yolov3_\11_剪枝存的文件\baseline\nohup_raw_repo.out'
path11 = open(path1, 'r', encoding='utf-8').readlines()
map_list = []
for line in path11[1:]:
    line = line.strip(' ').strip('\n')
    content = re.findall('mAP', line)
    if len(content) != 0:
        map_list.append(float(line.split(' ')[-1]))
print(map_list)
max_element = max(map_list)
print(max_element)
