# -*- coding:utf-8 -*-
__author__ = 'lynn'
__date__ = '2021/1/28 11:25'


import os, shutil


img_path = 'data/math_blank/VOC2007/JPEGImages'
xml_path = 'data/math_blank/VOC2007/Annotations'

save_path_img = 'data/math_blank/VOC2007/jpg'
save_path_xml = 'data/math_blank/VOC2007/xml'

img_list = os.listdir(img_path)
xml_list = os.listdir(xml_path)
for ele in img_list:
    name = ele.replace('.jpg', '.xml')
    try:
        if name in xml_list:
            shutil.move(os.path.join(img_path, ele), os.path.join(save_path_img, ele))
            shutil.move(os.path.join(xml_path, name), os.path.join(save_path_xml, name))
        else:
            print('imgname:', ele)
            continue
    except Exception as e:
        print('imgname:', name)
        continue