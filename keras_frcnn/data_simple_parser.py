#encoding:utf-8
import cv2
import numpy as np

def get_data(input_path):
    '''
    输入：训练集标签文本路径，文本格式：(filename,x1,y1,x2,y2,class_name)
    返回：all_data，字典一级key：filename  二级key：filepath,width,height,trainval/test，bboxes
        classes_count，字典key：classname，value：每类图像数量
        class_mapping，字典key：class_name，value：每类对应一个id，其中背景再最后一项
    '''
    found_bg = False
    all_imgs = {}
    classes_count = {}
    class_mapping = {}

    #1.打开文件
    with open(input_path,'r') as f:

        print('Parsing annotation files')
        #2.按行读取信息：图片名称，坐标，类名
        for line in f:
            line_split = line.strip().split(',')
            (filename,x1,y1,x2,y2,class_name) = line_split
            #计算每一类的数量
            if class_name not in classes_count:
                classes_count[class_name] = 1
            else:
                classes_count[class_name] += 1

            #背景类，常用于难分负样本挖掘
            if class_name not in class_mapping:
                if class_name == 'bg' and found_bg == False:
                    print('Found class name with special name bg. Will be treated as a background region (this is usually for hard negative mining).')
                    found_bg = True
                
                class_mapping[class_name] = len(class_mapping)
            #3.建立每张图像的基本信息
            if filename not in all_imgs:
                all_imgs[filename] = {}				
                img = cv2.imread(filename)
                (rows,cols) = img.shape[:2]
                all_imgs[filename]['filepath'] = filename
                all_imgs[filename]['width'] = cols
                all_imgs[filename]['height'] = rows
                all_imgs[filename]['bboxes'] = []
                if np.random.randint(0,6) > 0:
                    all_imgs[filename]['imageset'] = 'trainval'
                else:
                    all_imgs[filename]['imageset'] = 'test'
            # 4.建立bbox的信息
            all_imgs[filename]['bboxes'].append({'class': class_name, 'x1': int(x1), 'x2': int(x2), 'y1': int(y1), 'y2': int(y2)})


        all_data = []
        #key： filename：filepath,width,height,trainval/test，bboxes
        for key in all_imgs:
            all_data.append(all_imgs[key])

        # 确认背景类在list中最后一项
        if found_bg:
            if class_mapping['bg'] != len(class_mapping) - 1:
                key_to_switch = [key for key in class_mapping.keys() if class_mapping[key] == len(class_mapping)-1][0]
                val_to_switch = class_mapping['bg']
                class_mapping['bg'] = len(class_mapping) - 1
                class_mapping[key_to_switch] = val_to_switch
        else:
            classes_count['bg'] = 0
            class_mapping['bg'] = len(class_mapping)

        return all_data, classes_count, class_mapping


