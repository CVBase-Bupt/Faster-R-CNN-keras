#encoding:utf-8
import cv2
import numpy as np
class Process:
    def __init__(self,img,C):
        self.img = img
        self.C = C
        
    # 短边缩放到img_min_side=600，长边等比例缩放
    def format_img_size(self,):
        img_min_side = float(self.C.im_size)
        (height,width,_) = self.img.shape
            
        if width <= height:
            ratio = img_min_side/width
            new_height = int(ratio * height)
            new_width = int(img_min_side)
        else:
            ratio = img_min_side/height
            new_width = int(ratio * width)
            new_height = int(img_min_side)
        img = cv2.resize(self.img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        return img, ratio

    # 图像预处理
    def format_img_channels(self,):
        # cv2读入的形式为BGR,需转成RGB
        img = self.img[:, :, (2, 1, 0)] 
        img = img.astype(np.float32)
        # RGB每通道减均值
        img[:, :, 0] -= self.C.img_channel_mean[0]
        img[:, :, 1] -= self.C.img_channel_mean[1]
        img[:, :, 2] -= self.C.img_channel_mean[2]
        img /= self.C.img_scaling_factor
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img

    def format_img(self,):
        """ formats an image for model prediction based on config """
        self.img, ratio = self.format_img_size()
        img = self.format_img_channels()
        img = np.transpose(img, (0, 2, 3, 1))
        return img, ratio

