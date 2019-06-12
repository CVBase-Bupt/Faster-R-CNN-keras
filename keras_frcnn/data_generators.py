#--coding:utf-8--
from __future__ import absolute_import
import numpy as np
import cv2
import random
import copy
from . import data_augment
import threading
import itertools


# 默认坐标顺序 (x1,y1,x2,y2)
def union(au, bu, area_intersection):
	#并集
	area_a = (au[2] - au[0]) * (au[3] - au[1])
	area_b = (bu[2] - bu[0]) * (bu[3] - bu[1])
	area_union = area_a + area_b - area_intersection
	return area_union


def intersection(ai, bi):
	#交集
	x = max(ai[0], bi[0])
	y = max(ai[1], bi[1])
	w = min(ai[2], bi[2]) - x
	h = min(ai[3], bi[3]) - y
	if w < 0 or h < 0:
		return 0
	return w*h


def iou(a, b):
	
	if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
		return 0.0

	area_i = intersection(a, b)
	area_u = union(a, b, area_i)

	return float(area_i) / float(area_u + 1e-6)


def get_new_img_size(width, height, img_min_side=600):
	#将短边缩放到600，长边等比例缩放
	if width <= height:
		f = float(img_min_side) / width
		resized_height = int(f * height)
		resized_width = img_min_side
	else:
		f = float(img_min_side) / height
		resized_width = int(f * width)
		resized_height = img_min_side

	return resized_width, resized_height


class SampleSelector:
	# 忽略样本数量为零的类别
	def __init__(self, class_count):
		self.classes = [b for b in class_count.keys() if class_count[b] > 0]
		self.class_cycle = itertools.cycle(self.classes) # cycle()会把传入的一个序列无限重复下去
		self.curr_class = next(self.class_cycle)

	def skip_sample_for_balanced_class(self, img_data):

		class_in_img = False

		for bbox in img_data['bboxes']:
			cls_name = bbox['class']
			if cls_name == self.curr_class:
				class_in_img = True
				self.curr_class = next(self.class_cycle)
				break

		if class_in_img:
			return False
		else:
			return True


def calc_rpn(C, img_data, width, height, resized_width, resized_height, img_length_calc_function):
    '''
    标记anchor正负样本，以及正样本对应的偏移值
    输入：
        C：配置文件 
        img_data 图像字典key：filepath,width,height,trainval/test，bboxes
        图像resize之前的长宽：width, height,
        图像resize之后的长宽：resized_width, resized_height,
        img_length_calc_function：计算特征图的长宽，输入resized_width, resized_height，输出featuremap_width, featuremap_height
    '''

    downscale = float(C.rpn_stride)
    anchor_sizes = C.anchor_box_scales
    anchor_ratios = C.anchor_box_ratios
    n_anchratios=len(anchor_ratios)
    # self.anchor_box_scales = [128, 256, 512]
    # self.anchor_box_ratios = [[1, 1], [1./math.sqrt(2), 2./math.sqrt(2)], [2./math.sqrt(2), 1./math.sqrt(2)]]

    num_anchors = len(anchor_sizes) * len(anchor_ratios)	

    # 网络输出特征图的尺寸
    (output_width, output_height) = img_length_calc_function(resized_width, resized_height)

    # 初始化数组
    y_rpn_overlap = np.zeros((output_height, output_width, num_anchors))
    y_is_box_valid = np.zeros((output_height, output_width, num_anchors))
    y_rpn_regr = np.zeros((output_height, output_width, num_anchors * 4))

    num_bboxes = len(img_data['bboxes'])
    num_anchors_for_bbox = np.zeros(num_bboxes).astype(int)
    best_anchor_for_bbox = -1*np.ones((num_bboxes, 4)).astype(int)
    best_iou_for_bbox = np.zeros(num_bboxes).astype(np.float32)
    best_x_for_bbox = np.zeros((num_bboxes, 4)).astype(int)
    best_dx_for_bbox = np.zeros((num_bboxes, 4)).astype(np.float32)

    gta = np.zeros((num_bboxes, 4))

    #真实标签，根据图像的缩放比例同步更新坐标，得到resize之后对应的坐标
    for bbox_num, bbox in enumerate(img_data['bboxes']):
        gta[bbox_num, 0] = bbox['x1'] * (resized_width / float(width))
        gta[bbox_num, 1] = bbox['x2'] * (resized_width / float(width))
        gta[bbox_num, 2] = bbox['y1'] * (resized_height / float(height))
        gta[bbox_num, 3] = bbox['y2'] * (resized_height / float(height))

    # 标记每一个anchor
    for anchor_size_idx in range(len(anchor_sizes)):
        for anchor_ratio_idx in range(len(anchor_ratios)):
            #尺寸×比例，得到边长
            anchor_x = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][0]
            anchor_y = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][1]	
            

            # 论文中提到，为每一个anchor标记为是否包含物体的标签，可以标为正样本的有
            # (i) 和真实box的iou最高的anchor
            # (ii) 和真实box的iou超过0.7
            # 最后可能一个真实box对应多个不同的anchor
            # 负样本<0.3,其余忽略

            #这里的操作略微有些繁琐，可以向量化一下
            for ix in range(output_width):
                # xmin,xmax	
                x1_anc = downscale * (ix + 0.5) - anchor_x / 2
                x2_anc = downscale * (ix + 0.5) + anchor_x / 2	

                # 忽略超出边界的框			
                if x1_anc < 0 or x2_anc > resized_width:
                	continue
                
                for jy in range(output_height):

                    # ymin,ymax
                    y1_anc = downscale * (jy + 0.5) - anchor_y / 2
                    y2_anc = downscale * (jy + 0.5) + anchor_y / 2

                    # 忽略超出边界的框	
                    if y1_anc < 0 or y2_anc > resized_height:
                        continue

                    # bbox_type表示当前框是否包含目标
                    bbox_type = 'neg'

                    # anchor best IOU  not GT bbox best IOU
                    best_iou_for_loc = 0.0

                    for bbox_num in range(num_bboxes):
                    
                        # 计算当前anchor和所有真实框之间的IOU
                        curr_iou = iou([gta[bbox_num, 0], gta[bbox_num, 2], gta[bbox_num, 1], gta[bbox_num, 3]], [x1_anc, y1_anc, x2_anc, y2_anc])
                
                            # 如果此时IOU是真实框目前最大的，或者IOU大于阈值，计算回归值
                        if curr_iou > best_iou_for_bbox[bbox_num] or curr_iou > C.rpn_max_overlap:
                            cx = (gta[bbox_num, 0] + gta[bbox_num, 1]) / 2.0
                            cy = (gta[bbox_num, 2] + gta[bbox_num, 3]) / 2.0
                            cxa = (x1_anc + x2_anc)/2.0
                            cya = (y1_anc + y2_anc)/2.0

                            tx = (cx - cxa) / (x2_anc - x1_anc)
                            ty = (cy - cya) / (y2_anc - y1_anc)
                            tw = np.log((gta[bbox_num, 1] - gta[bbox_num, 0]) / (x2_anc - x1_anc))
                            th = np.log((gta[bbox_num, 3] - gta[bbox_num, 2]) / (y2_anc - y1_anc))
                
                        if img_data['bboxes'][bbox_num]['class'] != 'bg':
                            # 如果此时IOU是真实框目前最大的，记录与bbox对应IOU最大的anchor
                            if curr_iou > best_iou_for_bbox[bbox_num]:
                                best_anchor_for_bbox[bbox_num] = [jy, ix, anchor_ratio_idx, anchor_size_idx]
                                best_iou_for_bbox[bbox_num] = curr_iou
                                best_x_for_bbox[bbox_num,:] = [x1_anc, x2_anc, y1_anc, y2_anc]
                                best_dx_for_bbox[bbox_num,:] = [tx, ty, tw, th]
                
                            #IOU>7，正样本
                            if curr_iou > C.rpn_max_overlap:
                                bbox_type = 'pos'
                                num_anchors_for_bbox[bbox_num] += 1
                                # 如果当前anchor与多个真实框的IOU都大于阈值，记录IOU最大的偏移值
                                if curr_iou > best_iou_for_loc:
                                    best_iou_for_loc = curr_iou
                                    best_regr = (tx, ty, tw, th)

                            # 0.3<IOU<0.7,忽略
                            if C.rpn_min_overlap < curr_iou < C.rpn_max_overlap:
                                if bbox_type != 'pos':
                                    bbox_type = 'neutral'

                    # 根据正负样本来设置输出，y_is_box_valid，[fh,fw,n_anchors]anchor是否有效(正负样本)，第jy，ix位置的第n个anchor
                    # y_rpn_overlap，是否为目标[fh,fw,n_anchors]
                    # y_rpn_regr[fh,fw,n_anchors*4],如果是正样本，需要记录偏移值
                    if bbox_type == 'neg':
                        y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
                        y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
                    elif bbox_type == 'neutral':
                        y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
                        y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
                    elif bbox_type == 'pos':
                        y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
                        y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
                        start = 4 * (anchor_ratio_idx + n_anchratios * anchor_size_idx)
                        y_rpn_regr[jy, ix, start:start+4] = best_regr

    # 确保每一个bbox都有至少一个正RPN区域与之对应
    for idx in range(num_anchors_for_bbox.shape[0]):
        # 如果之前iou>0.7没有匹配到anchor
        if num_anchors_for_bbox[idx] == 0:
            # 没有与之IOU大于0的anchor
            if best_anchor_for_bbox[idx, 0] == -1:
                continue
            # 设置与之IOU最大的anchor为正样本
            y_is_box_valid[best_anchor_for_bbox[idx,0], best_anchor_for_bbox[idx,1], 
                           best_anchor_for_bbox[idx,2] + n_anchratios * best_anchor_for_bbox[idx,3]] = 1
            y_rpn_overlap[best_anchor_for_bbox[idx,0], best_anchor_for_bbox[idx,1], 
                           best_anchor_for_bbox[idx,2] + n_anchratios * best_anchor_for_bbox[idx,3]] = 1
            start = 4 * (best_anchor_for_bbox[idx,2] + n_anchratios * best_anchor_for_bbox[idx,3])
            y_rpn_regr[best_anchor_for_bbox[idx,0], best_anchor_for_bbox[idx,1], start:start+4] = best_dx_for_bbox[idx, :]



    y_rpn_overlap = np.expand_dims(y_rpn_overlap, axis=0) # 维度[1,fh,fw,n_anchors]
    y_is_box_valid = np.expand_dims(y_is_box_valid, axis=0)
    y_rpn_regr = np.expand_dims(y_rpn_regr, axis=0) 
    pos_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 1, y_is_box_valid[0, :, :, :] == 1))
    neg_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 0, y_is_box_valid[0, :, :, :] == 1))    
    num_pos = len(pos_locs[0])

	#调节正负样本的比例

    #  we randomly sample 256 anchors in an image to compute the loss function of a mini-batch, where
    #  the sampled positive and negative anchors have a ratio of up to 1:1. If there are fewer than 128 positive
    #  samples in an image, we pad the mini-batch with negative ones.

    num_regions = 256

    if len(pos_locs[0]) > num_regions/2:
        val_locs = random.sample(range(len(pos_locs[0])), len(pos_locs[0]) - num_regions/2)
        y_is_box_valid[0, pos_locs[0][val_locs], pos_locs[1][val_locs], pos_locs[2][val_locs]] = 0
        num_pos = num_regions/2

    if len(neg_locs[0]) + num_pos > num_regions:
        # val_locs = random.sample(range(len(neg_locs[0])), len(neg_locs[0]) - num_pos)
        val_locs = random.sample(range(len(neg_locs[0])),  len(neg_locs[0])-(num_regions - num_pos) )       
        y_is_box_valid[0, neg_locs[0][val_locs], neg_locs[1][val_locs], neg_locs[2][val_locs]] = 0

    y_rpn_cls = np.concatenate([y_is_box_valid, y_rpn_overlap], axis=3) # 维度[1,fh,fw,n_anchors][1,fh,fw,n_anchors]
    y_rpn_regr = np.concatenate([np.repeat(y_rpn_overlap, 4, axis=3), y_rpn_regr], axis=3) # 维度[1,fh,fw,4n_anchors][1,fh,fw,4n_anchors]
    return np.copy(y_rpn_cls), np.copy(y_rpn_regr)


class threadsafe_iter:
	"""Takes an iterator/generator and makes it thread-safe by
	serializing call to the `next` method of given iterator/generator.
	"""
	def __init__(self, it):
		self.it = it
		self.lock = threading.Lock()

	def __iter__(self):
		return self

	def next(self):
		with self.lock:
			return next(self.it)		

	
def threadsafe_generator(f):
	"""A decorator that takes a generator function and makes it thread-safe.
	"""
	def g(*a, **kw):
		return threadsafe_iter(f(*a, **kw))
	return g



def get_anchor_gt(all_img_data, class_count, C, img_length_calc_function, backend, mode='train'):
    '''
    生成anchor
    
    返回：图像，[类别,偏移值]，图像属性
    '''
    sample_selector = SampleSelector(class_count)

    while True:
        #如果是训练模式，打乱训练集
        if mode == 'train':
            np.random.shuffle(all_img_data)

        # 一次处理一张图像，这样batch是1，应该可以修改代码
        for img_data in all_img_data:
            try:

                if C.balanced_classes and sample_selector.skip_sample_for_balanced_class(img_data):
                    continue

                # 读取图像，图像增强（训练模式），返回标签和图像
                if mode == 'train':
                    img_data_aug, x_img = data_augment.augment(img_data, C, augment=True)
                else:
                    img_data_aug, x_img = data_augment.augment(img_data, C, augment=False)

                (width, height) = (img_data_aug['width'], img_data_aug['height'])
                (rows, cols, _) = x_img.shape

                assert cols == width
                assert rows == height

                # 得到resize之后图像的维度
                (resized_width, resized_height) = get_new_img_size(width, height, C.im_size)

                # 根据维度改变图像，短边600px，INTER_CUBIC立方插值
                x_img = cv2.resize(x_img, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)

                # 得到anchor正负样本标记，以及正样本对应的偏移值
                try:
                    y_rpn_cls, y_rpn_regr = calc_rpn(C, img_data_aug, width, height, resized_width, resized_height, img_length_calc_function)
                except:
                    continue

                # 预处理图像，减均值
                x_img = x_img[:,:, (2, 1, 0)]  # BGR -> RGB
                x_img = x_img.astype(np.float32)
                x_img[:, :, 0] -= C.img_channel_mean[0]
                x_img[:, :, 1] -= C.img_channel_mean[1]
                x_img[:, :, 2] -= C.img_channel_mean[2]
                x_img /= C.img_scaling_factor        
                x_img = np.expand_dims(x_img, axis=0)
                y_rpn_regr[:, :, :, y_rpn_regr.shape[3]//2:] *= C.std_scaling        
                yield np.copy(x_img), [np.copy(y_rpn_cls), np.copy(y_rpn_regr)], img_data_aug
                
            except Exception as e:
                print(e)
                continue
