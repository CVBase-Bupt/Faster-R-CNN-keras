#--coding:utf-8--
from __future__ import division
import random
import pprint
import sys
import os
import time
import numpy as np
import argparse
import pickle

from keras import backend as K
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Input
from keras.models import Model
from keras_frcnn import config, data_generators
from keras_frcnn import network_losses as losses
import keras_frcnn.network_aux_roi as roi_helpers
from keras.utils import generic_utils
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras_frcnn import network_model_vgg as nn
	

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
con = tf.ConfigProto()
con.gpu_options.per_process_gpu_memory_fraction = 1
set_session(tf.Session(config=con))

#con = tf.ConfigProto()
#con.gpu_options.allow_growth = True
#sess = tf.Session(config=con)

sys.setrecursionlimit(40000)

parser = argparse.ArgumentParser()
#训练集路径
parser.add_argument("-p", "--path", dest="train_path", help="Path to training data.")
#解析方式
parser.add_argument("-o", "--parser", dest="parser", help="Parser to use. One of simple or pascal_voc",
				default="pascal_voc")
#一次处理的RoI的数目
parser.add_argument("-n", "--num_rois", type=int,dest="num_rois", help="Number of RoIs to process at once.", default=32)
#基础网络结构
parser.add_argument("--network", dest="network", help="Base network to use. Supports vgg or resnet50.", default='resnet50')
#数据增强：水平翻转
parser.add_argument("--hf", dest="horizontal_flips", help="Augment with horizontal flips in training. (Default=false).", action="store_true", default=False)
#数据增强：垂直翻转
parser.add_argument("--vf", dest="vertical_flips", help="Augment with vertical flips in training. (Default=false).", action="store_true", default=False)
#数据增强：旋转
parser.add_argument("--rot", "--rot_90", dest="rot_90", help="Augment with 90 degree rotations in training. (Default=false).",
				  action="store_true", default=False)

parser.add_argument("--num_epochs", type=int, dest="num_epochs", help="Number of epochs.", default=2000)
# parser.add_argument("--epoch_length", type=int, dest="epoch_length", help="samples per eopch.", default=100)
# 配置文件存储路径
parser.add_argument("--config_filename", dest="config_filename", help=
				"Location to store all the metadata related to the training (to be used when testing).",
				default="config.pickle")
# 权值保存路径
parser.add_argument("--output_weight_path", dest="output_weight_path", help="Output path for weights.", default='./model_frcnn.hdf5')
# 权重导入路径
parser.add_argument("--input_weight_path", dest="input_weight_path", help="Input path for weights. If not specified, will try to load default weights provided by keras.")
# 解析命令行参数
args = parser.parse_args()

if not args.train_path:
	parser.error('Error: path to training data must be specified. Pass --path to command line')
if args.parser == 'pascal_voc':
	from keras_frcnn.data_voc_parser import get_data
elif args.parser == 'simple':
	from keras_frcnn.data_simple_parser import get_data
else:
	raise ValueError("Command line option parser must be one of 'pascal_voc' or 'simple'")


# 读取命令行参数，并记入配置文件
C = config.Config()

# 是否数据增强
C.use_horizontal_flips = bool(args.horizontal_flips)
C.use_vertical_flips = bool(args.vertical_flips)
C.rot_90 = bool(args.rot_90)

C.model_path = args.output_weight_path
C.num_rois = int(args.num_rois)

# 检查命令行是否传入权重路径
if args.input_weight_path:
	C.base_net_weights = args.input_weight_path
else:
# 基于模型设置权值路径
	C.base_net_weights = nn.get_weight_path()

# 获取数据
all_imgs, classes_count, class_mapping = get_data(args.train_path)
C.class_mapping = class_mapping


print('Training images per class:')
pprint.pprint(classes_count)
print('Num classes (including bg) = {}'.format(len(classes_count)))

config_output_filename = args.config_filename

# 将训练参数存入文件
with open(config_output_filename, 'wb') as config_f:
	pickle.dump(C,config_f)
	print('Config has been written to {}, and can be loaded when testing to ensure correct results'.format(config_output_filename))
#打乱图片的顺序
random.shuffle(all_imgs)
#全部图片的数量
num_imgs = len(all_imgs)
#训练集以及验证集
train_imgs = [s for s in all_imgs if s['imageset'] == 'trainval']
val_imgs = [s for s in all_imgs if s['imageset'] == 'test']

print('Num train samples {}'.format(len(train_imgs)))
print('Num val samples {}'.format(len(val_imgs)))

#生成训练集和验证集
data_gen_train = data_generators.get_anchor_gt(train_imgs, classes_count, C, nn.get_img_output_length, K.image_dim_ordering(), mode='train')
data_gen_val = data_generators.get_anchor_gt(val_imgs, classes_count, C, nn.get_img_output_length,K.image_dim_ordering(), mode='val')

K.set_learning_phase(1)

# tensorflow通道在最后一维
input_shape_img = (None, None, 3)

img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(None, 4))


# 1. 载入基础网络(可以是VGG,ResNet,Inception等)
shared_layers = nn.nn_base(img_input)

# 2. RPN
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn = nn.rpn(shared_layers, num_anchors)

# 3. RCNN
classifier = nn.classifier(shared_layers, roi_input, C.num_rois, nb_classes=len(classes_count))

# 4. base + RPN
model_rpn = Model(img_input, rpn[:2])

#    base + RCNN
model_classifier = Model([img_input, roi_input], classifier)

# 5. Fast R-CNN + RPN, 用来加载/存储权值
model_all = Model([img_input, roi_input], rpn[:2] + classifier)

#加载权重
try:
	print('loading weights from {}'.format(C.base_net_weights))
	model_rpn.load_weights(C.base_net_weights, by_name=True)
	model_classifier.load_weights(C.base_net_weights, by_name=True)
except:
	print('Could not load pretrained model weights. Weights can be found in the keras application folder \
		https://github.com/fchollet/keras/tree/master/keras/applications')

# 设置优化算法Adam，学习率
optimizer = Adam(lr=1e-5)
optimizer_classifier = Adam(lr=1e-5)

# 编译所有的网络
model_rpn.compile(optimizer=optimizer, loss=[losses.rpn_loss_cls(num_anchors), losses.rpn_loss_regr(num_anchors)])
model_classifier.compile(optimizer=optimizer_classifier, loss=[losses.class_loss_cls, losses.class_loss_regr(len(classes_count)-1)], metrics={'dense_class_{}'.format(len(classes_count)): 'accuracy'})
model_all.compile(optimizer='sgd', loss='mae')

# 训练时迭代数目
# epoch_length = int(args.epoch_length)
epoch_length = 100
num_epochs = int(args.num_epochs)
iter_num = 0

losses = np.zeros((epoch_length, 5))
rpn_accuracy_rpn_monitor = []
rpn_accuracy_for_epoch = []

#计时
start_time = time.time()

#初始化为一个无穷大的数值
best_loss = np.Inf 

class_mapping_inv = {v: k for k, v in class_mapping.items()}

print('Starting training')
vis = True

#训练
for epoch_num in range(num_epochs):
	progbar = generic_utils.Progbar(epoch_length)
	print('Epoch {}/{}'.format(epoch_num + 1, num_epochs))

	while True:
		try:
			if len(rpn_accuracy_rpn_monitor) == epoch_length and C.verbose:
				mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor))/len(rpn_accuracy_rpn_monitor)
				rpn_accuracy_rpn_monitor = []
				print('Average number of overlapping bounding boxes from RPN = {} for {} previous iterations'.format(mean_overlapping_bboxes, epoch_length))
				if mean_overlapping_bboxes == 0:
					print('RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')

			#样本，X是图像，Y是anchor正负样本以及偏移值标签，img_data图像真实标签
			X, Y, img_data = next(data_gen_train)
			        
			#训练RPN
			loss_rpn = model_rpn.train_on_batch(X, Y)

			#使用RPN预测
			P_rpn = model_rpn.predict_on_batch(X)

			# [tx,ty,tw,th]->[x1, y1, x2, y2]（相对conv5_3特征图坐标），去掉出界的框,nms
			R = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], C, K.image_dim_ordering(), use_regr=True, overlap_thresh=0.7, max_boxes=300)
			
	
			#X2 roi相对conv5_3特征图坐标 [x1, y1, w, h])
			#Y1 类别(one-hot编码） Y2 正样本坐标[sx*tx, sy*ty, sw*tw, sh*th] 以及类别[1, 1, 1, 1]
			#IoUs best_iou
			X2, Y1, Y2, IouS = roi_helpers.calc_iou(R, img_data, C, class_mapping)

			# 如果没有ROI
			if X2 is None:
				rpn_accuracy_rpn_monitor.append(0)
				rpn_accuracy_for_epoch.append(0)
				continue

			# 正负样本，np.where得到正负样本的下标，最后一个维度(-1)是背景类
			neg_samples = np.where(Y1[0, :, -1] == 1)
			pos_samples = np.where(Y1[0, :, -1] == 0)

			if len(neg_samples) > 0:
				neg_samples = neg_samples[0]
			else:
				neg_samples = []

			if len(pos_samples) > 0:
				# 相当于降维了
				pos_samples = pos_samples[0]
			else:
				pos_samples = []
			
			# 正样本的数量
			rpn_accuracy_rpn_monitor.append(len(pos_samples))
			rpn_accuracy_for_epoch.append((len(pos_samples)))

			#roi正负样本选择
			if C.num_rois > 1:
				# 如果正样本的数量少于roi数量的一半，就直接用这些正样本
				if len(pos_samples) < C.num_rois//2:
					selected_pos_samples = pos_samples.tolist()
				else:
					# 如果多于一半，随机选正样本
					selected_pos_samples = np.random.choice(pos_samples, C.num_rois//2, replace=False).tolist()
				try:
					# 负样本，样本不可重复出现
					selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples), replace=False).tolist()
				except:
					# 样本可重复出现
					selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples), replace=True).tolist()

				sel_samples = selected_pos_samples + selected_neg_samples
			else:
				# 极限情况num_rois = 1, 随机选一个正样本或负样本
				selected_pos_samples = pos_samples.tolist()
				selected_neg_samples = neg_samples.tolist()
				if np.random.randint(0, 2):
					sel_samples = random.choice(neg_samples)
				else:
					sel_samples = random.choice(pos_samples)

			#在标注好的正负样本roi上训练 model_classifier = Model([img_input, roi_input], classifier)
			loss_class = model_classifier.train_on_batch([X, X2[:, sel_samples, :]], [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])

			losses[iter_num, 0] = loss_rpn[1]
			losses[iter_num, 1] = loss_rpn[2]

			losses[iter_num, 2] = loss_class[1]
			losses[iter_num, 3] = loss_class[2]
			losses[iter_num, 4] = loss_class[3]

			iter_num += 1

			progbar.update(iter_num, [('rpn_cls', np.mean(losses[:iter_num, 0])), ('rpn_regr', np.mean(losses[:iter_num, 1])),
									  ('detector_cls', np.mean(losses[:iter_num, 2])), ('detector_regr', np.mean(losses[:iter_num, 3]))])

			if iter_num == epoch_length:
				loss_rpn_cls = np.mean(losses[:, 0])
				loss_rpn_regr = np.mean(losses[:, 1])
				loss_class_cls = np.mean(losses[:, 2])
				loss_class_regr = np.mean(losses[:, 3])
				class_acc = np.mean(losses[:, 4])

				mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
				rpn_accuracy_for_epoch = []

				if C.verbose:
					print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(mean_overlapping_bboxes))
					print('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
					print('Loss RPN classifier: {}'.format(loss_rpn_cls))
					print('Loss RPN regression: {}'.format(loss_rpn_regr))
					print('Loss Detector classifier: {}'.format(loss_class_cls))
					print('Loss Detector regression: {}'.format(loss_class_regr))
					print('Elapsed time: {}'.format(time.time() - start_time))

				curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
				iter_num = 0
				start_time = time.time()

				# 如果当前损失函数最小，更新模型
				if curr_loss < best_loss:
					if C.verbose:
						print('Total loss decreased from {} to {}, saving weights'.format(best_loss,curr_loss))
					best_loss = curr_loss
					model_all.save_weights(C.model_path)

				break

		except Exception as e:
			print('Exception: {}'.format(e))
			continue

print('Training complete, exiting.')
