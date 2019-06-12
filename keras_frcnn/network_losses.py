#--encoding: utf-8--
from keras import backend as K
from keras.objectives import categorical_crossentropy
import tensorflow as tf

lambda_rpn_regr = 1.0
lambda_rpn_class = 1.0

lambda_cls_regr = 1.0
lambda_cls_class = 1.0

epsilon = 1e-4

# RPN损失函数
# 回归 smoothL1(x)=(0.5x^2) 如果|x|<1 , 
#                  |x|-0.5  其它
def rpn_loss_regr(num_anchors):
	def rpn_loss_regr_fixed_num(y_true, y_pred):		
		# y_rpn_regr = np.concatenate([np.repeat(y_rpn_overlap, 4, axis=1), y_rpn_regr], axis=1)
		x = y_true[:, :, :, 4 * num_anchors:] - y_pred
		x_abs = K.abs(x)
		x_bool = K.cast(K.less_equal(x_abs, 1.0), tf.float32)
		return lambda_rpn_regr * K.sum(
				y_true[:, :, :, :4 * num_anchors] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + y_true[:, :, :, :4 * num_anchors])
	return rpn_loss_regr_fixed_num

# 分类 二项交叉熵
def rpn_loss_cls(num_anchors):
	def rpn_loss_cls_fixed_num(y_true, y_pred):
        # y_rpn_cls = np.concatenate([y_is_box_valid, y_rpn_overlap], axis=1)
		return lambda_rpn_class * K.sum(y_true[:, :, :, :num_anchors] * K.binary_crossentropy(y_pred[:, :, :, :], y_true[:, :, :, num_anchors:])) / K.sum(epsilon + y_true[:, :, :, :num_anchors])
	return rpn_loss_cls_fixed_num

# Fast R-CNN损失函数
# 回归
def class_loss_regr(num_classes):
	def class_loss_regr_fixed_num(y_true, y_pred):
        #Y = np.concatenate([np.array(y_class_regr_label),np.array(y_class_regr_coords)],axis=1)
		x = y_true[:, :, 4*num_classes:] - y_pred
		x_abs = K.abs(x)
		x_bool = K.cast(K.less_equal(x_abs, 1.0), 'float32')
		return lambda_cls_regr * K.sum(y_true[:, :, :4*num_classes] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + y_true[:, :, :4*num_classes])
	return class_loss_regr_fixed_num

# 分类 多项交叉熵
def class_loss_cls(y_true, y_pred):
    # 	Y = np.array(y_class_num)
	return lambda_cls_class * K.mean(categorical_crossentropy(y_true[0, :, :], y_pred[0, :, :]))
