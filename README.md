Faster R-CNN的keras版本代码：

训练自己的数据：

制作自己的标签：txt文件，文件格式 每一行为：图像路径 xmin, ymin, xmax, ymax, class_name

python train_frcnn.py -p txt文件路径 -o simple --input_weight_path 预训练权值路径

# Faster-R-CNN-keras
