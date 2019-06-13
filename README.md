# Faster-R-CNN-keras

Faster R-CNN的keras版本代码：

使用自己的数据训练网络：

1.制作标签：txt文件，文件格式 每一行为：图像路径 xmin, ymin, xmax, ymax, class_name       
2.python train_frcnn.py -p txt文件路径 -o simple --input_weight_path 预训练权值路径

