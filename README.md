学习笔记

# [卷积神经网络概念与原理](https://blog.csdn.net/yunpiao123456/article/details/52437794)
## 一. 卷积神经网络
### 1. 卷积层
卷积神经网络有两种神器可以降低参数数目:
- 局部感知野
- 权值共享

多卷积核：单个卷积核特征提取是不充分的，我们可以添加多个卷积核，获取不同的特征。

### 2. 激活函数
    如果不使用非线性激活函数，那么每一层输出都是上层输入的线性组合；
    增加模型的非线性表达能力, 加强网络的表示能力，解决线性模型无法解决的问题
    
### 3. 池化层
    主要分平均池化和最大池化。
    进行下采样可以利用局部相关性可以减少后续的数据处理量，同时又保留有用信息。

### 4. [全连接层](https://zhuanlan.zhihu.com/p/33841176)
    把特征representation整合到一起，输出为一个值,就大大减少特征位置对分类带来的影响
    全连接层之前的作用是提取特征,全理解层的作用是分类
在实际应用中，往往使用多层卷积，然后再使用全连接层进行训练，多层卷积的目的是一层卷积学到的特征往往是局部的，层数越高，学到的特征就越全局化

## 二. 卷积神经网络的训练
- 前向传播
- 反向传播

卷积神经网络中卷积层的权重更新过程本质是卷积核的更新过程。

## 三.卷积神经网络演变史



参考：
- [卷积神经网络概念与原理](https://blog.csdn.net/yunpiao123456/article/details/52437794)
- [卷积神经网络CNN总结](https://www.cnblogs.com/skyfsm/p/6790245.html)
- [什么是全连接层](https://zhuanlan.zhihu.com/p/33841176)
- [CNN网络架构演进：从LeNet到DenseNet](https://www.cnblogs.com/skyfsm/p/8451834.html)


# 目标检测：
## Faster-RCNN:
![GitHub](./picture/faster_rcnn.jpg "GitHub,Social Coding")

### 1. 特征抽取
    通过卷积神经网络，如VGG-16，抽取特征，获取FeatureMap
### 2. RPN（RegionProposal Network， 区域生成网络）
    2.1 生成anchors，FeatureMap的每个点生成k个anchors
    2.2 用3*3的卷积核对上一步骤得到的FeatureMap做卷积计算，加入附近像素的信息
    2.3 分两路，一路用来判断候选框是前景还是背景，另一路用来确定候选框的位置
    2.3.1 用2k个卷积核对FeatureMap做卷积，得到2k个FeatureMap，先reshape成一维向量，然后softmax来判断是前景还是背景，然后reshape恢复为二维feature map
    2.3.2 # 用4k个1 * 1 * 512 的卷积核卷积，最后输出4k个数，这里的4是一个建议框的参数，即(x, y, w, h)  
    2.4 根据anchor的概率（rpn_cls_prob）和位置（bbox_pred）选出rois：按照前景概率得分筛出前TopN个bbox，再通过NMS后保留前TopN2 个bbox 
### 3. ROI Pooling层
    解决之前得到的proposal大小形状各不相同，导致没法做全连接
### 4. 分类层
    ROI Pooling层后的特征图，通过全连接层与softmax，就可以计算属于哪个具体类别，比如人，狗，飞机，并可以得到cls_prob概率向量。同时再次利用bounding box regression精细调整proposal位置，得到bbox_pred，用于回归更加精确的目标检测框。
    这样就完成了faster R-CNN的整个过程了。算法还是相当复杂的，对于每个细节需要反复理解。faster R-CNN使用resNet101模型作为卷积层，在voc2012数据集上可以达到83.8%的准确率，超过yolo ssd和yoloV2。其最大的问题是速度偏慢，每秒只能处理5帧，达不到实时性要求。



参考：
- [目标检测算法综述：R-CNN，faster R-CNN，yolo，SSD，yoloV2](https://www.imooc.com/article/37757)