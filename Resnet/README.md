#### 1.教程
- ###### CNN原理
  1. 卷积**conv**的意义：[1:52:40]
  提取特征
  图片A -**filter**卷积核B1-> 图片A1，A1保留了对应特征
  通过不同的卷积核，强调了不同的特征
  2. 学习特征：
  初始化随机卷积核，让计算机自己计算出需要的卷积核，即舍弃xx特征，需要xx特征
  训练过程：训练集 源图A -卷积核X-> 标签A'
  3. ***术语*** :
  **步长 strides**: `每次卷积，卷积核移动的距离`
  **filters channel**：`输入通过n个filters，得到n个results，把结果堆叠在一起，得到具有n个通道的很'厚'的图片块，即ouput channel`
  **pooling**:  ``
  **全连接层 fully connect**: ` `
  **cross-entropy**: ` `
  **batch-normalization**: ` `
  4. 加速：
  filter遍历图像块串行卷积计算 -GPU-> 多个图像区块并行计算
- ###### ResNet
  1. CNN问题:
  - VGG：有很多个卷积层组成
    - 每经过一层卷积，即图像经过一次处理后，值变小，最终得到的结果接近0
    - 模型层数深的原因：每经过一层卷积，即提取了一次特征，希望提取越来越细致的特征，但越细致的特征，图像对应位置上显现该特征的强度就会越低
    - 为使得特征信息在卷积层种继续传递下去，由此出现残差网络
  2. ResNet:


- ###### 知乎代码逐行讲解
  [知乎](https://www.zhihu.com/zvideo/1317247224532119552)
  [pythorch_resnet](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py)

#### 2.论文阅读
https://arxiv.org/pdf/1512.03385.pdf



#### 3.resnet实战
数据集：[imageNet](https://image-net.org/)
