## SRCNN Super-Resolution
### pytorch实现SRCNN 训练 / 测试
***
#### TODO:
1. train
2. test
3. 论文解读
***
- 论文地址：[Image Super-Resolution Using Deep Convolutional Networks](http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html)
- 使用**pytorch**实现SRCNN 训练与测试
- **基于SRCNN的多源信息融合** ***解决超分辨率问题***
- 数据集：
    - ***多光谱序列*** 以及对应的 ***RGB*** 序列
    - [百度云](https://pan.baidu.com/s/158ehIQl9iMdlfaO4wS1pDQ) 提取码：ol5z
    - 多光谱相机集数据速度为 ***30*** 帧/秒，每帧图像包含 460-630nm 区间段的 16 个波段信息
    - RGB 和多光谱数据已作匹配，这也导致了图像**清晰度的降低**
    - 每组数据中，包含**多光谱源数据**、**RGB 图**以及**多光谱伪彩色图**，均给出目标真实状态[𝑥, 𝑦, 𝑤, ℎ] 
    ![图片]()
***
#### 参考
- [(SRCNN)及pytorch实现_Learning a Deep Convolutional Network for Image Super-Resolution——超分辨率（二)](https://blog.csdn.net/xu_fu_yong/article/details/96434132)
    - https://github.com/fuyongXu/SRCNN_Pytorch_1.0
- [超分辨率SRCNN理解（附pytorch代码）](https://blog.csdn.net/zzy_pphz/article/details/108408053)
- https://github.com/daikun/SRCNN-pytorch
- [SRCNN_Learning](https://gitee.com/vegee/SRCNN_Learning)

<br>