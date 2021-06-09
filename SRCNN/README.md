## SRCNN Super-Resolution
### 使用pytorch实现SRCNN 训练 / 测试

- 论文地址：[SRCNN paper](http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html)
- 使用pytorch实现SRCNN 训练与测试
- 基于SRCNN的多源信息融合 解决超分辨率问题
- 数据集：
    - 多光谱序列以及对应的 RGB 序列
    - [百度云](https://pan.baidu.com/s/158ehIQl9iMdlfaO4wS1pDQ) 提取码：ol5z
    - 多光谱相机集数据速度为 30 帧/秒，每帧图像包含 460-630nm 区间段的 16 个波段信息
    - RGB 和多光谱数据已作匹配，这也导致了图像清晰度的降低
    - 每组数据中，包含多光谱源数据、RGB 图以及多光谱伪彩色图，均给出目标真实状态[𝑥, 𝑦, 𝑤, ℎ] 
    ![图片]()