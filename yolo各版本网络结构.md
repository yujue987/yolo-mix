# 各版本网络结构
## v1
```yaml
# YOLOv1 网络结构配置文件
# 参考: You Only Look Once: Unified, Real-Time Object Detection

# 参数配置
nc: 20  # 默认使用PASCAL VOC的20个类别
depth_multiple: 1.0  # 模型深度倍数
width_multiple: 1.0  # 层通道数倍数

# 骨干网络 (Backbone)
backbone:
  # [from, number, module, args]
  - [-1, 1, Conv, [64, 7, 2, 3]]  # 0-P1/2
  - [-1, 1, MaxPool, [2, 2]]       # 1-P2/4
  - [-1, 1, Conv, [192, 3, 1, 1]]  # 2
  - [-1, 1, MaxPool, [2, 2]]       # 3-P3/8
  - [-1, 1, Conv, [128, 1, 1, 0]]  # 4
  - [-1, 1, Conv, [256, 3, 1, 1]]  # 5
  - [-1, 1, Conv, [256, 1, 1, 0]]  # 6
  - [-1, 1, Conv, [512, 3, 1, 1]]  # 7
  - [-1, 1, MaxPool, [2, 2]]       # 8-P4/16
  - [-1, 4, ConvBlock, [256, 512]] # 9-12 (重复4次)
  - [-1, 1, Conv, [512, 1, 1, 0]]  # 13
  - [-1, 1, Conv, [1024, 3, 1, 1]] # 14
  - [-1, 1, MaxPool, [2, 2]]       # 15-P5/32
  - [-1, 2, ConvBlock, [512, 1024]] # 16-17 (重复2次)
  - [-1, 1, Conv, [1024, 3, 1, 1]] # 18
  - [-1, 1, Conv, [1024, 3, 2, 1]] # 19
  - [-1, 1, Conv, [1024, 3, 1, 1]] # 20
  - [-1, 1, Conv, [1024, 3, 1, 1]] # 21

# 检测头 (Head)
head:
  - [-1, 1, Flatten, []]           # 22 展平层
  - [-1, 1, Linear, [4096]]        # 23 全连接层
  - [-1, 1, Dropout, [0.5]]        # 24 Dropout层
  - [-1, 1, Linear, [1470]]        # 25 输出层: 7*7*(2*5+20)=1470
  - [-1, 1, Reshape, [7, 7, 30]]   # 26 重塑为最终输出格式
  - [[26], 1, YOLOv1Detect, [nc]]  # 27 检测层
```
![v1](pict\v1.png)

### YOLOv1 介绍

#### Backbone
- Conv [64,7,2,3]: 初始大核卷积层，利用7x7内核和步幅2快速降低空间分辨率，捕获全局低级特征，有助于后续层聚焦更重要区域。
- MaxPool [2,2]: 最大池化层，进一步减小特征图尺寸，增强特征的空间不变性，减少计算量。
- Conv [192,3,1,1]: 3x3卷积层，提升通道数，增强特征表达能力，捕捉更复杂的空间模式。
- MaxPool [2,2]: 再次池化，压缩空间维度，提升模型对尺度变化的适应性。
- Conv [128,1,1,0]: 1x1卷积，调整通道数，压缩特征，降低参数量，提升非线性表达。
- Conv [256,3,1,1]: 3x3卷积，提取更细粒度的局部特征，增强模型对目标细节的捕捉。
- Conv [256,1,1,0]: 1x1卷积，进一步压缩通道，减少冗余信息。
- Conv [512,3,1,1]: 3x3卷积，提升特征深度，增强模型对复杂结构的识别。
- MaxPool [2,2]: 池化层，继续降低空间分辨率，提升特征抽象能力。
- ConvBlock [256,512] (重复4次): 多层卷积块，模拟残差结构，提升梯度流动，增强深层特征学习能力。
- Conv [512,1,1,0]: 1x1卷积，通道压缩，提升计算效率。
- Conv [1024,3,1,1]: 深层卷积，捕捉高阶语义信息。
- MaxPool [2,2]: 池化，进一步抽象特征。
- ConvBlock [512,1024] (重复2次): 更深卷积块，提升模型对复杂场景的表达。
- Conv [1024,3,1,1]: 3x3卷积，增强特征融合。
- Conv [1024,3,2,1]: 步幅卷积，缩小特征图，聚合空间信息。
- Conv [1024,3,1,1]: 最终卷积，输出高维语义特征，为检测头做准备。

#### Head
- Flatten: 展平特征图为向量，便于后续全连接层处理。
- Linear [4096]: 大规模全连接层，整合全局空间信息，提升模型对整体场景的理解。
- Dropout [0.5]: 随机丢弃部分神经元，减少过拟合风险，提升泛化能力。
- Linear [1470]: 输出层，生成每个网格的检测结果，包括类别、置信度和边界框参数。
- Reshape [7,7,30]: 将输出重塑为7x7网格，每个网格对应30维特征，便于空间定位。
- YOLOv1Detect: 检测层，负责解码边界框和类别概率，实现端到端目标检测。

---
## v2
```yaml
# YOLOv2 (YOLO9000) 网络结构配置文件
# 参考: YOLO9000: Better, Faster, Stronger

# 参数配置
nc: 80  # COCO数据集80个类别
depth_multiple: 1.0
width_multiple: 1.0
anchors:  # YOLOv2使用的先验框尺寸(COCO数据集)
  - [1.3221, 1.73145]
  - [3.19275, 4.00944]
  - [5.05587, 8.09892]
  - [9.47112, 4.84053]
  - [11.2364, 10.0071]

# Darknet-19 骨干网络
backbone:
  # [from, number, module, args]
  - [-1, 1, Conv, [32, 3, 1, 1]]   # 0
  - [-1, 1, MaxPool, [2, 2]]       # 1-P1/2
  - [-1, 1, Conv, [64, 3, 1, 1]]   # 2
  - [-1, 1, MaxPool, [2, 2]]       # 3-P2/4
  - [-1, 1, Conv, [128, 3, 1, 1]]  # 4
  - [-1, 1, Conv, [64, 1, 1, 0]]   # 5
  - [-1, 1, Conv, [128, 3, 1, 1]]  # 6
  - [-1, 1, MaxPool, [2, 2]]       # 7-P3/8
  - [-1, 1, Conv, [256, 3, 1, 1]]  # 8
  - [-1, 1, Conv, [128, 1, 1, 0]]  # 9
  - [-1, 1, Conv, [256, 3, 1, 1]]  # 10
  - [-1, 1, MaxPool, [2, 2]]       # 11-P4/16
  - [-1, 1, Conv, [512, 3, 1, 1]]  # 12
  - [-1, 1, Conv, [256, 1, 1, 0]]  # 13
  - [-1, 1, Conv, [512, 3, 1, 1]]  # 14
  - [-1, 1, Conv, [256, 1, 1, 0]]  # 15
  - [-1, 1, Conv, [512, 3, 1, 1]]  # 16
  - [-1, 1, MaxPool, [2, 2]]       # 17-P5/32
  - [-1, 1, Conv, [1024, 3, 1, 1]] # 18
  - [-1, 1, Conv, [512, 1, 1, 0]]  # 19
  - [-1, 1, Conv, [1024, 3, 1, 1]] # 20
  - [-1, 1, Conv, [512, 1, 1, 0]]  # 21
  - [-1, 1, Conv, [1024, 3, 1, 1]] # 22

# YOLOv2 检测头
head:
  - [-1, 1, Conv, [1024, 3, 1, 1]] # 23
  - [-1, 1, Conv, [1024, 3, 1, 1]] # 24
  
  # Passthrough层 (特征重组，将高分辨率特征与语义特征融合)
  - [16, 1, Passthrough, []]       # 25 从第16层获取特征
  
  - [[24, 25], 1, Concat, [1]]     # 26 特征拼接
  - [-1, 1, Conv, [1024, 3, 1, 1]] # 27
  
  # 检测层: 输出维度为 (batch, height, width, anchors*(5+classes))
  # 对于COCO: 13*13*5*(5+80) = 13*13*425
  - [-1, 1, Conv, [len(anchors)*(5+nc), 1, 1, 0]] # 28
  
  # YOLOv2检测层
  - [[28], 1, YOLOv2Detect, [anchors, nc]] # 29
```
![v2](D:\Code\ob\code\pict\v2.jpg)
### YOLOv2 介绍

#### Backbone (Darknet-19)
- Conv [32,3,1,1]: 初始3x3卷积，捕捉图像基础纹理和边缘特征，为后续层提供丰富底层信息。
- MaxPool [2,2]: 池化减小尺寸，提升特征抽象能力，减少计算量。
- Conv [64,3,1,1]: 增加通道，增强特征表达，提升模型对复杂结构的识别。
- MaxPool [2,2]: 池化，进一步压缩空间维度，提升尺度适应性。
- Conv [128,3,1,1]: 深层卷积，提取更复杂空间模式。
- Conv [64,1,1,0]: 1x1卷积压缩通道，降低参数量，提升非线性表达。
- Conv [128,3,1,1]: 3x3卷积，增强局部特征捕捉。
- MaxPool [2,2]: 池化，继续抽象特征。
- Conv [256,3,1,1]: 卷积，提升特征深度。
- Conv [128,1,1,0]: 1x1卷积，通道压缩，减少冗余。
- Conv [256,3,1,1]: 3x3卷积，融合多尺度信息。
- MaxPool [2,2]: 池化，提升抽象层次。
- Conv [512,3,1,1]: 深层卷积，捕捉高阶语义。
- Conv [256,1,1,0]: 1x1卷积，压缩通道，提升效率。
- Conv [512,3,1,1]: 3x3卷积，增强特征融合。
- Conv [256,1,1,0]: 1x1卷积，进一步压缩。
- Conv [512,3,1,1]: 3x3卷积，提升表达能力。
- MaxPool [2,2]: 池化，继续抽象。
- Conv [1024,3,1,1]: 深层卷积，捕捉全局语义。
- Conv [512,1,1,0]: 1x1卷积，通道压缩。
- Conv [1024,3,1,1]: 3x3卷积，增强特征融合。
- Conv [512,1,1,0]: 1x1卷积，压缩通道。
- Conv [1024,3,1,1]: 3x3卷积，输出高维语义特征。

#### Head
- Conv [1024,3,1,1]: 深层卷积，提升检测头对高阶语义的表达能力。
- Conv [1024,3,1,1]: 再次卷积，增强特征融合与空间信息整合。
- Passthrough: 特征重组，将高分辨率特征与深层语义特征融合，提升小目标检测能力。
- Concat: 拼接特征，实现多尺度信息整合，增强检测鲁棒性。
- Conv [1024,3,1,1]: 卷积，进一步融合多尺度特征。
- Conv [len(anchors)*(5+nc),1,1,0]: 输出卷积，生成每个anchor的检测结果，包括类别、置信度和边界框参数。
- YOLOv2Detect: 检测层，利用anchors机制预测边界框，实现高效目标定位。

---
## v3

```yaml
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# Ultralytics YOLOv3 object detection model with P3/8 - P5/32 outputs
# Model docs: https://docs.ultralytics.com/models/yolov3
# Task docs: https://docs.ultralytics.com/tasks/detect

# Parameters
nc: 80 # number of classes
depth_multiple: 1.0 # model depth multiple
width_multiple: 1.0 # layer channel multiple

# darknet53 backbone
backbone:
  # [from, number, module, args]
  - [-1, 1, Conv, [32, 3, 1]] # 0
  - [-1, 1, Conv, [64, 3, 2]] # 1-P1/2
  - [-1, 1, Bottleneck, [64]]
  - [-1, 1, Conv, [128, 3, 2]] # 3-P2/4
  - [-1, 2, Bottleneck, [128]]
  - [-1, 1, Conv, [256, 3, 2]] # 5-P3/8
  - [-1, 8, Bottleneck, [256]]
  - [-1, 1, Conv, [512, 3, 2]] # 7-P4/16
  - [-1, 8, Bottleneck, [512]]
  - [-1, 1, Conv, [1024, 3, 2]] # 9-P5/32
  - [-1, 4, Bottleneck, [1024]] # 10

# YOLOv3 head
head:
  - [-1, 1, Bottleneck, [1024, False]]
  - [-1, 1, Conv, [512, 1, 1]]
  - [-1, 1, Conv, [1024, 3, 1]]
  - [-1, 1, Conv, [512, 1, 1]]
  - [-1, 1, Conv, [1024, 3, 1]] # 15 (P5/32-large)

  - [-2, 1, Conv, [256, 1, 1]]
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 8], 1, Concat, [1]] # cat backbone P4
  - [-1, 1, Bottleneck, [512, False]]
  - [-1, 1, Bottleneck, [512, False]]
  - [-1, 1, Conv, [256, 1, 1]]
  - [-1, 1, Conv, [512, 3, 1]] # 22 (P4/16-medium)

  - [-2, 1, Conv, [128, 1, 1]]
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]] # cat backbone P3
  - [-1, 1, Bottleneck, [256, False]]
  - [-1, 2, Bottleneck, [256, False]] # 27 (P3/8-small)

  - [[27, 22, 15], 1, Detect, [nc]] # Detect(P3, P4, P5)

```
![v3](D:\Code\ob\code\pict\v3.png)
### YOLOv3 介绍

#### Backbone (darknet53)
- Conv [32,3,1]: 初始卷积，捕捉图像基础特征，为后续层提供丰富底层信息。
- Conv [64,3,2]: 步幅卷积，快速降低空间分辨率，提升计算效率。
- Bottleneck [64]: 残差瓶颈块，缓解梯度消失，提升深层特征学习能力。
- Conv [128,3,2]: 卷积，进一步压缩空间维度，增强特征抽象。
- Bottleneck [128] (x2): 多层残差结构，提升表达能力和模型稳定性。
- Conv [256,3,2]: 卷积，聚合空间信息。
- Bottleneck [256] (x8): 深层残差堆叠，增强模型对复杂结构的识别。
- Conv [512,3,2]: 卷积，提升特征深度。
- Bottleneck [512] (x8): 多层残差，提升高阶语义表达。
- Conv [1024,3,2]: 卷积，进一步抽象特征。
- Bottleneck [1024] (x4): 深层残差，捕捉全局语义信息。

#### Head
- Bottleneck [1024, False]: 非残差瓶颈块，整合高阶语义特征，提升检测头表达能力。
- Conv [512,1,1]: 1x1卷积，压缩通道，提升计算效率。
- Conv [1024,3,1]: 3x3卷积，增强空间信息融合。
- Conv [512,1,1]: 1x1卷积，进一步压缩通道。
- Conv [1024,3,1]: 3x3卷积，输出高维特征，为大目标检测做准备。
- Conv [256,1,1]: 1x1卷积，通道压缩，便于多尺度融合。
- nn.Upsample: 上采样，提升特征分辨率，增强小目标检测能力。
- Concat: 拼接P4，实现多尺度信息整合。
- Bottleneck [512, False] (x2): 非残差瓶颈，提升融合特征表达。
- Conv [256,1,1]: 1x1卷积，压缩通道。
- Conv [512,3,1]: 3x3卷积，输出中尺度特征。
- Conv [128,1,1]: 1x1卷积，通道压缩。
- nn.Upsample: 上采样，提升分辨率。
- Concat: 拼接P3，多尺度融合。
- Bottleneck [256, False] (x3): 非残差瓶颈，提升小目标检测能力。
- Detect: 检测层，输出多尺度目标检测结果。

---
## v4
```yaml
# YOLOv4 网络结构配置文件
# 参考: YOLOv4: Optimal Speed and Accuracy of Object Detection

# 参数配置
nc: 80  # COCO数据集80个类别
depth_multiple: 1.0
width_multiple: 1.0
anchors:
  - [12, 16, 19, 36, 40, 28]   # P5/32
  - [36, 75, 76, 55, 72, 146]  # P4/16
  - [142, 110, 192, 243, 459, 401]  # P3/8

# CSPDarknet53 骨干网络
backbone:
  # [from, number, module, args]
  # 初始卷积层
  - [-1, 1, Conv, [32, 3, 1]]          # 0-P1/2
  - [-1, 1, Conv, [64, 3, 2]]          # 1-P2/4
  - [-1, 1, CSPBlock, [64]]            # 2
  - [-1, 1, Conv, [128, 3, 2]]         # 3-P3/8
  - [-1, 2, CSPBlock, [128]]           # 4-5
  - [-1, 1, Conv, [256, 3, 2]]         # 6-P4/16
  - [-1, 8, CSPBlock, [256]]           # 7-14
  - [-1, 1, Conv, [512, 3, 2]]         # 15-P5/32
  - [-1, 8, CSPBlock, [512]]           # 16-23
  - [-1, 1, SPP, [512, [5, 9, 13]]]    # 24 SPP模块

# 颈部网络 (Neck): SPP + PANet
neck:
  # 上采样路径 (自顶向下)
  - [-1, 1, Conv, [256, 1, 1]]         # 25
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]] # 26
  - [[-1, 14], 1, Concat, [1]]         # 27 连接第14层
  - [-1, 2, CSPBlock, [256, False]]    # 28-29
  
  - [-1, 1, Conv, [128, 1, 1]]         # 30
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]] # 31
  - [[-1, 5], 1, Concat, [1]]          # 32 连接第5层
  - [-1, 2, CSPBlock, [128, False]]    # 33-34 (P3/8-small)
  
  # 下采样路径 (自底向上)
  - [-1, 1, Conv, [256, 3, 2]]         # 35
  - [[-1, 29], 1, Concat, [1]]         # 36 连接第29层
  - [-1, 2, CSPBlock, [256, False]]    # 37-38 (P4/16-medium)
  
  - [-1, 1, Conv, [512, 3, 2]]         # 39
  - [[-1, 24], 1, Concat, [1]]         # 40 连接第24层
  - [-1, 2, CSPBlock, [512, False]]    # 41-42 (P5/32-large)

# YOLOv4 检测头
head:
  - [[34, 38, 42], 1, YOLOv4Detect, [nc, anchors]]  # 检测层(P3, P4, P5)
```
![v4](D:\Code\ob\code\pict\v4.jpg)
### YOLOv4 介绍

#### Backbone (CSPDarknet53)
- Conv [32,3,1]: 初始卷积，捕捉基础特征，为后续层提供丰富底层信息。
- Conv [64,3,2]: 步幅2卷积，快速降低空间分辨率，提升计算效率。
- CSPBlock [64]: CSP块，交叉阶段部分，分离梯度流，提升特征表达和模型稳定性。
- Conv [128,3,2]: 卷积，进一步压缩空间维度，增强特征抽象。
- CSPBlock [128] (x2): 多层CSP结构，提升表达能力和梯度流动。
- Conv [256,3,2]: 卷积，聚合空间信息。
- CSPBlock [256] (x8): 深层CSP堆叠，增强模型对复杂结构的识别。
- Conv [512,3,2]: 卷积，提升特征深度。
- CSPBlock [512] (x8): 多层CSP，提升高阶语义表达。
- SPP [512, [5,9,13]]: 空间金字塔池化，融合多尺度特征，提升模型对不同尺寸目标的检测能力。

#### Neck (SPP + PANet)
- Conv [256,1,1]: 1x1卷积，通道压缩，便于特征融合。
- nn.Upsample: 上采样，提升特征分辨率，增强小目标检测能力。
- Concat: 拼接，实现多尺度信息整合。
- CSPBlock [256, False] (x2): 无残差CSP块，提升融合特征表达。
- Conv [128,1,1]: 1x1卷积，进一步压缩通道。
- nn.Upsample: 上采样，提升分辨率。
- Concat: 拼接，实现多尺度融合。
- CSPBlock [128, False] (x2): (P3)无残差CSP，提升小目标表达。
- Conv [256,3,2]: 下采样，降低分辨率，聚合空间信息。
- Concat: 拼接，实现中尺度特征融合。
- CSPBlock [256, False] (x2): (P4)无残差CSP，提升中尺度目标表达。
- Conv [512,3,2]: 下采样，进一步降低分辨率。
- Concat: 拼接，实现大尺度特征融合。
- CSPBlock [512, False] (x2): (P5)无残差CSP，提升大目标表达。

#### Head
- YOLOv4Detect: 检测层，利用anchors机制进行多尺度目标检测，提升定位精度和鲁棒性。

---
## v5
```yaml
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# Ultralytics YOLOv5 object detection model with P3/8 - P5/32 outputs
# Model docs: https://docs.ultralytics.com/models/yolov5
# Task docs: https://docs.ultralytics.com/tasks/detect

# Parameters
nc: 80 # number of classes
scales: # model compound scaling constants, i.e. 'model=yolov5n.yaml' will call yolov5.yaml with scale 'n'
  # [depth, width, max_channels]
  n: [0.33, 0.25, 1024]
  s: [0.33, 0.50, 1024]
  m: [0.67, 0.75, 1024]
  l: [1.00, 1.00, 1024]
  x: [1.33, 1.25, 1024]

# YOLOv5 v6.0 backbone
backbone:
  # [from, number, module, args]
  - [-1, 1, Conv, [64, 6, 2, 2]] # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4
  - [-1, 3, C3, [128]]
  - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8
  - [-1, 6, C3, [256]]
  - [-1, 1, Conv, [512, 3, 2]] # 5-P4/16
  - [-1, 9, C3, [512]]
  - [-1, 1, Conv, [1024, 3, 2]] # 7-P5/32
  - [-1, 3, C3, [1024]]
  - [-1, 1, SPPF, [1024, 5]] # 9

# YOLOv5 v6.0 head
head:
  - [-1, 1, Conv, [512, 1, 1]]
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]] # cat backbone P4
  - [-1, 3, C3, [512, False]] # 13

  - [-1, 1, Conv, [256, 1, 1]]
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]] # cat backbone P3
  - [-1, 3, C3, [256, False]] # 17 (P3/8-small)

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 14], 1, Concat, [1]] # cat head P4
  - [-1, 3, C3, [512, False]] # 20 (P4/16-medium)

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 10], 1, Concat, [1]] # cat head P5
  - [-1, 3, C3, [1024, False]] # 23 (P5/32-large)

  - [[17, 20, 23], 1, Detect, [nc]] # Detect(P3, P4, P5)

```
![v5](D:\Code\ob\code\pict\v5.jpg)
### YOLOv5 介绍

#### Backbone
- Conv [64,6,2,2]: 大内核卷积，初始特征提取，利用6x6内核和步幅2快速降低空间分辨率，捕捉全局低级特征，有助于后续层聚焦重要区域。
- Conv [128,3,2]: 步幅卷积，进一步压缩空间维度，提升计算效率。
- C3 [128]: C3块，CSP-like结构，分离梯度流，提升特征表达和模型稳定性。
- Conv [256,3,2]: 卷积，聚合空间信息。
- C3 [256] (x6): 多层C3堆叠，增强模型对复杂结构的识别。
- Conv [512,3,2]: 卷积，提升特征深度。
- C3 [512] (x9): 深层C3，提升高阶语义表达。
- Conv [1024,3,2]: 卷积，进一步抽象特征。
- C3 [1024] (x3): 深层C3，捕捉全局语义信息。
- SPPF [1024,5]: 快速空间金字塔池化，融合多尺度特征，提升模型对不同尺寸目标的检测能力。

#### Head
- Conv [512,1,1]: 1x1卷积，通道压缩，便于特征融合。
- nn.Upsample: 上采样，提升特征分辨率，增强小目标检测能力。
- Concat: 拼接P4，实现多尺度信息整合。
- C3 [512, False] (x3): 无残差C3块，提升融合特征表达。
- Conv [256,1,1]: 1x1卷积，进一步压缩通道。
- nn.Upsample: 上采样，提升分辨率。
- Concat: 拼接P3，多尺度融合。
- C3 [256, False] (x3): (P3)无残差C3，提升小目标表达。
- Conv [256,3,2]: 下采样，降低分辨率，聚合空间信息。
- Concat: 拼接，实现中尺度特征融合。
- C3 [512, False] (x3): (P4)无残差C3，提升中尺度目标表达。
- Conv [512,3,2]: 下采样，进一步降低分辨率。
- Concat: 拼接，实现大尺度特征融合。
- C3 [1024, False] (x3): (P5)无残差C3，提升大目标表达。
- Detect: 检测层，输出多尺度目标检测结果。

---
## v6
```yaml
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# Meituan YOLOv6 object detection model with P3/8 - P5/32 outputs
# Model docs: https://docs.ultralytics.com/models/yolov6
# Task docs: https://docs.ultralytics.com/tasks/detect

# Parameters
nc: 80 # number of classes
activation: nn.ReLU() # (optional) model default activation function
scales: # model compound scaling constants, i.e. 'model=yolov6n.yaml' will call yolov8.yaml with scale 'n'
  # [depth, width, max_channels]
  n: [0.33, 0.25, 1024]
  s: [0.33, 0.50, 1024]
  m: [0.67, 0.75, 768]
  l: [1.00, 1.00, 512]
  x: [1.00, 1.25, 512]

# YOLOv6-3.0s backbone
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]] # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4
  - [-1, 6, Conv, [128, 3, 1]]
  - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8
  - [-1, 12, Conv, [256, 3, 1]]
  - [-1, 1, Conv, [512, 3, 2]] # 5-P4/16
  - [-1, 18, Conv, [512, 3, 1]]
  - [-1, 1, Conv, [1024, 3, 2]] # 7-P5/32
  - [-1, 6, Conv, [1024, 3, 1]]
  - [-1, 1, SPPF, [1024, 5]] # 9

# YOLOv6-3.0s head
head:
  - [-1, 1, Conv, [256, 1, 1]]
  - [-1, 1, nn.ConvTranspose2d, [256, 2, 2, 0]]
  - [[-1, 6], 1, Concat, [1]] # cat backbone P4
  - [-1, 1, Conv, [256, 3, 1]]
  - [-1, 9, Conv, [256, 3, 1]] # 14

  - [-1, 1, Conv, [128, 1, 1]]
  - [-1, 1, nn.ConvTranspose2d, [128, 2, 2, 0]]
  - [[-1, 4], 1, Concat, [1]] # cat backbone P3
  - [-1, 1, Conv, [128, 3, 1]]
  - [-1, 9, Conv, [128, 3, 1]] # 19

  - [-1, 1, Conv, [128, 3, 2]]
  - [[-1, 15], 1, Concat, [1]] # cat head P4
  - [-1, 1, Conv, [256, 3, 1]]
  - [-1, 9, Conv, [256, 3, 1]] # 23

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 10], 1, Concat, [1]] # cat head P5
  - [-1, 1, Conv, [512, 3, 1]]
  - [-1, 9, Conv, [512, 3, 1]] # 27

  - [[19, 23, 27], 1, Detect, [nc]] # Detect(P3, P4, P5)

```

![v6](D:\Code\ob\code\pict\v6.jpeg)
### YOLOv6 介绍

#### Backbone
- Conv [64,3,2]: 初始卷积，捕捉基础特征，快速降低空间分辨率。
- Conv [128,3,2]: 步幅卷积，进一步压缩空间维度，提升计算效率。
- Conv [128,3,1] (x6): 多层卷积，增强特征表达能力，提升模型对复杂结构的识别。
- Conv [256,3,2]: 卷积，聚合空间信息。
- Conv [256,3,1] (x12): 多层卷积，提升深层特征学习能力。
- Conv [512,3,2]: 卷积，提升特征深度。
- Conv [512,3,1] (x18): 多层卷积，增强高阶语义表达。
- Conv [1024,3,2]: 卷积，进一步抽象特征。
- Conv [1024,3,1] (x6): 多层卷积，捕捉全局语义信息。
- SPPF [1024,5]: 快速空间金字塔池化，融合多尺度特征，提升模型对不同尺寸目标的检测能力。

#### Head
- Conv [256,1,1]: 1x1卷积，通道压缩，便于特征融合。
- nn.ConvTranspose2d [256,2,2,0]: 转置卷积上采样，提升特征分辨率，增强小目标检测能力。
- Concat: 拼接P4，实现多尺度信息整合。
- Conv [256,3,1]: 3x3卷积，增强空间信息融合。
- Conv [256,3,1] (x9): 多层卷积，提升融合特征表达。
- Conv [128,1,1]: 1x1卷积，进一步压缩通道。
- nn.ConvTranspose2d [128,2,2,0]: 上采样，提升分辨率。
- Concat: 拼接P3，多尺度融合。
- Conv [128,3,1]: 3x3卷积，提升小目标表达。
- Conv [128,3,1] (x9): 多层卷积，增强小目标检测能力。
- Conv [128,3,2]: 下采样，降低分辨率，聚合空间信息。
- Concat: 拼接，实现中尺度特征融合。
- Conv [256,3,1]: 3x3卷积，提升中尺度目标表达。
- Conv [256,3,1] (x9): 多层卷积，增强中尺度检测能力。
- Conv [256,3,2]: 下采样，进一步降低分辨率。
- Concat: 拼接，实现大尺度特征融合。
- Conv [512,3,1]: 3x3卷积，提升大目标表达。
- Conv [512,3,1] (x9): 多层卷积，增强大目标检测能力。
- Detect: 检测层，输出多尺度目标检测结果。

---
## v7
```yaml
# YOLOv7 网络结构配置文件
# 参考: YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors

# 参数配置
nc: 80  # 类别数
depth_multiple: 1.0  # 模型深度倍数
width_multiple: 1.0  # 层通道数倍数
anchors:
  - [12,16, 19,36, 40,28]    # P3/8
  - [36,75, 76,55, 72,146]   # P4/16
  - [142,110, 192,243, 459,401]  # P5/32

# 骨干网络 (Backbone)
backbone:
  # [from, number, module, args]
  # 初始卷积层
  - [-1, 1, Conv, [32, 3, 1]]  # 0-P1/2
  - [-1, 1, Conv, [64, 3, 2]]  # 1-P2/4
  - [-1, 1, Conv, [64, 3, 1]]  # 2
  - [-1, 1, Conv, [128, 3, 2]] # 3-P3/8
  - [-1, 1, ELAN, [128, 64, 1]] # 4

  - [-1, 1, MPConv, []]        # 5-P4/16
  - [-1, 1, ELAN, [256, 128, 1]] # 6

  - [-1, 1, MPConv, []]        # 7-P5/32
  - [-1, 1, ELAN, [512, 256, 1]] # 8

  - [-1, 1, MPConv, []]        # 9
  - [-1, 1, ELAN, [512, 256, 1]] # 10

  - [-1, 1, SPPCSPC, [256]]    # 11

# 颈部网络 (Neck)
neck:
  # 上采样路径
  - [-1, 1, Conv, [128, 1, 1]]   # 12
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']] # 13
  - [[-1, 6], 1, Concat, [1]]    # 14 连接第6层
  - [-1, 1, ELAN, [256, 128, 1]] # 15

  - [-1, 1, Conv, [64, 1, 1]]    # 16
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']] # 17
  - [[-1, 4], 1, Concat, [1]]    # 18 连接第4层
  - [-1, 1, ELAN, [128, 64, 1]]  # 19 (P3/8-small)

  # 下采样路径
  - [-1, 1, MPConv, []]          # 20-P4/16
  - [[-1, 15], 1, Concat, [1]]   # 21 连接第15层
  - [-1, 1, ELAN, [256, 128, 1]] # 22 (P4/16-medium)

  - [-1, 1, MPConv, []]          # 23-P5/32
  - [[-1, 11], 1, Concat, [1]]   # 24 连接第11层
  - [-1, 1, ELAN, [512, 256, 1]] # 25 (P5/32-large)

# 检测头 (Head)
head:
  # 主检测头
  - [[19, 22, 25], 1, RepConv, []]  # 重参数化卷积
  - [[-3, -2, -1], 1, YOLOv7Detect, [nc, anchors]]   # 检测层

  # 辅助检测头 (仅训练时使用)
  - [[19, 22, 25], 1, RepConv, []]  # 重参数化卷积
  - [[-3, -2, -1], 1, YOLOv7AuxDetect, [nc, anchors]]   # 辅助检测层
```
![v7](D:\Code\ob\code\pict\v7.png)
### YOLOv7 介绍

#### Backbone
- Conv [32,3,1]: 初始卷积层，捕捉基础特征，为后续层提供丰富底层信息。
- Conv [64,3,2]: 步幅2卷积，快速降低空间分辨率，提升计算效率。
- Conv [64,3,1]: 普通卷积，增强特征表达。
- Conv [128,3,2]: 步幅2卷积，进一步压缩空间维度。
- ELAN [128,64,1]: ELAN模块，采用多分支结构，提升特征融合能力和梯度流动，增强模型表达力。
- MPConv: MPConv模块，混合池化与卷积操作，提升特征多样性和空间信息聚合。
- ELAN [256,128,1]: 更深层ELAN模块，进一步增强特征融合和表达。
- MPConv: MPConv模块，提升空间信息聚合能力。
- ELAN [512,256,1]: 深层ELAN模块，提升高阶语义表达。
- MPConv: MPConv模块，增强大目标特征。
- ELAN [512,256,1]: ELAN模块，整合多尺度信息，提升全局特征表达。
- SPPCSPC [256]: 空间金字塔池化与CSP结合，提升多尺度检测能力，增强对不同尺寸目标的适应性。

#### Neck
- Conv [128,1,1]: 1x1卷积，通道压缩，便于特征融合。
- nn.Upsample: 上采样，提升特征分辨率，增强小目标检测能力。
- Concat: 拼接，实现多尺度信息整合。
- ELAN [256,128,1]: ELAN模块，提升融合特征表达。
- Conv [64,1,1]: 1x1卷积，进一步压缩通道。
- nn.Upsample: 上采样，提升分辨率。
- Concat: 拼接，多尺度融合。
- ELAN [128,64,1]: ELAN模块，提升小目标表达。
- MPConv: MPConv模块，混合池化与卷积，增强小目标检测能力。
- Concat: 拼接，实现中尺度特征融合。
- ELAN [256,128,1]: ELAN模块，提升中尺度目标表达。
- MPConv: MPConv模块，增强中尺度检测能力。
- Concat: 拼接，实现大尺度特征融合。
- ELAN [512,256,1]: ELAN模块，提升大目标表达。

#### Head
- RepConv: 重参数化卷积，提升推理速度和表达能力。
- YOLOv7Detect: 检测层，输出多尺度目标检测结果。
- RepConv: 重参数化卷积，辅助训练时提升特征表达。
- YOLOv7AuxDetect: 辅助检测层，提升训练阶段的检测性能。

---
## v8
```yaml
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# Ultralytics YOLOv8 object detection model with P3/8 - P5/32 outputs
# Model docs: https://docs.ultralytics.com/models/yolov8
# Task docs: https://docs.ultralytics.com/tasks/detect

# Parameters
nc: 80 # number of classes
scales: # model compound scaling constants, i.e. 'model=yolov8n.yaml' will call yolov8.yaml with scale 'n'
  # [depth, width, max_channels]
  n: [0.33, 0.25, 1024] # YOLOv8n summary: 225 layers,  3157200 parameters,  3157184 gradients,   8.9 GFLOPs
  s: [0.33, 0.50, 1024] # YOLOv8s summary: 225 layers, 11166560 parameters, 11166544 gradients,  28.8 GFLOPs
  m: [0.67, 0.75, 768] # YOLOv8m summary: 295 layers, 25902640 parameters, 25902624 gradients,  79.3 GFLOPs
  l: [1.00, 1.00, 512] # YOLOv8l summary: 365 layers, 43691520 parameters, 43691504 gradients, 165.7 GFLOPs
  x: [1.00, 1.25, 512] # YOLOv8x summary: 365 layers, 68229648 parameters, 68229632 gradients, 258.5 GFLOPs

# YOLOv8.0n backbone
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]] # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4
  - [-1, 3, C2f, [128, True]]
  - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8
  - [-1, 6, C2f, [256, True]]
  - [-1, 1, Conv, [512, 3, 2]] # 5-P4/16
  - [-1, 6, C2f, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]] # 7-P5/32
  - [-1, 3, C2f, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]] # 9

# YOLOv8.0n head
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]] # cat backbone P4
  - [-1, 3, C2f, [512]] # 12

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]] # cat backbone P3
  - [-1, 3, C2f, [256]] # 15 (P3/8-small)

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 12], 1, Concat, [1]] # cat head P4
  - [-1, 3, C2f, [512]] # 18 (P4/16-medium)

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 9], 1, Concat, [1]] # cat head P5
  - [-1, 3, C2f, [1024]] # 21 (P5/32-large)

  - [[15, 18, 21], 1, Detect, [nc]] # Detect(P3, P4, P5)

```
![v8](D:\Code\ob\code\pict\v8.jpg)
### YOLOv8 介绍
#### Backbone
- Conv: 标准卷积层，用于初始特征提取，参数包括输出通道、内核大小和步幅。
- C2f: Cross Stage Partial with Focus模块，提高计算效率，通过部分连接减少参数量。
- SPPF: Spatial Pyramid Pooling Fast模块，快速处理多尺度特征，提高对不同大小目标的检测能力。

#### Head
- nn.Upsample: 上采样层，使用最近邻插值放大特征图。
- Concat: 特征图连接层，沿通道维度合并多个特征图。
- C2f: 同Backbone中的C2f，用于进一步特征融合。
- Conv: 卷积层，用于下采样或特征调整。
- Detect: 检测头，输出边界框、类别和置信度。

---
## v9
```yaml
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# YOLOv9c object detection model with P3/8 - P5/32 outputs
# Model docs: https://docs.ultralytics.com/models/yolov9
# Task docs: https://docs.ultralytics.com/tasks/detect
# 618 layers, 25590912 parameters, 104.0 GFLOPs

# Parameters
nc: 80 # number of classes

# GELAN backbone
backbone:
  - [-1, 1, Conv, [64, 3, 2]] # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4
  - [-1, 1, RepNCSPELAN4, [256, 128, 64, 1]] # 2
  - [-1, 1, ADown, [256]] # 3-P3/8
  - [-1, 1, RepNCSPELAN4, [512, 256, 128, 1]] # 4
  - [-1, 1, ADown, [512]] # 5-P4/16
  - [-1, 1, RepNCSPELAN4, [512, 512, 256, 1]] # 6
  - [-1, 1, ADown, [512]] # 7-P5/32
  - [-1, 1, RepNCSPELAN4, [512, 512, 256, 1]] # 8
  - [-1, 1, SPPELAN, [512, 256]] # 9

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]] # cat backbone P4
  - [-1, 1, RepNCSPELAN4, [512, 512, 256, 1]] # 12

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]] # cat backbone P3
  - [-1, 1, RepNCSPELAN4, [256, 256, 128, 1]] # 15 (P3/8-small)

  - [-1, 1, ADown, [256]]
  - [[-1, 12], 1, Concat, [1]] # cat head P4
  - [-1, 1, RepNCSPELAN4, [512, 512, 256, 1]] # 18 (P4/16-medium)

  - [-1, 1, ADown, [512]]
  - [[-1, 9], 1, Concat, [1]] # cat head P5
  - [-1, 1, RepNCSPELAN4, [512, 512, 256, 1]] # 21 (P5/32-large)

  - [[15, 18, 21], 1, Detect, [nc]] # Detect(P3, P4, P5)

```
![v9](D:\Code\ob\code\pict\v9.jpg)
### YOLOv9 介绍
#### Backbone
- Conv: 标准卷积层，用于特征提取。
- RepNCSPELAN4: 可重复的NCSP ELAN模块，增强特征表示。
- ADown: Attention Downsampling模块，高效下采样。
- SPPELAN: Spatial Pyramid Pooling ELAN模块，融合多尺度特征。

#### Head
- nn.Upsample: 上采样层。
- Concat: 连接层。
- RepNCSPELAN4: 同Backbone。
- ADown: 同Backbone。
- Detect: 检测头。

---
## v10
```yaml
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# YOLOv10n object detection model with P3/8 - P5/32 outputs
# Model docs: https://docs.ultralytics.com/models/yolov10
# Task docs: https://docs.ultralytics.com/tasks/detect

# Parameters
nc: 80 # number of classes
scales: # model compound scaling constants, i.e. 'model=yolov10n.yaml' will call yolov10.yaml with scale 'n'
  # [depth, width, max_channels]
  n: [0.33, 0.25, 1024]

backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]] # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4
  - [-1, 3, C2f, [128, True]]
  - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8
  - [-1, 6, C2f, [256, True]]
  - [-1, 1, SCDown, [512, 3, 2]] # 5-P4/16
  - [-1, 6, C2f, [512, True]]
  - [-1, 1, SCDown, [1024, 3, 2]] # 7-P5/32
  - [-1, 3, C2f, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]] # 9
  - [-1, 1, PSA, [1024]] # 10

# YOLOv10.0n head
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]] # cat backbone P4
  - [-1, 3, C2f, [512]] # 13

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]] # cat backbone P3
  - [-1, 3, C2f, [256]] # 16 (P3/8-small)

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 13], 1, Concat, [1]] # cat head P4
  - [-1, 3, C2f, [512]] # 19 (P4/16-medium)

  - [-1, 1, SCDown, [512, 3, 2]]
  - [[-1, 10], 1, Concat, [1]] # cat head P5
  - [-1, 3, C2fCIB, [1024, True, True]] # 22 (P5/32-large)

  - [[16, 19, 22], 1, v10Detect, [nc]] # Detect(P3, P4, P5)

```
![v10](D:\Code\ob\code\pict\v10.jpg)
### YOLOv10 介绍
#### Backbone
- Conv: 卷积层。
- C2f: C2f模块。
- SCDown: Stride Convolution Downsampling。
- SPPF: SPPF模块。
- PSA: Partial Self-Attention模块，提高注意力机制效率。

#### Head
- nn.Upsample: 上采样。
- Concat: 连接。
- C2f: C2f模块。
- Conv: 卷积。
- C2fCIB: C2f with Channel Independent Branch。
- v10Detect: YOLOv10检测头。

---
## v11
```yaml
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# Ultralytics YOLO11 object detection model with P3/8 - P5/32 outputs
# Model docs: https://docs.ultralytics.com/models/yolo11
# Task docs: https://docs.ultralytics.com/tasks/detect

# Parameters
nc: 80 # number of classes
scales: # model compound scaling constants, i.e. 'model=yolo11n.yaml' will call yolo11.yaml with scale 'n'
  # [depth, width, max_channels]
  n: [0.50, 0.25, 1024] # summary: 319 layers, 2624080 parameters, 2624064 gradients, 6.6 GFLOPs
  s: [0.50, 0.50, 1024] # summary: 319 layers, 9458752 parameters, 9458736 gradients, 21.7 GFLOPs
  m: [0.50, 1.00, 512] # summary: 409 layers, 20114688 parameters, 20114672 gradients, 68.5 GFLOPs
  l: [1.00, 1.00, 512] # summary: 631 layers, 25372160 parameters, 25372144 gradients, 87.6 GFLOPs
  x: [1.00, 1.50, 512] # summary: 631 layers, 56966176 parameters, 56966160 gradients, 196.0 GFLOPs

# YOLO11n backbone
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]] # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8
  - [-1, 2, C3k2, [512, False, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]] # 5-P4/16
  - [-1, 2, C3k2, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]] # 7-P5/32
  - [-1, 2, C3k2, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]] # 9
  - [-1, 2, C2PSA, [1024]] # 10

# YOLO11n head
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]] # cat backbone P4
  - [-1, 2, C3k2, [512, False]] # 13

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]] # cat backbone P3
  - [-1, 2, C3k2, [256, False]] # 16 (P3/8-small)

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 13], 1, Concat, [1]] # cat head P4
  - [-1, 2, C3k2, [512, False]] # 19 (P4/16-medium)

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 10], 1, Concat, [1]] # cat head P5
  - [-1, 2, C3k2, [1024, True]] # 22 (P5/32-large)

  - [[16, 19, 22], 1, Detect, [nc]] # Detect(P3, P4, P5)

```
![v11](D:\Code\ob\code\pict\v11.png)
### YOLOv11介绍
#### Backbone
- Conv: 卷积层。
- C3k2: C3模块变体，使用k=2。
- SPPF: SPPF模块。
- C2PSA: C2 with Partial Self-Attention。

#### Head
- nn.Upsample: 上采样。
- Concat: 连接。
- C3k2: C3k2模块。
- Conv: 卷积。
- Detect: 检测头。

---
## v12
```yaml
# YOLOv12 🚀, AGPL-3.0 license
# YOLOv12 object detection model with P3-P5 outputs. For Usage examples see https://docs.ultralytics.com/tasks/detect
# CFG file for YOLOv12-turbo

# Parameters
nc: 80 # number of classes
scales: # model compound scaling constants, i.e. 'model=yolov12n.yaml' will call yolov12.yaml with scale 'n'
  # [depth, width, max_channels]
  n: [0.50, 0.25, 1024] # summary: 497 layers, 2,553,904 parameters, 2,553,888 gradients, 6.2 GFLOPs
  s: [0.50, 0.50, 1024] # summary: 497 layers, 9,127,424 parameters, 9,127,408 gradients, 19.7 GFLOPs
  m: [0.50, 1.00, 512] # summary: 533 layers, 19,670,784 parameters, 19,670,768 gradients, 60.4 GFLOPs
  l: [1.00, 1.00, 512] # summary: 895 layers, 26,506,496 parameters, 26,506,480 gradients, 83.3 GFLOPs
  x: [1.00, 1.50, 512] # summary: 895 layers, 59,414,176 parameters, 59,414,160 gradients, 185.9 GFLOPs


# YOLO12-turbo backbone
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv,  [64, 3, 2]] # 0-P1/2
  - [-1, 1, Conv,  [128, 3, 2, 1, 2]] # 1-P2/4
  - [-1, 2, C3k2,  [256, False, 0.25]]
  - [-1, 1, Conv,  [256, 3, 2, 1, 4]] # 3-P3/8
  - [-1, 2, C3k2,  [512, False, 0.25]]
  - [-1, 1, Conv,  [512, 3, 2]] # 5-P4/16
  - [-1, 4, A2C2f, [512, True, 4]]
  - [-1, 1, Conv,  [1024, 3, 2]] # 7-P5/32
  - [-1, 4, A2C2f, [1024, True, 1]] # 8

# YOLO12-turbo head
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]] # cat backbone P4
  - [-1, 2, A2C2f, [512, False, -1]] # 11

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]] # cat backbone P3
  - [-1, 2, A2C2f, [256, False, -1]] # 14

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 11], 1, Concat, [1]] # cat head P4
  - [-1, 2, A2C2f, [512, False, -1]] # 17

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 8], 1, Concat, [1]] # cat head P5
  - [-1, 2, C3k2, [1024, True]] # 20 (P5/32-large)

  - [[14, 17, 20], 1, Detect, [nc]] # Detect(P3, P4, P5)

```
![v12](D:\Code\ob\code\pict\v12.png)
### YOLOv12 介绍
#### Backbone
- Conv: 卷积层。
- C3k2: C3k2模块。
- A2C2f: Attention to C2f模块。

#### Head
- nn.Upsample: 上采样。
- Concat: 连接。
- A2C2f: A2C2f模块。
- Conv: 卷积。
- C3k2: C3k2模块。
- Detect: 检测头。

---
## v13
```yaml
nc: 80 # number of classes
scales: # model compound scaling constants, i.e. 'model=yolov13n.yaml' will call yolov13.yaml with scale 'n'
  # [depth, width, max_channels]
  n: [0.50, 0.25, 1024]   # Nano
  s: [0.50, 0.50, 1024]   # Small
  l: [1.00, 1.00, 512]    # Large
  x: [1.00, 1.50, 512]    # Extra Large

backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv,  [64, 3, 2]] # 0-P1/2
  - [-1, 1, Conv,  [128, 3, 2, 1, 2]] # 1-P2/4
  - [-1, 2, DSC3k2,  [256, False, 0.25]]
  - [-1, 1, Conv,  [256, 3, 2, 1, 4]] # 3-P3/8
  - [-1, 2, DSC3k2,  [512, False, 0.25]]
  - [-1, 1, DSConv,  [512, 3, 2]] # 5-P4/16
  - [-1, 4, A2C2f, [512, True, 4]]
  - [-1, 1, DSConv,  [1024, 3, 2]] # 7-P5/32
  - [-1, 4, A2C2f, [1024, True, 1]] # 8

head:
  - [[4, 6, 8], 2, HyperACE, [512, 8, True, True, 0.5, 1, "both"]]
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [ 9, 1, DownsampleConv, []]
  - [[6, 9], 1, FullPAD_Tunnel, []]  #12     
  - [[4, 10], 1, FullPAD_Tunnel, []]  #13    
  - [[8, 11], 1, FullPAD_Tunnel, []] #14 
  
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 12], 1, Concat, [1]] # cat backbone P4
  - [-1, 2, DSC3k2, [512, True]] # 17
  - [[-1, 9], 1, FullPAD_Tunnel, []]  #18

  - [17, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 13], 1, Concat, [1]] # cat backbone P3
  - [-1, 2, DSC3k2, [256, True]] # 21
  - [10, 1, Conv, [256, 1, 1]]
  - [[21, 22], 1, FullPAD_Tunnel, []]  #23
  
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 18], 1, Concat, [1]] # cat head P4
  - [-1, 2, DSC3k2, [512, True]] # 26
  - [[-1, 9], 1, FullPAD_Tunnel, []]  

  - [26, 1, Conv, [512, 3, 2]]
  - [[-1, 14], 1, Concat, [1]] # cat head P5
  - [-1, 2, DSC3k2, [1024,True]] # 30 (P5/32-large)
  - [[-1, 11], 1, FullPAD_Tunnel, []]  
  
  - [[23, 27, 31], 1, Detect, [nc]] # Detect(P3, P4, P5)

```
![v13-1](D:\Code\ob\code\pict\v13-1.png)
### YOLOv13 介绍
#### Backbone
- Conv: 卷积层。
- DSC3k2: Depthwise Separable C3k2模块。
- DSConv: Depthwise Separable Conv。
- A2C2f: A2C2f模块。

#### Head
- HyperACE: 超参数自适应通道增强。
- nn.Upsample: 上采样。
- DownsampleConv: 下采样卷积。
- FullPAD_Tunnel: 全填充隧道模块。
- Concat: 连接。
- DSC3k2: DSC3k2模块。
- Conv: 卷积。
- Detect: 检测头。

# 对比

## v2与v1的区别
- **Backbone**: v2使用Darknet-19，v1使用自定义网络。Darknet-19更高效。
- **Head**: v2引入anchors和Passthrough层融合特征。
- **创新点**: Batch Normalization、多尺度训练、anchors机制，提高精度和速度。

## v3与v2的区别
- **Backbone**: v3使用Darknet-53，更深层。
- **Head**: 多尺度预测、FPN-like结构。
- **创新点**: 多尺度检测、残差连接，提高小物体检测。

## v4与v3的区别
- **Backbone**: v4使用CSPDarknet53，引入CSP连接。
- **Head**: SPP + PANet颈部。
- **创新点**: Mosaic数据增强、Mish激活、CIOU损失。

## v5与v4的区别
- **Backbone**: v5使用CSP-like结构，Focus层。
- **Head**: 类似v4，但优化C3模块。
- [**创新点**: AutoAnchor、PyTorch实现、超参数进化。](https://blog.csdn.net/qq_44878985/article/details/129287587)

## v6与v5的区别
- **Backbone**: v6优化为更高效卷积。
- **Head**: 使用转置卷积上采样。
- **创新点**: 高效层聚合、RepVGG训练。

## v7与v6的区别
- **Backbone**: v7引入ELAN模块。
- **Head**: 辅助头训练。
- **创新点**: Trainable bag-of-freebies、RepConv。

## v8与v7的区别
- **Backbone**: v8使用C2f模块。
- **Head**: Anchor-free检测。
- [**创新点**: Ultralytics框架、任务对齐学习。](https://www.ultralytics.com/blog/comparing-ultralytics-yolo11-vs-previous-yolo-models)

## v9与v8的区别
- **Backbone**: v9引入GELAN和PGI。
- **Head**: 类似v8，但优化聚合。
- **创新点**: 可编程梯度信息、深度监督。

## v10与v9的区别
- **Backbone**: v10增强CSPNet。
- **Head**: 双标签分配。
- **创新点**: 一致双分配、排名损失。

## v11与v10的区别
- **Backbone**: v11使用C3k2和C2PSA。
- **Head**: 优化上采样和连接。
- [**创新点**: 多任务支持、实时性能提升。](https://www.ultralytics.com/blog/comparing-ultralytics-yolo11-vs-previous-yolo-models)

## v12与v11的区别
- **Backbone**: v12引入A2C2f模块。
- **Head**: 类似，但优化C3k2。
- **创新点**: 涡轮变体、注意力机制。

## v13与v12的区别
- **Backbone**: v13使用DSC3k2和DSConv。
- **Head**: 引入HyperACE和FullPAD_Tunnel。
- **创新点**: 超参数自适应通道增强、全填充隧道，提高精度。
# 核心维度对比

1) 骨干（Backbone）与颈部（Neck）
- v1: 自定义 CNN（受 GoogLeNet 启发）+ 全连接预测，未成型的 Neck。
- v2: Darknet-19，passthrough 细粒度特征；Neck 原型化。
- v3: Darknet-53（残差）+ FPN 多尺度融合。
- v4: CSPDarknet53 + SPP + PAN（成熟三件套），Neck 成型（SPP/PAN）。
- v5: CSP-Darknet 变体 + Focus/SPPF，Neck 延续 PAN/FPN 思想。
- v6: EfficientRep（RepVGG 思想）+ Rep-PAN（为部署/量化友好）。
- v7: E-ELAN 主干，强化梯度流与特征重用；PAN/FPN 融合强化。
- v8: C2f 主干，轻量高效；PAN/FPN 优化。
- v9: GELAN 主干，强调可编程梯度引导下的高效表达。
- v10: 面向 E2E 的高效主干与 Neck，兼顾表达与稀疏输出。
- v11: 在 C2f/CSP 系上进一步细化与统一多任务设计。
- v12: 注意力轻量化模块融入 Backbone/Neck 流程。
- v13: 在检测头前/后加入轻量超图关系建模组件。

2) Anchor 范式
- v1: 无 Anchor（网格直回归）。
- v2: 首次系统性使用 Anchor，K-means 聚类。
- v3: Anchor-based+多尺度（9 anchors/3 scales）。
- v4: Anchor-based 为主，三尺度检测成熟化。
- v5: Anchor-based 主流（自动锚框/适配工具）。
- v6: Anchor-free 为核心设计，配合解耦头。
- v7: 同时兼容多范式，工程灵活度高。
- v8: 默认 Anchor-free（解耦头）。
- v9: 以 PGI 为核心，可兼容主流头部范式（实现侧多样）。
- v10: 针对 E2E 优化的输出与匹配（是否 Anchor 与实现相关）。
- v11/12/13: 更偏 Anchor-free 和轻量结构，强调可部署与鲁棒性。

3) 多尺度与特征融合
- v1: 7×7 网格单尺度，细粒度不足。
- v2: 多尺度训练+passthrough 细粒度特征。
- v3: FPN 三尺度检测，显著改善小目标。
- v4: SPP+PAN 强化跨层与跨尺度融合。
- v5/6/7/8: 延续并加强 FPN/PAN 融合策略（搭配不同主干/模块）。
- v9: 在 PGI 引导下的多尺度头优化。
- v10+: 进一步面向 E2E/注意力/关系建模的多尺度协同。

4) 损失函数与标签分配
- v1: 多项和损失（坐标/置信度/分类），非 IoU 族。
- v2/3: logistic/BCE 与坐标回归，Anchor 相关匹配。
- v4: 引入 CIoU/Focal 等，BoF/BoS 系统化。
- v5: CIoU + DFL 等工程优选组合，自动类权/标签平滑。
- v6: Varifocal + DFL，灰边框标签分配，自蒸馏。
- v7: EIoU/CIoU 等，动态标签分配与辅助头深监督。
- v8: Anchor-free 的任务对齐/质量分配（工程化变体），IoU 族 + DFL。
- v9: 在 PGI 引导下的损失/匹配细化。
- v10: E2E 排序损失 + 匹配（匈牙利/变体），减少对 NMS 的依赖。
- v11/12/13: 在 Anchor-free/注意力/关系建模下的持续打磨。

5) 训练与数据增强
- v1: 经典增强（缩放/裁剪/抖动）。
- v2: BN、分辨率提升、多尺度训练。
- v3: 继续沿用多尺度与基本增强。
- v4: Mosaic、MixUp、SAT、DropBlock、Label Smoothing 等系统化组合。
- v5: AutoAnchor、AMP、EMA、AutoAug、超参进化、早停、完善可视化。
- v6: 自蒸馏、灰边框分配、量化友好训练。
- v7: 可训练 BoF、辅助头、动态标签与增量训练。
- v8: 增强强度调度、Anchor-free 友好流程。
- v9: PGI 驱动的训练收敛质量提升。
- v10+: 排序/匹配目标与端到端优化相配套；注意力/图正则逐步引入（v12/v13）。

6) NMS 与端到端（E2E）
- v1–v9: 依赖 NMS（或其变体）进行后处理。
- v10: 端到端目标，显式弱化/消除 NMS；输出更稀疏、排序更可学。
- v11–v13: 延续 E2E 友好与工程简化方向。

