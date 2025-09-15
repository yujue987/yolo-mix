
## autopad(函数)

###  函数代码

```python
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
```

###  参数介绍

#### `k` (kernel size, 卷积核大小)
- **类型**：`int` 或 `list[int]`
- **作用**：表示卷积核（filter）的尺寸。例如：
    - `k=3` → 卷积核大小是 `3×3`
    - `k=[3,5]` → 卷积核大小分别在 **高度方向 3**，**宽度方向 5**
- **注意点**：在卷积计算时，卷积核的大小会影响到感受野和最终输出的空间分辨率。

#### `p` (padding, 填充大小)
- **类型**：`int` 或 `list[int]`，默认为 `None`
- **作用**：决定输入张量在卷积操作前 **四周补零的数量**，影响输出张量的大小。
- **常见设置**：
    - `p=0` → 不填充，输出尺寸比输入小
    - `p=k//2` → 常见的 **"same" 卷积**，保证输入和输出空间尺寸一致
- **在函数中**：如果 `p` 没有手动指定（`p=None`），就会自动计算一个合适的填充值。

#### `d` (dilation, 空洞卷积膨胀率)
- **类型**：`int`，默认为 `1`
- **作用**：
    - `d=1` → 标准卷积，卷积核按正常方式滑动
    - `d>1` → 空洞卷积（dilated convolution），在卷积核元素之间插入空隙，使感受野变大但参数量不变
- **影响**：  
    空洞卷积实际上的卷积核尺寸会被“扩展”：
    keffective=d×(k−1)+1k_{effective} = d \times (k - 1) + 1
    例如：`k=3, d=2` → 实际卷积核感受野为 `5`

###  函数功能解析

#### 处理 dilation 对 kernel 的影响

```python
if d > 1:
    k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
```

- 当 `d > 1` 时，计算 **实际的卷积核大小**（考虑空洞率）  
    例如：
    - `k=3, d=2` → 实际卷积核大小 = `2 * (3-1) + 1 = 5`
    - `k=[3,5], d=2` → 实际大小 = `[5, 9]`

#### 自动计算 padding

```python
if p is None:
    p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
```

- 如果 `p` 没有指定，函数会自动选择一个值使得卷积输出和输入保持相同空间大小（**"same" padding**）
- 计算方式：
    - 当 `k` 是奇数时，`p = k // 2`
        - `k=3` → `p=1`
        - `k=5` → `p=2`
    - 如果 `k` 是偶数，结果就会稍微不对称（因为 `//` 是整除）

#### 返回结果

```python
return p
```

- 返回最终的 padding 值（整数或列表），供卷积层使用。

###  作用

这个函数的主要作用是：  
**根据卷积核大小 `k` 和膨胀率 `d`，自动计算合适的 padding 值 `p`，确保卷积的输出空间尺寸与输入一致（same convolution）。**

## Conv(类)
 [关于卷积基础的介绍](https://blog.csdn.net/palet/article/details/88862647)
### 类代码

```python
class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Apply convolution and activation without batch normalization."""
        return self.act(self.conv(x))
```

---

### 参数介绍

#### `c1` (ch_in, 输入通道数)
- **类型**：`int`
- **作用**：输入特征图的通道数。例如图像是 RGB → `c1=3`。

#### `c2` (ch_out, 输出通道数)
- **类型**：`int`
- **作用**：卷积核的个数，也就是输出特征图的通道数。
- **影响**：控制卷积层的表达能力和输出维度。

#### `k` (kernel size, 卷积核大小)
- **类型**：`int` 或 `list[int]`，默认 `1`
- **作用**：卷积核的宽和高大小，例如：
    - `k=3` → `3×3` 卷积
    - `k=1` → `1×1` 卷积（常用于降维或升维）

#### `s` (stride, 步幅)
- **类型**：`int`，默认 `1`
- **作用**：卷积核每次滑动的步长。
    - `s=1` → 输出尺寸与输入接近（受 padding 影响）
    - `s=2` → 输出尺寸约为输入的一半（下采样）

#### `p` (padding, 填充)
- **类型**：`int` 或 `list[int]`，默认 `None`
- **作用**：控制卷积时输入边界的补零数量。
- **在函数中**：如果不手动指定，会调用 `autopad(k, p, d)` 自动计算 padding，确保 **"same" 卷积**。

#### `g` (groups, 分组卷积)
- **类型**：`int`，默认 `1`
- **作用**：分组卷积的分组数。
    - `g=1` → 普通卷积
    - `g=c1` → 深度可分离卷积（Depthwise Convolution）

#### `d` (dilation, 空洞卷积膨胀率)
- **类型**：`int`，默认 `1`
- **作用**：控制卷积核的膨胀。`d>1` 时，卷积核内部元素之间会插空，使感受野扩大。

#### `act` (activation, 激活函数)
- **类型**：`bool | nn.Module`，默认 `True`
- **作用**：指定卷积后的非线性激活函数。
    - `True` → 使用默认激活函数 `nn.SiLU()`
    - `nn.Module` → 用户传入自定义激活函数（如 `nn.ReLU()`）
    - `False` 或 `None` → 不使用激活函数（替换为 `nn.Identity()`）

### 成员属性

#### `self.conv`

```python
self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
```
- **作用**：定义一个 2D 卷积层。
- **特点**：`bias=False` → 因为后面有 `BatchNorm`，不需要卷积层偏置。

#### `self.bn`

```python
self.bn = nn.BatchNorm2d(c2)
```
- **作用**：对卷积输出进行批量归一化，加速收敛并稳定训练。

#### `self.act`

```python
self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
```
- **作用**：定义激活函数。
- **逻辑**：
    - `act=True` → 使用默认 `SiLU`
    - `act=nn.Module` → 使用传入的激活函数
    - 其他情况 → 不使用激活函数（恒等映射）

### 成员方法

#### `forward(self, x)`

```python
return self.act(self.bn(self.conv(x)))
```
- **作用**：前向传播。
- **步骤**：
    1. 输入 `x` 经过卷积 `self.conv`
    2. 经过批量归一化 `self.bn`
    3. 经过激活函数 `self.act`        
- **用途**：标准的卷积块（Conv → BN → Act）。

#### `forward_fuse(self, x)`

```python
return self.act(self.conv(x))
```
- **作用**：融合推理模式，去掉 `BatchNorm`。
- **步骤**：
    1. 输入 `x` 经过卷积 `self.conv`
    2. 直接通过激活函数 `self.act`
- **用途**：在模型推理部署时，`Conv` 和 `BatchNorm` 可以融合成一个卷积层，从而提高推理速度。

### 作用

这个类的主要作用是：  
**实现一个常见的卷积模块（Conv → BN → Act），同时支持自动 padding、分组卷积、空洞卷积，以及激活函数的灵活选择。**  
在 YOLO 系列网络中，这是最基本的构建块，几乎所有层都基于它实现。

## Conv2(类)

### 类代码
```python
class Conv2(Conv):
    """Simplified RepConv module with Conv fusing."""

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)  # add 1x1 conv

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def forward_fuse(self, x):
        """Apply fused convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def fuse_convs(self):
        """Fuse parallel convolutions."""
        w = torch.zeros_like(self.conv.weight.data)
        i = [x // 2 for x in w.shape[2:]]
        w[:, :, i[0] : i[0] + 1, i[1] : i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__("cv2")
        self.forward = self.forward_fuse
```

### 参数介绍

和 `Conv` 类的参数一致，含义也相同：
- **`c1`**：输入通道数
- **`c2`**：输出通道数
- **`k`**：卷积核大小，默认 `3`
- **`s`**：步幅（stride），默认 `1`
- **`p`**：padding，若为 `None`，则使用 `autopad` 自动计算
- **`g`**：分组卷积参数，默认 `1`
- **`d`**：空洞卷积膨胀率，默认 `1`    
- **`act`**：激活函数，默认使用 `nn.SiLU()`

### 成员属性

#### `self.conv`

继承自 `Conv`，是主卷积层（一般是 `k×k` 卷积，比如 `3×3`）。

#### `self.bn`

继承自 `Conv`，对主卷积输出进行 Batch Normalization。

#### `self.act`

继承自 `Conv`，用于非线性激活（默认 `SiLU`）。

#### `self.cv2`

```python
self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)
```
- 新增的 **辅助卷积**：`1×1` 卷积
- **作用**：与主卷积并联，增强特征表达能力。
- **特点**：不带 bias，因为 BN 层会处理偏置项。

### 成员方法

#### `forward(self, x)`

```python
return self.act(self.bn(self.conv(x) + self.cv2(x)))
```
- **作用**：标准训练时的前向传播。
- **步骤**：
    1. 输入 `x` 分别经过 `k×k` 卷积（`self.conv`）和 `1×1` 卷积（`self.cv2`）
    2. 两者结果相加
    3. 经过 BatchNorm
    4. 经过激活函数
- **等价结构**：
    ```
    (Conv(k×k) + Conv(1×1)) → BN → Act
    ```

#### `forward_fuse(self, x)`

```python
return self.act(self.bn(self.conv(x)))
```
- **作用**：推理模式的前向传播。
- **区别**：此时 `cv2` 已经融合进 `conv` 的权重，不再单独计算 `1×1` 卷积。

#### `fuse_convs(self)`

```python
w = torch.zeros_like(self.conv.weight.data)
i = [x // 2 for x in w.shape[2:]]
w[:, :, i[0] : i[0] + 1, i[1] : i[1] + 1] = self.cv2.weight.data.clone()
self.conv.weight.data += w
self.__delattr__("cv2")
self.forward = self.forward_fuse
```
- **作用**：把 `1×1` 卷积 `cv2` 的权重融合进 `conv`，实现 **RepConv 重参数化**。
- **步骤**：
    1. 创建一个与 `self.conv.weight` 形状相同的零张量 `w`
    2. 找到 `conv` 卷积核中心位置索引 `i`
    3. 将 `cv2.weight` 的参数放到 `w` 的中心（等价于把 `1×1` 卷积嵌入 `k×k` 卷积核中心）
    4. 把 `w` 加到 `conv.weight`，即权重融合
    5. 删除 `cv2` 属性（`self.__delattr__("cv2")`）
    6. 修改 `forward` 为 `forward_fuse`，推理时只走单一卷积

### 作用

这个类的主要作用是：  
**实现一个简化的 RepConv 模块**
- 训练阶段：并联 `k×k` 和 `1×1` 卷积，提升表达能力。
- 推理阶段：通过 `fuse_convs()` 将两者融合为单一卷积层，降低计算量，加速推理。

###  总结

- **训练时结构**：
    ```
    [Conv(k×k) + Conv(1×1)] → BN → Act
    ```
- **推理时结构**：
    ```
    Conv(fused) → BN → Act
    ```
- **优点**：
    - 训练更强大 → 组合卷积增强特征学习能力   
    - 推理更高效 → 单一卷积加速推理

## DSConv(类)

### 类代码
```python
class DSConv(nn.Module):
    """The Basic Depthwise Separable Convolution."""
    def __init__(self, c_in, c_out, k=3, s=1, p=None, d=1, bias=False):
        super().__init__()
        if p is None:
            p = (d * (k - 1)) // 2
        self.dw = nn.Conv2d(
            c_in, c_in, kernel_size=k, stride=s,
            padding=p, dilation=d, groups=c_in, bias=bias
        )
        self.pw = nn.Conv2d(c_in, c_out, 1, 1, 0, bias=bias)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        return self.act(self.bn(x))
```

### 参数介绍

#### `c_in` (输入通道数)
- **类型**：`int`
- **作用**：输入特征图的通道数。

#### `c_out` (输出通道数)
- **类型**：`int`
- **作用**：最终输出特征图的通道数。

#### `k` (kernel size, 卷积核大小)
- **类型**：`int`，默认 `3`
- **作用**：深度卷积（depthwise convolution）的卷积核大小。

#### `s` (stride, 步幅)
- **类型**：`int`，默认 `1`
- **作用**：深度卷积的步幅。

#### `p` (padding, 填充)
- **类型**：`int`，默认 `None`
- **作用**：深度卷积的填充大小。如果为 `None`，则会自动计算以保持空间分辨率。

#### `d` (dilation, 空洞卷积膨胀率)
- **类型**：`int`，默认 `1`
- **作用**：深度卷积的膨胀率。

#### `bias` (偏置)
- **类型**：`bool`，默认 `False`
- **作用**：是否在卷积层中使用偏置项。由于后面有 `BatchNorm`，通常设为 `False`。

### 成员属性

#### `self.dw`
- **类型**: `nn.Conv2d`
- **作用**: 深度卷积（Depthwise Convolution）。对输入的每个通道独立进行卷积操作。`groups=c_in` 是其关键特征。

#### `self.pw`
- **类型**: `nn.Conv2d`
- **作用**: 点卷积（Pointwise Convolution）。使用 `1x1` 卷积核，用于组合深度卷积的输出，改变通道数。

#### `self.bn`
- **类型**: `nn.BatchNorm2d`
- **作用**: 对点卷积的输出进行批量归一化。

#### `self.act`
- **类型**: `nn.SiLU`
- **作用**: 激活函数，增加非线性。

### 成员方法

#### `forward(self, x)`
- **作用**: 前向传播。
- **步骤**:
    1. 输入 `x` 经过深度卷积 `self.dw`
    2. 结果经过点卷积 `self.pw`
    3. 结果经过批量归一化 `self.bn`
    4. 最后通过激活函数 `self.act`

### 作用

**实现一个深度可分离卷积（Depthwise Separable Convolution）模块。**
这种卷积将标准卷积分解为深度卷积和点卷积两个步骤，可以在大幅减少计算量和参数量的同时，保持相当的性能。这是移动端和轻量级网络设计的核心模块之一。

## LightConv(类)

### 类代码
```python
class LightConv(nn.Module):
    """
    Light convolution with args(ch_in, ch_out, kernel).

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        """Apply 2 convolutions to input tensor."""
        return self.conv2(self.conv1(x))
```

### 参数介绍

#### `c1` (输入通道数)
- **类型**：`int`
- **作用**：输入特征图的通道数。

#### `c2` (输出通道数)
- **类型**：`int`
- **作用**：输出特征图的通道数。

#### `k` (kernel size, 卷积核大小)
- **类型**：`int`，默认 `1`
- **作用**：深度卷积（`DWConv`）的卷积核大小。

#### `act` (activation, 激活函数)
- **类型**：`nn.Module`，默认 `nn.ReLU()`
- **作用**：深度卷积后的激活函数。

### 成员属性

#### `self.conv1`
- **类型**: `Conv`
- **作用**: 一个 `1x1` 的标准卷积，用于升维或降维，并且不使用激活函数 (`act=False`)。

#### `self.conv2`
- **类型**: `DWConv`
- **作用**: 一个深度卷积，对 `self.conv1` 的输出进行卷积操作，并使用指定的激活函数。

### 成员方法

#### `forward(self, x)`
- **作用**: 前向传播。
- **步骤**:
    1. 输入 `x` 首先经过 `1x1` 的标准卷积 `self.conv1`。
    2. 然后结果经过深度卷积 `self.conv2`。

### 作用

**实现一个轻量级的卷积模块。**

它通过结合一个 `1x1` 的标准卷积和一个深度卷积，实现了高效的特征提取。这种结构在一些轻量级网络（如 HGNetV2）中被使用，以平衡计算成本和模型性能。

## DWConv(类)

### 类代码
```python
class DWConv(Conv):
    """Depth-wise convolution."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        """Initialize Depth-wise convolution with given parameters."""
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)
```

### 参数介绍

`DWConv` 继承自 `Conv`，其参数与 `Conv` 基本一致，但有一个关键区别：

- **`g` (groups, 分组卷积)**: 在 `DWConv` 中，这个参数被固定设置为 `math.gcd(c1, c2)`，即输入通道数 `c1` 和输出通道数 `c2` 的最大公约数。在典型的深度卷积应用中，`c1` 通常等于 `c2`，此时 `g=c1=c2`，意味着每个输入通道都由自己独立的卷积核进行处理。

其他参数 `c1`, `c2`, `k`, `s`, `d`, `act` 的含义与 `Conv` 类完全相同。

### 成员属性和方法

`DWConv` 直接复用了 `Conv` 类的所有成员属性（`self.conv`, `self.bn`, `self.act`）和成员方法（`forward`, `forward_fuse`）。

### 作用

**实现一个深度卷积（Depthwise Convolution）模块。**

与标准卷积不同，深度卷积对输入的每个通道独立进行卷积操作，从而大大减少了参数量和计算量。它通常与一个 `1x1` 的点卷积（Pointwise Convolution）结合使用，构成深度可分离卷积（Depthwise Separable Convolution），在保持模型性能的同时，极大地提升了计算效率。

## DWConvTranspose2d(类)

### 类代码
```python
class DWConvTranspose2d(nn.ConvTranspose2d):
    """Depth-wise transpose convolution."""

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):  # ch_in, ch_out, kernel, stride, padding, padding_out
        """Initialize DWConvTranspose2d class with given parameters."""
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))
```

### 参数介绍

`DWConvTranspose2d` 继承自 `nn.ConvTranspose2d`，其参数与标准转置卷积基本一致，但关键区别在于 `groups` 参数的设定。

- **`c1` (ch_in, 输入通道数)**
- **`c2` (ch_out, 输出通道数)**
- **`k` (kernel size, 卷积核大小)**
- **`s` (stride, 步幅)**
- **`p1` (padding, 填充)**
- **`p2` (output_padding, 输出填充)**
- **`groups`**: 与 `DWConv` 类似，这个参数被固定设置为 `math.gcd(c1, c2)`，以实现深度可分离的转置卷积。

### 作用

**实现一个深度可分离的转置卷积（Depthwise Transpose Convolution）模块。**

转置卷积通常用于上采样操作，例如在分割任务或生成模型中恢复特征图的分辨率。`DWConvTranspose2d` 在执行此操作时，采用了深度可分离的机制，从而比标准的转置卷积更高效。

## ConvTranspose(类)

### 类代码
```python
class ConvTranspose(nn.Module):
    """Convolution transpose 2d layer."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """Initialize ConvTranspose2d layer with batch normalization and activation function."""
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Applies transposed convolutions, batch normalization and activation to input."""
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):
        """Applies activation and convolution transpose operation to input."""
        return self.act(self.conv_transpose(x))
```

### 参数介绍

#### `c1` (ch_in, 输入通道数)
- **类型**：`int`
- **作用**：输入特征图的通道数。

#### `c2` (ch_out, 输出通道数)
- **类型**：`int`
- **作用**：输出特征图的通道数。

#### `k` (kernel size, 卷积核大小)
- **类型**：`int`，默认 `2`
- **作用**：转置卷积的卷积核大小。

#### `s` (stride, 步幅)
- **类型**：`int`，默认 `2`
- **作用**：转置卷积的步幅。通常 `s>1` 用于上采样。

#### `p` (padding, 填充)
- **类型**：`int`，默认 `0`
- **作用**：转置卷积的填充大小。

#### `bn` (Batch Normalization)
- **类型**：`bool`，默认 `True`
- **作用**：是否使用批量归一化。

#### `act` (activation, 激活函数)
- **类型**：`bool | nn.Module`，默认 `True`
- **作用**：是否使用激活函数。
    - `True` → 使用默认的 `nn.SiLU()`
    - `nn.Module` → 使用指定的激活函数
    - `False` → 不使用激活函数

### 成员属性

#### `self.conv_transpose`
- **类型**: `nn.ConvTranspose2d`
- **作用**: 核心的转置卷积层，用于上采样。`bias` 设为 `not bn`，因为如果使用BN，则不需要偏置。

#### `self.bn`
- **类型**: `nn.BatchNorm2d` 或 `nn.Identity`
- **作用**: 批量归一化层。如果不使用 (`bn=False`)，则替换为恒等映射 `nn.Identity`。

#### `self.act`
- **类型**: `nn.Module`
- **作用**: 激活函数层。

### 成员方法

#### `forward(self, x)`
- **作用**: 标准的前向传播。
- **步骤**:
    1. 输入 `x` 经过转置卷积 `self.conv_transpose`
    2. 结果经过批量归一化 `self.bn`
    3. 最后通过激活函数 `self.act`

#### `forward_fuse(self, x)`
- **作用**: 融合推理模式的前向传播。
- **步骤**:
    1. 输入 `x` 经过转置卷积 `self.conv_transpose`
    2. 直接通过激活函数 `self.act`
- **用途**: 在推理时，可以将 `ConvTranspose` 和 `BatchNorm` 融合，以提高速度。

### 作用

**实现一个标准的转置卷积模块（ConvTranspose -> BN -> Act），通常用于上采样特征图。**

这个模块封装了转置卷积、批量归一化和激活函数，构成了上采样操作的基本单元，常见于解码器或分割头中。

## Focus(类)

### 类代码
```python
class Focus(nn.Module):
    """Focus wh down-sampling."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """Initializes Focus module with convolution, batch normalization and activation."""
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h)
        """Applies focus down-sampling to input tensor."""
        return self.conv(torch.cat(
            (x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))  # cat channels
        # return self.conv(self.contract(x))  # yolov5 v6.0
```

### 参数介绍

#### `c1` (ch_in, 输入通道数)
- **类型**：`int`
- **作用**：输入特征图的通道数。

#### `c2` (ch_out, 输出通道数)
- **类型**：`int`
- **作用**：输出特征图的通道数。

#### `k` (kernel size, 卷积核大小)
- **类型**：`int`，默认 `1`
- **作用**：卷积的卷积核大小。

#### `s` (stride, 步幅)
- **类型**：`int`，默认 `1`
- **作用**：卷积的步幅。

#### `p` (padding, 填充)
- **类型**：`int`，默认 `None`
- **作用**：卷积的填充大小。

#### `g` (groups, 分组)
- **类型**：`int`，默认 `1`
- **作用**：卷积的分组数。

#### `act` (activation, 激活函数)
- **类型**：`bool | nn.Module`，默认 `True`
- **作用**：是否使用激活函数。

### 成员属性

#### `self.conv`
- **类型**: `Conv`
- **作用**: 核心的卷积层。输入通道数为 `c1 * 4`，因为 `Focus` 模块会将输入特征图的通道数变为4倍。

### 成员方法

#### `forward(self, x)`
- **作用**: 标准的前向传播。
- **步骤**:
    1. 将输入 `x` 在 `w` 和 `h` 维度上进行切片，得到4个子特征图。
    2. 将4个子特征图在通道维度上进行拼接，得到一个通道数为 `c1 * 4` 的特征图。
    3. 将拼接后的特征图输入到 `self.conv` 中进行卷积操作。

### 作用

**实现一种特殊的下采样操作，可以在减小特征图尺寸的同时，保留更多的信息。**

`Focus` 模块通过将输入特征图进行切片和拼接，将空间信息转化为通道信息，从而在下采样的同时，保留了更多的信息。这种操作可以有效地减少计算量，同时提高模型的性能。

## GhostConv(类)

### 类代码
```python
class GhostConv(nn.Module):
    """Ghost Convolution https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """Initializes the GhostConv module."""
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """Forward propagation through the GhostConv module."""
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)
```

### 参数介绍

#### `c1` (ch_in, 输入通道数)
- **类型**：`int`
- **作用**：输入特征图的通道数。

#### `c2` (ch_out, 输出通道数)
- **类型**：`int`
- **作用**：输出特征图的通道数。

#### `k` (kernel size, 卷积核大小)
- **类型**：`int`，默认 `1`
- **作用**：第一个卷积层的卷积核大小。

#### `s` (stride, 步幅)
- **类型**：`int`，默认 `1`
- **作用**：第一个卷积层的步幅。

#### `g` (groups, 分组)
- **类型**：`int`，默认 `1`
- **作用**：第一个卷积层的分组数。

#### `act` (activation, 激活函数)
- **类型**：`bool | nn.Module`，默认 `True`
- **作用**：是否使用激活函数。

### 成员属性

#### `self.cv1`
- **类型**: `Conv`
- **作用**: 第一个卷积层，用于生成“内在”特征图（intrinsic feature maps）。

#### `self.cv2`
- **类型**: `Conv`
- **作用**: 第二个卷积层，用于对“内在”特征图进行深度卷积，生成“鬼影”特征图（ghost feature maps）。

### 成员方法

#### `forward(self, x)`
- **作用**: 标准的前向传播。
- **步骤**:
    1. 输入 `x` 经过 `self.cv1` 卷积，得到“内在”特征图 `y`。
    2. 将 `y` 经过 `self.cv2` 卷积，得到“鬼影”特征图。
    3. 将“内在”特征图 `y` 和“鬼影”特征图在通道维度上进行拼接，得到最终的输出。

### 作用

**实现一种轻量级的卷积操作，可以在减少计算量的同时，保持模型的性能。**

`GhostConv` 模块通过将标准卷积分为两步：第一步用少量卷积核生成“内在”特征图，第二步用更廉价的线性变换（深度卷积）生成“鬼影”特征图，然后将两者拼接。这种方法可以有效地减少参数量和计算量，同时保持模型的性能。


## RepConv(类)

### 类代码
```python
class RepConv(nn.Module):
    """RepConv is a basic rep-style block, including training and deploy status.

    This module is used in RT-DETR.
    Based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """Initializes Light Convolution layer with inputs, outputs & optional activation function."""
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """Forward process."""
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward process."""
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        """Returns equivalent kernel and bias by adding 3x3 kernel, 1x1 kernel and identity kernel with their biases."""
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    @staticmethod
    def _pad_1x1_to_3x3_tensor(kernel1x1):
        """Pads a 1x1 tensor to a 3x3 tensor."""
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """Generates appropriate kernels and biases for convolution by fusing branches of the neural network."""
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, "id_tensor"):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        """Combines two convolution layers into a single layer and removes unused attributes from the class."""
        if hasattr(self, "conv"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(
            in_channels=self.conv1.conv.in_channels,
            out_channels=self.conv1.conv.out_channels,
            kernel_size=self.conv1.conv.kernel_size,
            stride=self.conv1.conv.stride,
            padding=self.conv1.conv.padding,
            dilation=self.conv1.conv.dilation,
            groups=self.conv1.conv.groups,
            bias=True,
        ).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("conv1")
        self.__delattr__("conv2")
        if hasattr(self, "nm"):
            self.__delattr__("nm")
        if hasattr(self, "bn"):
            self.__delattr__("bn")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")
```

### 参数介绍

#### `c1` (输入通道数)
- **类型**：`int`
- **作用**：输入特征图的通道数。

#### `c2` (输出通道数)
- **类型**：`int`
- **作用**：输出特征图的通道数。

#### `k` (卷积核大小)
- **类型**：`int`，默认 `3`
- **作用**：卷积核的大小，必须为3。

#### `s` (步幅)
- **类型**：`int`，默认 `1`
- **作用**：卷积的步幅。

#### `p` (填充)
- **类型**：`int`，默认 `1`
- **作用**：卷积的填充大小。

#### `g` (分组数)
- **类型**：`int`，默认 `1`
- **作用**：卷积的分组数。

#### `d` (空洞卷积膨胀率)
- **类型**：`int`，默认 `1`
- **作用**：卷积的膨胀率。

#### `act` (激活函数)
- **类型**：`bool | nn.Module`，默认 `True`
- **作用**：激活函数配置。

#### `bn` (批量归一化)
- **类型**：`bool`，默认 `False`
- **作用**：是否使用批量归一化。

#### `deploy` (部署模式)
- **类型**：`bool`，默认 `False`
- **作用**：是否处于部署模式。

### 成员属性

#### `self.g`
- **类型**：`int`
- **作用**：分组数。

#### `self.c1`
- **类型**：`int`
- **作用**：输入通道数。

#### `self.c2`
- **类型**：`int`
- **作用**：输出通道数。

#### `self.act`
- **类型**：`nn.Module`
- **作用**：激活函数。

#### `self.bn`
- **类型**：`nn.BatchNorm2d` 或 `None`
- **作用**：批量归一化层。

#### `self.conv1`
- **类型**：`Conv`
- **作用**：3x3卷积分支。

#### `self.conv2`
- **类型**：`Conv`
- **作用**：1x1卷积分支。

#### `self.id_tensor`
- **类型**：`torch.Tensor`
- **作用**：身份映射的卷积核张量。

### 成员方法

#### `forward_fuse(self, x)`
- **作用**：融合模式的前向传播。
- **步骤**：输入经过融合后的卷积层和激活函数。

#### `forward(self, x)`
- **作用**：训练模式的前向传播。
- **步骤**：将3x3卷积、1x1卷积和身份映射三个分支的输出相加，然后通过激活函数。

#### `get_equivalent_kernel_bias(self)`
- **作用**：获取等效的卷积核和偏置。
- **返回**：融合后的卷积核和偏置。

#### `_pad_1x1_to_3x3_tensor(self, kernel1x1)`
- **作用**：将1x1卷积核填充到3x3。
- **返回**：填充后的3x3卷积核。

#### `_fuse_bn_tensor(self, branch)`
- **作用**：融合BN和卷积参数。
- **返回**：融合后的卷积核和偏置。

#### `fuse_convs(self)`
- **作用**：融合卷积分支为单个卷积层。
- **步骤**：计算等效卷积核和偏置，创建融合后的卷积层，并删除原始分支。

### 作用

**实现一种结构重参数化的卷积模块，可以在训练时使用多分支结构提升性能，在部署时融合成单分支结构加速推理。**

`RepConv` 模块的核心思想是"训练一个更多分支的模型，部署一个更少分支的模型"。在训练时，通过 `3x3` 卷积、`1x1` 卷积和身份映射三个分支的组合，可以提升模型的性能。在部署时，通过结构重参数化，将三个分支融合成一个 `3x3` 的卷积层，可以加速模型的推理速度。


## ChannelAttention(类)

### 类代码
```python
class ChannelAttention(nn.Module):
    """Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet."""

    def __init__(self, channels: int) -> None:
        """Initializes the ChannelAttention module."""
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies channel attention to the input tensor."""
        return x * self.act(self.fc(self.pool(x)))
```

### 参数介绍

#### `channels` (通道数)
- **类型**：`int`
- **作用**：输入特征图的通道数。

### 成员属性

#### `self.pool`
- **类型**: `nn.AdaptiveAvgPool2d`
- **作用**: 自适应平均池化层，将输入特征图在空间维度上进行平均池化，得到一个 `1x1` 的特征图。

#### `self.fc`
- **类型**: `nn.Conv2d`
- **作用**: 全连接层，用于学习通道间的相关性。

#### `self.act`
- **类型**: `nn.Sigmoid`
- **作用**: Sigmoid 激活函数，用于生成通道注意力权重。

### 成员方法

#### `forward(self, x)`
- **作用**: 标准的前向传播。
- **步骤**:
    1. 输入 `x` 经过 `self.pool` 进行自适应平均池化。
    2. 结果经过 `self.fc` 全连接层。
    3. 结果经过 `self.act` Sigmoid 激活函数，得到通道注意力权重。
    4. 将通道注意力权重与输入 `x` 相乘，得到最终的输出。

### 作用

**实现一种通道注意力机制，可以学习通道间的相关性，从而提升模型的性能。**

`ChannelAttention` 模块通过自适应平均池化、全连接层和 Sigmoid 激活函数，学习了通道间的相关性，并生成了通道注意力权重。将通道注意力权重与输入特征图相乘，可以使模型更加关注重要的通道，从而提升模型的性能。

## SpatialAttention(类)

### 类代码
```python
class SpatialAttention(nn.Module):
    """Spatial-attention module."""

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module with kernel size."""
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        """Applies spatial attention to the input tensor."""
        return x * self.sig(self.cv1(torch.cat((torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]), 1)))
```

### 参数介绍

#### `kernel_size` (卷积核大小)
- **类型**：`int`，默认 `7`
- **作用**：卷积核大小，只能是 `3` 或 `7`。

### 成员属性

#### `self.cv1`
- **类型**: `nn.Conv2d`
- **作用**: 卷积层，用于学习空间注意力权重。

#### `self.sig`
- **类型**: `nn.Sigmoid`
- **作用**: Sigmoid 激活函数，用于生成空间注意力权重。

### 成员方法

#### `forward(self, x)`
- **作用**: 标准的前向传播。
- **步骤**:
    1. 输入 `x` 在通道维度上进行平均池化和最大池化。
    2. 将平均池化和最大池化的结果在通道维度上进行拼接。
    3. 结果经过 `self.cv1` 卷积层。
    4. 结果经过 `self.sig` Sigmoid 激活函数，得到空间注意力权重。
    5. 将空间注意力权重与输入 `x` 相乘，得到最终的输出。

### 作用

**实现一种空间注意力机制，可以学习空间区域的重要性，从而提升模型的性能。**

`SpatialAttention` 模块通过在通道维度上进行平均池化和最大池化，然后通过卷积层和 Sigmoid 激活函数，学习了空间区域的重要性，并生成了空间注意力权重。将空间注意力权重与输入特征图相乘，可以使模型更加关注重要的空间区域，从而提升模型的性能。

## CBAM(类)

### 类代码
```python
class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, c1, kernel_size=7):
        """Initialize CBAM with given channels and kernel size."""
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """Applies the CBAM attention mechanism to the input tensor."""
        return self.spatial_attention(self.channel_attention(x))
```

### 参数介绍

#### `c1` (ch_in, 输入通道数)
- **类型**：`int`
- **作用**：输入特征图的通道数。

#### `kernel_size` (卷积核大小)
- **类型**：`int`，默认 `7`
- **作用**：空间注意力模块的卷积核大小。

### 成员属性

#### `self.channel_attention`
- **类型**: `ChannelAttention`
- **作用**: 通道注意力模块。

#### `self.spatial_attention`
- **类型**: `SpatialAttention`
- **作用**: 空间注意力模块。

### 成员方法

#### `forward(self, x)`
- **作用**: 标准的前向传播。
- **步骤**:
    1. 输入 `x` 经过 `self.channel_attention` 通道注意力模块。
    2. 结果经过 `self.spatial_attention` 空间注意力模块。

### 作用

**实现一种卷积块注意力模块（Convolutional Block Attention Module），可以依次应用通道注意力和空间注意力，从而提升模型的性能。**

`CBAM` 模块将通道注意力和空间注意力两个模块串联起来，先通过通道注意力模块学习通道间的重要性，再通过空间注意力模块学习空间区域的重要性。这种串联的方式可以使模型同时关注重要的通道和重要的空间区域，从而更有效地提升模型的性能。

## Concat(类)

### 类代码
```python
class Concat(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension=1):
        """Initializes the Concat module."""
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """Forward pass for the Concat module."""
        return torch.cat(x, self.d)
```

### 参数介绍

#### `dimension` (拼接维度)
- **类型**：`int`，默认 `1`
- **作用**：指定在哪个维度上进行拼接。

### 成员属性

#### `self.d`
- **类型**: `int`
- **作用**: 存储拼接的维度。

### 成员方法

#### `forward(self, x)`
- **作用**: 标准的前向传播。
- **步骤**:
    1. 输入 `x` 是一个张量列表。
    2. 使用 `torch.cat` 将 `x` 中的所有张量在 `self.d` 维度上进行拼接。

### 作用

**实现一个通用的拼接模块，可以将一个张量列表在指定维度上进行拼接。**

`Concat` 模块封装了 `torch.cat` 函数，使其可以作为一个独立的模块在 `nn.Sequential` 中使用。这在需要将多个分支的输出进行拼接时非常有用。

## Index(类)

### 类代码
```python
class Index(nn.Module):
    """Returns a particular index of the input."""

    def __init__(self, c1, c2, index=0):
        """Returns a particular index of the input."""
        super().__init__()
        self.index = index

    def forward(self, x):
        """
        Forward pass.

        Expects a list of tensors as input.
        """
        return x[self.index]
```

### 参数介绍

#### `c1` (输入通道数)
- **类型**：`int`
- **作用**：指定输入通道数（未在模块中使用，可能用于兼容性）。

#### `c2` (输出通道数)
- **类型**：`int`
- **作用**：指定输出通道数（未在模块中使用，可能用于兼容性）。

#### `index` (索引)
- **类型**：`int`
- **默认值**：`0`
- **作用**：指定要从输入列表中选择的索引。

### 成员属性

#### `self.index`
- **类型**: `int`
- **作用**: 存储要选择的特征张量的索引。

### 成员方法

#### `forward(self, x)`
- **作用**: 标准的前向传播。
- **步骤**:
    1. 输入 `x` 是一个特征张量列表。
    2. 返回列表中指定索引处的张量。

### 作用

**实现一个索引模块，从特征张量列表中选择指定索引的张量。**

`Index` 模块在处理多尺度特征时有用，允许从不同层级的特征列表中选择特定特征，这对于目标检测等任务至关重要。
