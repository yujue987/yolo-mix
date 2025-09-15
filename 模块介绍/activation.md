
## AGLU(类)

### 类代码
```python
class AGLU(nn.Module):
    """Unified activation function module from https://github.com/kostas1515/AGLU."""

    def __init__(self, device=None, dtype=None) -> None:
        """Initialize the Unified activation function."""
        super().__init__()
        self.act = nn.Softplus(beta=-1.0)
        self.lambd = nn.Parameter(nn.init.uniform_(torch.empty(1, device=device, dtype=dtype)))  # lambda parameter
        self.kappa = nn.Parameter(nn.init.uniform_(torch.empty(1, device=device, dtype=dtype)))  # kappa parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the forward pass of the Unified activation function."""
        lam = torch.clamp(self.lambd, min=0.0001)
        return torch.exp((1 / lam) * self.act((self.kappa * x) - torch.log(lam)))
```

### 参数介绍

#### `device` (设备)
- **类型**：`torch.device` 或 `None`，默认 `None`
- **作用**：指定参数所在的设备（CPU或GPU）。

#### `dtype` (数据类型)
- **类型**：`torch.dtype` 或 `None`，默认 `None`
- **作用**：指定参数的数据类型。

### 成员属性

#### `self.act`
- **类型**：`nn.Softplus`
- **作用**：Softplus激活函数，beta参数设置为-1.0。

#### `self.lambd`
- **类型**：`nn.Parameter`
- **作用**：lambda参数，使用均匀分布初始化。

#### `self.kappa`
- **类型**：`nn.Parameter`
- **作用**：kappa参数，使用均匀分布初始化。

### 成员方法

#### `forward(self, x)`
- **作用**：计算统一激活函数的前向传播。
- **步骤**：
  1. 对lambda参数进行裁剪，确保最小值
  2. 计算激活函数的输出
  3. 返回指数运算结果

### 作用

**实现一种统一的激活函数模块，基于AGLU（Adaptive Gated Linear Unit）设计。**

`AGLU` 模块是一种自适应的门控线性单元激活函数，通过可学习的参数 `lambda` 和 `kappa` 来动态调整激活函数的形状。该激活函数结合了Softplus激活函数的平滑特性和指数运算的非线性特性，能够自适应地调整激活函数的饱和点和斜率，为神经网络提供更强的表达能力。

该模块特别适用于需要自适应激活函数的场景，可以根据输入数据的特点自动调整激活函数的参数，从而提高模型的性能和泛化能力。
