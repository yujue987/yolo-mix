## TransformerEncoderLayer(类)

### 类代码
```python
class TransformerEncoderLayer(nn.Module):
    """Defines a single layer of the transformer encoder."""

    def __init__(self, c1, cm=2048, num_heads=8, dropout=0.0, act=nn.GELU(), normalize_before=False):
        """Initialize the TransformerEncoderLayer with specified parameters."""
        super().__init__()
        from ...utils.torch_utils import TORCH_1_9

        if not TORCH_1_9:
            raise ModuleNotFoundError(
                "TransformerEncoderLayer() requires torch>=1.9 to use nn.MultiheadAttention(batch_first=True)."
            )
        self.ma = nn.MultiheadAttention(c1, num_heads, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.fc1 = nn.Linear(c1, cm)
        self.fc2 = nn.Linear(cm, c1)

        self.norm1 = nn.LayerNorm(c1)
        self.norm2 = nn.LayerNorm(c1)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.act = act
        self.normalize_before = normalize_before

    @staticmethod
    def with_pos_embed(tensor, pos=None):
        """Add position embeddings to the tensor if provided."""
        return tensor if pos is None else tensor + pos

    def forward_post(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        """Performs forward pass with post-normalization."""
        q = k = self.with_pos_embed(src, pos)
        src2 = self.ma(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.fc2(self.dropout(self.act(self.fc1(src))))
        src = src + self.dropout2(src2)
        return self.norm2(src)

    def forward_pre(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        """Performs forward pass with pre-normalization."""
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.ma(q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.fc2(self.dropout(self.act(self.fc1(src2))))
        return src + self.dropout2(src2)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        """Forward propagates the input through the encoder module."""
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)
```

### 参数介绍

#### `c1` (通道数)
- **类型**：`int`
- **作用**：输入通道数。

#### `cm` (中间层通道数)
- **类型**：`int`，默认 `2048`
- **作用**：前馈网络的中间层维度。

#### `num_heads` (注意力头数)
- **类型**：`int`，默认 `8`
- **作用**：多头注意力机制中的注意力头数量。

#### `dropout` (dropout比率)
- **类型**：`float`，默认 `0.0`
- **作用**：dropout层的丢弃概率。

#### `act` (激活函数)
- **类型**：`nn.Module`，默认 `nn.GELU()`
- **作用**：前馈网络中的激活函数。

#### `normalize_before` (预归一化标志)
- **类型**：`bool`，默认 `False`
- **作用**：是否在注意力前进行层归一化。

### 成员属性

#### `self.ma`
- **类型**：`nn.MultiheadAttention`
- **作用**：多头自注意力机制，使用batch_first=True模式。

#### `self.fc1/self.fc2`
- **类型**：`nn.Linear`
- **作用**：前馈网络的线性变换层。

#### `self.norm1/self.norm2`
- **类型**：`nn.LayerNorm`
- **作用**：层归一化层。

#### `self.dropout/self.dropout1/self.dropout2`
- **类型**：`nn.Dropout`
- **作用**：dropout层，用于正则化。

#### `self.act`
- **类型**：`nn.Module`
- **作用**：激活函数实例。

#### `self.normalize_before`
- **类型**：`bool`
- **作用**：归一化顺序标志。

### 成员方法

#### `with_pos_embed(self, tensor, pos=None)`
- **作用**：为输入张量添加位置编码（如果提供）。
- **返回**：添加了位置编码的张量或原张量。

#### `forward_post(self, src, src_mask=None, src_key_padding_mask=None, pos=None)`
- **作用**：后归一化模式的前向传播。
- **步骤**：
  1. 添加位置编码到查询和键
  2. 执行多头注意力
  3. 残差连接和dropout
  4. 层归一化
  5. 前馈网络处理
  6. 最终残差连接和归一化

#### `forward_pre(self, src, src_mask=None, src_key_padding_mask=None, pos=None)`
- **作用**：预归一化模式的前向传播。
- **步骤**：
  1. 先进行层归一化
  2. 执行多头注意力
  3. 残差连接和dropout
  4. 再次归一化
  5. 前馈网络处理
  6. 最终残差连接

#### `forward(self, src, src_mask=None, src_key_padding_mask=None, pos=None)`
- **作用**：主前向传播函数，根据normalize_before选择模式。
- **返回**：经过Transformer编码器层处理的特征。

### 作用

**定义Transformer编码器的单个层，实现多头自注意力机制和位置前馈网络的组合。**

`TransformerEncoderLayer` 是Transformer架构的核心组件，通过自注意力机制捕获序列中的长程依赖关系，并通过前馈网络进行非线性变换。支持预归一化和后归一化两种模式，适用于各种序列建模任务。该实现特别优化了batch_first=True的模式，提高了内存效率。

## AIFI(类)

### 类代码
```python
class AIFI(TransformerEncoderLayer):
    """Defines the AIFI transformer layer."""

    def __init__(self, c1, cm=2048, num_heads=8, dropout=0, act=nn.GELU(), normalize_before=False):
        """Initialize the AIFI instance with specified parameters."""
        super().__init__(c1, cm, num_heads, dropout, act, normalize_before)

    def forward(self, x):
        """Forward pass for the AIFI transformer layer."""
        c, h, w = x.shape[1:]
        pos_embed = self.build_2d_sincos_position_embedding(w, h, c)
        # Flatten [B, C, H, W] to [B, HxW, C]
        x = super().forward(x.flatten(2).permute(0, 2, 1), pos=pos_embed.to(device=x.device, dtype=x.dtype))
        return x.permute(0, 2, 1).view([-1, c, h, w]).contiguous()

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.0):
        """Builds 2D sine-cosine position embedding."""
        assert embed_dim % 4 == 0, "Embed dimension must be divisible by 4 for 2D sin-cos position embedding"
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="ij")
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (temperature**omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], 1)[None]
```

### 参数介绍
