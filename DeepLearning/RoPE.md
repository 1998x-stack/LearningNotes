# RoPE

WHAT：Rope改进了传统的绝对位置编码（如sin-cos编码）方法，使用**旋转机制**来编码位置信息。这种方法在某些应用场景中（如长序列建模）能够提高模型的效率和效果。

WHY：传统的绝对位置编码方法（如sin-cos编码）在长序列建模中存在一些问题，如：不能直接对序列的长度和结构做出自适应的调整。

核心思想：
Rope是为了解决这些问题提出的一种位置编码方法，其核心思想是使用旋转矩阵（或类似的机制）将位置信息动态地集成到注意力机制中，而不是将位置编码作为附加向量直接加到输入上。

公式：
$$
\begin{aligned}
& \text{Rope} = \text{Rope} \times \text{RotationMatrix} \\
& \text{RotationMatrix} = \begin{bmatrix} \cos(\theta) & -\sin(\theta) \\ \sin(\theta) & \cos(\theta) \end{bmatrix}
\end{aligned}
$$

代码：

```python
import torch
import math

class RotaryPositionalEmbedding:
    def __init__(self, dim: int, max_length: int):
        self.dim = dim
        self.max_length = max_length
        
        # Precompute rotation matrices for all positions up to max_length
        self.rot_matrices = self.create_rotary_matrices()

    def create_rotary_matrices(self):
        """
        创建旋转矩阵
        :return: 返回旋转矩阵列表
        """
        angles = torch.arange(0, self.max_length, dtype=torch.float32) / self.max_length * math.pi
        rotation_matrices = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        return rotation_matrices

    def apply_rotary_embedding(self, x: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        """
        将旋转位置编码应用于输入向量
        :param x: 输入张量 (batch_size, seq_len, dim)
        :param position: 序列位置
        :return: 应用位置编码后的张量
        """
        rotary_matrix = self.rot_matrices[position]
        return x * rotary_matrix

# Example of usage:
max_length = 512
dim = 256
batch_size = 32
seq_len = 100

rop = RotaryPositionalEmbedding(dim, max_length)
input_tensor = torch.randn(batch_size, seq_len, dim)
position = torch.arange(seq_len)

# Apply rotary embedding
output_tensor = rop.apply_rotary_embedding(input_tensor, position)
```

### 基于正弦和余弦的编码

基于正弦和余弦的编码方法是最常见的绝对位置编码方法，其核心思想是使用正弦（sin）和余弦（cos）函数生成一系列的数值，这些数值随着位置变化而变化，确保每个位置都有唯一的编码。这种方法的优点是它不需要学习额外的参数，且可以直接用于任意长度的序列。


公式：
$$
PE_{(pos, 2i)} = sin(pos / 10000^{2i / d_{model}})
\\
PE_{(pos, 2i+1)} = cos(pos / 10000^{2i / d_{model}})
$$
```python
import torch
import math

class PositionalEncoding:
    def __init__(self, d_model: int, max_len: int):
        """
        初始化位置编码类
        :param d_model: 模型的维度
        :param max_len: 序列的最大长度
        """
        self.d_model = d_model
        self.max_len = max_len

        # 初始化位置编码矩阵
        self.position_encoding = self.create_position_encoding()

    def create_position_encoding(self):
        """
        创建位置编码
        :return: 位置编码矩阵 (max_len, d_model)
        """
        position = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * -(math.log(10000.0) / self.d_model))  # (d_model / 2,)

        pe = torch.zeros(self.max_len, self.d_model)  # (max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)  # 2i
        pe[:, 1::2] = torch.cos(position * div_term)  # 2i+1

        return pe

    def forward(self, batch_size: int):
        """
        获取位置编码并与输入嵌入相加
        :param batch_size: 批次大小
        :return: 位置编码后的张量
        """
        return self.position_encoding.unsqueeze(0).repeat(batch_size, 1, 1)

# 示例使用
batch_size = 32
d_model = 512
max_len = 100

pe = PositionalEncoding(d_model, max_len)
pos_encoding = pe.forward(batch_size)
```