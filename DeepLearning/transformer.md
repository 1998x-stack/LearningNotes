# Transformer
> WHAT: 不依赖顺序的计算，通过自注意力机制实现并行计算，适用于序列到序列的任务，如机器翻译、文本摘要等。

> 关键部分：
1. Self-Attention Mechanism
2. Feed Forward Neural Network
3. Layer Normalization
4. Positional Encoding
5. Residual Connection
6. Multi-Head Attention

> 自注意力机制：模型处理序列中的每个元素时，会考虑序列中其他元素的信息。

公式：
$$Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$$
其中，$Q, K, V$分别是Query, Key, Value，$d_k$是Key的维度。

代码：
```python
def attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute the scaled dot-product attention.
    
    Args:
        Q (torch.Tensor): Query matrix of shape (batch_size, num_queries, d_k)
        K (torch.Tensor): Key matrix of shape (batch_size, num_keys, d_k)
        V (torch.Tensor): Value matrix of shape (batch_size, num_values, d_v)
    
    Returns:
        torch.Tensor: The resulting attention output of shape (batch_size, num_queries, d_v)
    """
    # Step 1: Compute the dot product of Q and K transpose, scaling by sqrt(d_k)
    d_k = Q.size(-1)  # d_k is the dimensionality of the key vectors
    attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    # Step 2: Apply softmax to get attention weights
    attention_weights = F.softmax(attention_scores, dim=-1)
    # Step 3: Compute the weighted sum of the value vectors
    output = torch.matmul(attention_weights, V)
    return output
```

> 多头注意力机制：通过多个注意力头并行计算，增加模型的表达能力。

主要思想是并行计算多个注意力头，从不同的子空间捕捉输入的多种信息。相比单头注意力机制，多头注意力通过将查询、键和值分成多个“头”来丰富模型对输入序列的表示能力，从而更好地捕捉复杂的关系和上下文。

步骤：

1. 将输入通过线性变换得到Q、K、V，每个头的维度为$d_k$，独立计算。
2. 对每个头进行scaled dot-product attention计算。
3. 将多个头的输出拼接，通过线性变换得到最终输出。

公式：
$$
Q_i = QW_i^Q\\
K_i = KW_i^K\\
V_i = VW_i^V
$$
其中：
- $Q, K, V$是输入的Query, Key, Value矩阵
- $W_i^Q, W_i^K, W_i^V$是第$i$个头的权重矩阵，shape为$(d_{model}, d_k)$

代码：
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """
    多头注意力机制 (Multi-Head Attention) 实现
    """
    def __init__(self, embed_size: int, num_heads: int):
        """
        初始化多头注意力机制。
        
        Args:
            embed_size (int): 输入的词嵌入的维度。
            num_heads (int): 注意力头的数量。
        """
        super(MultiHeadAttention, self).__init__()
        
        self.num_heads = num_heads
        self.embed_size = embed_size
        self.d_k = embed_size // num_heads  # 每个头的维度

        # 确保 embed_size 能被 num_heads 整除
        assert embed_size % num_heads == 0, "Embedding size must be divisible by the number of heads."

        # 查询、键、值的线性变换
        self.W_Q = nn.Linear(embed_size, embed_size)
        self.W_K = nn.Linear(embed_size, embed_size)
        self.W_V = nn.Linear(embed_size, embed_size)

        # 输出线性变换
        self.W_O = nn.Linear(embed_size, embed_size)
        
    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        将输入的张量拆分为多个头，形状变为 (batch_size, num_heads, seq_length, d_k)
        
        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, seq_length, embed_size)
        
        Returns:
            torch.Tensor: 拆分后的张量，形状为 (batch_size, num_heads, seq_length, d_k)
        """
        batch_size, seq_length, _ = x.size()
        x = x.view(batch_size, seq_length, self.num_heads, self.d_k)
        return x.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_length, d_k)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """
        前向传播过程，计算多头注意力。
        
        Args:
            Q (torch.Tensor): 查询矩阵，形状为 (batch_size, seq_length, embed_size)
            K (torch.Tensor): 键矩阵，形状为 (batch_size, seq_length, embed_size)
            V (torch.Tensor): 值矩阵，形状为 (batch_size, seq_length, embed_size)
        
        Returns:
            torch.Tensor: 多头注意力输出，形状为 (batch_size, seq_length, embed_size)
        """
        batch_size = Q.size(0)

        # 计算 Q、K、V 的线性变换
        Q = self.W_Q(Q)
        K = self.W_K(K)
        V = self.W_V(V)

        # 将 Q、K、V 拆分成多个头
        Q = self.split_heads(Q)  # (batch_size, num_heads, seq_length, d_k)
        K = self.split_heads(K)  # (batch_size, num_heads, seq_length, d_k)
        V = self.split_heads(V)  # (batch_size, num_heads, seq_length, d_k)

        # 计算点积注意力
        attention_scores = torch.matmul(Q, K.permute(0, 1, 3, 2)) / math.sqrt(self.d_k)  # (batch_size, num_heads, seq_length, seq_length)
        
        # 应用 softmax 得到权重
        attention_weights = F.softmax(attention_scores, dim=-1)  # (batch_size, num_heads, seq_length, seq_length)

        # 计算加权求和结果
        attention_output = torch.matmul(attention_weights, V)  # (batch_size, num_heads, seq_length, d_k)

        # 合并多个头的输出
        attention_output = attention_output.permute(0, 2, 1, 3).contiguous()  # (batch_size, seq_length, num_heads, d_k)
        attention_output = attention_output.view(batch_size, -1, self.embed_size)  # (batch_size, seq_length, embed_size)

        # 输出线性变换
        output = self.W_O(attention_output)
        
        return output

# 测试代码
batch_size = 2
seq_length = 5
embed_size = 8  # 嵌入维度
num_heads = 2  # 注意力头数

# 随机生成输入数据
Q = torch.rand(batch_size, seq_length, embed_size)
K = torch.rand(batch_size, seq_length, embed_size)
V = torch.rand(batch_size, seq_length, embed_size)

# 创建多头注意力模块并计算输出
multi_head_attention = MultiHeadAttention(embed_size, num_heads)
output = multi_head_attention(Q, K, V)

print("Output shape:", output.shape)
```