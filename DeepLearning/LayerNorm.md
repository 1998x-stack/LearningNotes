# LayerNorm
> WHAT:
针对神经网络训练中常见问题的技术，主要用于解决 Batch Normalization（BN）在某些情境下的局限性，尤其是在处理小批量数据或序列数据时。

> WHY:
1. 小批量数据：BN在小批量数据上的表现不佳（如batch size为1，2时，均值和方差计算有误），因为它是对每个特征维度进行归一化，而不是对整个样本进行归一化。
2. 序列数据：BN在处理序列数据时，由于每个时间步的输入都会有不同的均值和方差，因此BN在序列数据上的表现也不佳。

> HOW:
1. LayerNorm的计算方式与BN不同，它是对每个样本的每个特征维度进行归一化，而不是对每个特征维度进行归一化。
2. LayerNorm的计算公式如下：
$$
\text{LayerNorm}(x) = \gamma \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$
其中，$\mu$和$\sigma$分别是样本的均值和方差，$\gamma$和$\beta$是可学习的参数，$\epsilon$是一个很小的数，用于防止分母为0。

> LayerNorm的优点：

    - 不依赖于batch size，因此适用于小批量数据。
    - 不依赖于样本的顺序，因此适用于序列数据。
    - 可以用于训练和推理，因为不需要保存均值和方差。

> code:

```python
import torch
import torch.nn as nn
class LayerNormLayer(nn.Module):
    """
    自定义 Layer Normalization 层
    """
    def __init__(self, num_features, eps=1e-5):
        super(LayerNormLayer, self).__init__()
        self.num_features = num_features
        self.eps = eps
        
        # 可学习的缩放因子和偏置
        self.gamma = nn.Parameter(torch.ones(num_features))  # 缩放因子
        self.beta = nn.Parameter(torch.zeros(num_features))  # 偏置

    def forward(self, x):
        # 计算均值和方差
        mean = x.mean(dim=-1, keepdim=True)  # 在最后一个维度上计算均值
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # 计算方差
        
        # 标准化
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        # 缩放和平移
        out = self.gamma * x_norm + self.beta
        return out
```