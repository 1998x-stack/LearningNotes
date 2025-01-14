# BatchNorm
> WHAT:
深度学习中的技术，旨在解决训练深层神经网络时遇到的一些常见问题，如梯度消失、梯度爆炸以及模型训练速度慢等问题

> 核心思想：对每一层的输入进行归一化，使得每一层的输入具有相对稳定的分布，有利于模型的训练。
1. 规范化：将每一层的输入规范化为均值为0，方差为1的分布。
2. 平移和缩放：通过学习两个参数，对规范化后的数据进行平移和缩放，使得网络可以学习到更复杂的数据分布。

> 步骤：
1. 计算每一层的均值和方差。公式：$$\mu = \frac{1}{m} \sum_{i=1}^{m} x_i$$，$$\sigma^2 = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu)^2$$
2. 对每一层的输入进行规范化。公式：$$\hat{x_i} = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}$$
3. 对规范化后的数据进行平移和缩放。公式：$$y_i = \gamma \hat{x_i} + \beta$$

> 优点：
1. 加速收敛：使得网络更容易优化，加速模型的收敛速度。
2. 减少梯度消失：使得网络更深，减少梯度消失问题。

> 代码：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BatchNormLayer(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(BatchNormLayer, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # Learnable parameters
        self.gamma = nn.Parameter(torch.ones(num_features))  # Scale factor
        self.beta = nn.Parameter(torch.zeros(num_features))  # Shift (bias)

        # Running mean and variance (for inference)
        self.running_mean = torch.zeros(num_features)
        self.running_var = torch.ones(num_features)

    def forward(self, x):
        if self.training:
            # During training: compute the mean and variance of the current batch
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)

            # Normalize the batch
            x_norm = (x - batch_mean) / torch.sqrt(batch_var + self.eps)

            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
        else:
            # During inference: use running statistics (mean and variance)
            x_norm = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)

        # Apply scaling and shifting
        out = self.gamma * x_norm + self.beta
        return out
```