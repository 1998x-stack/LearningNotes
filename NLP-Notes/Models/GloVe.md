# GloVe
> what：基于全局共现统计特征，进行矩阵分解的词嵌入方法

> why：捕捉全局语义信息，避免了word2vec的局部信息捕捉问题

> 核心思想：通过最小化词向量之间的点积和共现词频的差异，来学习词向量；

> 步骤
1. 构建共现矩阵X，Xij表示词i和词j共现的次数（在特定窗口内）
2. 通过矩阵分解，得到词向量矩阵U和V，使得U*V^T尽可能接近X
3. 通过最小化损失函数，学习词向量
目标函数：
$$
J = \sum_{i,j=1}^{V} f(X_{ij})(u_i^Tv_j + b_i + b_j - log(X_{ij}))^2
$$
其中：
- $X_{ij}$表示词i和词j的共现次数
- $u_i$和$v_j$分别表示词i和词j的词向量
- $b_i$和$b_j$分别表示词i和词j的偏置项
- f是权重函数，用于平衡高频词和低频词的权重

> 特点
1. 捕捉全局语义信息
2. 降低了word2vec的训练复杂度
3. 词关系捕捉更加准确，能捕捉词与词之间的关系（如同义词、反义词等）

> 代码
```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import Counter
from typing import List, Tuple, Dict

# 1. 数据预处理 - 构建共现矩阵
class CoOccurrenceMatrix:
    """
    A class to build a co-occurrence matrix from a text corpus.
    """

    def __init__(self, window_size: int):
        """
        Initializes the CoOccurrenceMatrix class with a given window size for context.

        Args:
            window_size (int): The context window size to define word pairs for co-occurrence.
        """
        self.window_size = window_size
        self.word_to_index = {}
        self.index_to_word = {}
        self.cooccurrence_matrix = None

    def build_vocab(self, corpus: List[str]) -> None:
        """
        Builds the vocabulary from the given corpus.

        Args:
            corpus (List[str]): A list of sentences where each sentence is a string of words.
        """
        word_counts = Counter()
        for sentence in corpus:
            word_counts.update(sentence.split())

        # 构建词汇表
        self.word_to_index = {word: idx for idx, (word, _) in enumerate(word_counts.items())}
        self.index_to_word = {idx: word for word, idx in self.word_to_index.items()}

        # 初始化共现矩阵
        self.cooccurrence_matrix = np.zeros((len(self.word_to_index), len(self.word_to_index)))

    def build_cooccurrence_matrix(self, corpus: List[str]) -> None:
        """
        Builds the co-occurrence matrix from the given corpus.

        Args:
            corpus (List[str]): A list of sentences where each sentence is a string of words.
        """
        for sentence in corpus:
            words = sentence.split()
            for i, word in enumerate(words):
                word_idx = self.word_to_index[word]
                # Collect context words within the window size
                for j in range(max(0, i - self.window_size), min(len(words), i + self.window_size + 1)):
                    if i != j:
                        context_word_idx = self.word_to_index[words[j]]
                        self.cooccurrence_matrix[word_idx][context_word_idx] += 1

    def get_cooccurrence_matrix(self) -> np.ndarray:
        """Returns the co-occurrence matrix."""
        return self.cooccurrence_matrix

    def get_vocab_size(self) -> int:
        """Returns the size of the vocabulary."""
        return len(self.word_to_index)


# 2. GloVe 模型定义
class GloVe(nn.Module):
    """
    GloVe (Global Vectors for Word Representation) model class.
    """

    def __init__(self, vocab_size: int, embedding_dim: int, alpha: float = 0.75):
        """
        Initializes the GloVe model with embeddings for words and context.

        Args:
            vocab_size (int): The size of the vocabulary.
            embedding_dim (int): The dimensionality of the word embeddings.
            alpha (float): The exponent for the weighting function.
        """
        super(GloVe, self).__init__()

        # 初始化词向量和上下文向量
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # 权重初始化
        self.word_embeddings.weight.data.uniform_(-0.5 / embedding_dim, 0.5 / embedding_dim)
        self.context_embeddings.weight.data.uniform_(-0.5 / embedding_dim, 0.5 / embedding_dim)

        # 定义加权函数
        self.alpha = alpha

    def forward(self, word_indices: torch.Tensor, context_indices: torch.Tensor, cooccurrence: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass for the GloVe model.

        Args:
            word_indices (torch.Tensor): The indices of the target words.
            context_indices (torch.Tensor): The indices of the context words.
            cooccurrence (torch.Tensor): The co-occurrence counts for each word-context pair.

        Returns:
            torch.Tensor: The loss computed from the GloVe objective.
        """
        # 获取词向量和上下文向量
        word_vectors = self.word_embeddings(word_indices)
        context_vectors = self.context_embeddings(context_indices)

        # 计算预测值（内积）
        dot_product = (word_vectors * context_vectors).sum(dim=1)

        # 使用加权函数计算损失
        weight = torch.pow(cooccurrence, self.alpha)
        loss = torch.sum(weight * torch.pow(dot_product - torch.log(cooccurrence), 2))

        return loss


# 3. 模型训练
def train_glove(corpus: List[str], window_size: int, embedding_dim: int, epochs: int, learning_rate: float) -> GloVe:
    """
    Trains a GloVe model based on the given corpus.

    Args:
        corpus (List[str]): The input corpus (list of sentences).
        window_size (int): The context window size.
        embedding_dim (int): The dimensionality of the word embeddings.
        epochs (int): The number of training epochs.
        learning_rate (float): The learning rate for the optimizer.

    Returns:
        GloVe: The trained GloVe model.
    """
    # 1. 构建共现矩阵
    cooc_matrix = CoOccurrenceMatrix(window_size)
    cooc_matrix.build_vocab(corpus)
    cooc_matrix.build_cooccurrence_matrix(corpus)

    vocab_size = cooc_matrix.get_vocab_size()
    cooccurrence_matrix = cooc_matrix.get_cooccurrence_matrix()

    # 将共现矩阵转换为 PyTorch 张量
    cooccurrence_tensor = torch.tensor(cooccurrence_matrix, dtype=torch.float32)
    word_indices = []
    context_indices = []

    # 创建训练数据
    for i in range(vocab_size):
        for j in range(vocab_size):
            if cooccurrence_matrix[i][j] > 0:
                word_indices.append(i)
                context_indices.append(j)

    word_indices = torch.tensor(word_indices)
    context_indices = torch.tensor(context_indices)

    # 初始化模型和优化器
    model = GloVe(vocab_size, embedding_dim)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # 2. 训练模型
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # 计算损失
        loss = model(word_indices, context_indices, cooccurrence_tensor)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()

        # 打印损失
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    return model


# 4. 测试 GloVe 模型
def test_glove_model(model: GloVe, word_to_index: Dict[str, int], index_to_word: Dict[int, str], test_word: str) -> None:
    """
    Test the trained GloVe model by printing the embedding for a given word.

    Args:
        model (GloVe): The trained GloVe model.
        word_to_index (Dict[str, int]): The dictionary mapping words to indices.
        index_to_word (Dict[int, str]): The dictionary mapping indices to words.
        test_word (str): The word to test the model on.
    """
    word_idx = word_to_index[test_word]
    word_embedding = model.word_embeddings(torch.tensor([word_idx]))
    print(f"Embedding for word '{test_word}': {word_embedding}")

# 5. 主程序
if __name__ == "__main__":
    # 示例语料库
    corpus = [
        "I love machine learning",
        "FastText is a great tool for NLP",
        "Word embeddings are helpful for many NLP tasks",
        "GloVe is a powerful algorithm for word vectors"
    ]

    # 模型参数
    window_size = 2  # 上下文窗口大小
    embedding_dim = 50  # 嵌入维度
    epochs = 100  # 训练轮数
    learning_rate = 0.01  # 学习率

    # 训练 GloVe 模型
    model = train_glove(corpus, window_size, embedding_dim, epochs, learning_rate)

    # 获取词汇表和索引
    cooc_matrix = CoOccurrenceMatrix(window_size)
    cooc_matrix.build_vocab(corpus)
    word_to_index = cooc_matrix.word_to_index
    index_to_word = cooc_matrix.index_to_word

    # 测试模型
    test_word = "GloVe"
    test_glove_model(model, word_to_index, index_to_word, test_word)
```