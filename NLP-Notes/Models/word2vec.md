# word2vec
将词语映射到一个低维空间，使得语义相近的词在空间中距离较近。word2vec是一种预测模型，通过上下文预测中心词或者通过中心词预测上下文。
> word2vec有两种模型：CBOW和Skip-gram。
* CBOW (Continuous Bag of Words): 通过上下文预测目标词。
* Skip-Gram: 通过目标词预测上下文。
> 优点：
* 能够捕捉词之间的语义关系。
* 训练速度较快，尤其是通过负采样和分层softmax等优化方法。
* 可以处理大规模语料。
> 缺点：
* 无法捕捉词的多义性（同一个词在不同上下文中有不同的含义）。
* 只能生成固定大小的词向量，无法处理词序列中的语法信息。

## 训练过程
1. 输入大量文本数据，将文本数据分词，得到词汇表。
2. 训练目标：最小化预测目标词与真实词之间的差异（通常使用负对数似然损失函数）。
3. 优化算法：SGD、Negative Sampling、Hierarchical Softmax。

## 负采样
why：不需要计算所有词的概率，只需要计算少量负样本的概率，加速训练过程。
how：对于每个正样本，随机采样K个负样本，将正样本和负样本一起输入到模型中，计算损失函数。
缺点：近似计算，牺牲准确性；无法捕捉到长距离的语法依赖关系；对于oov词无法处理。
损失函数：对于一个正样本和K个负样本，损失函数为：
$$
L = -\log(\sigma(v_{w_O}^T v_{w_I})) - \sum_{k=1}^{K} \log(\sigma(-v_{w_k}^T v_{w_I}))
$$
其中：
1. $v_{w_O}$：目标词的词向量。
2. $v_{w_I}$：上下文词的词向量。
3. $v_{w_k}$：负样本的词向量。
4. $\sigma(x) = \frac{1}{1+e^{-x}}$：sigmoid函数。

负样本的采样策略：高频词更容易被采样到，低频词更难被采样到。如：p^(3/4)

之前的方式：通过softmax计算所有词的概率。公式：
$$
P(w_O|w_I) = \frac{e^{v_{w_O}^T v_{w_I}}}{\sum_{w=1}^{V} e^{v_{w}^T v_{w_I}}}
$$
其中：
1. $V$：词汇表大小。
2. $w_O$：目标词。
3. $w_I$：上下文词。


## 代码
```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import Counter
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from typing import List, Tuple

class Word2Vec(nn.Module):
    """Word2Vec model using PyTorch with Skip-Gram and Negative Sampling.

    Attributes:
        vocab_size (int): Size of the vocabulary.
        embedding_dim (int): Dimensionality of word embeddings.
        W1 (torch.nn.Embedding): Input embedding layer (target words).
        W2 (torch.nn.Embedding): Output embedding layer (context words).
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int):
        """
        Initializes the Word2Vec model with random embeddings.

        Args:
            vocab_size (int): Size of the vocabulary.
            embedding_dim (int): Dimensionality of the word embeddings.
        """
        super(Word2Vec, self).__init__()
        # Define the embeddings for input (W1) and output (W2) layers
        self.W1 = nn.Embedding(vocab_size, embedding_dim)
        self.W2 = nn.Embedding(vocab_size, embedding_dim)
        
        # Initialize the weights with random values
        self.W1.weight.data.uniform_(-1, 1)
        self.W2.weight.data.uniform_(-1, 1)
        
    def forward(self, target_idx: torch.Tensor, context_idx: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model, calculates the Skip-Gram with Negative Sampling loss.

        Args:
            target_idx (torch.Tensor): Indices of the target words.
            context_idx (torch.Tensor): Indices of the context words.

        Returns:
            torch.Tensor: The loss for the given target and context word indices.
        """
        # Get embeddings for target and context words
        target_embed = self.W1(target_idx)
        context_embed = self.W2(context_idx)
        
        # Calculate dot product between target and context word vectors
        score = torch.sum(target_embed * context_embed, dim=1)
        loss = torch.sum(torch.log(torch.sigmoid(score)))
        
        return -loss  # Negative log-likelihood loss
    
    def negative_sampling(self, target_idx: int, vocab_size: int, k: int = 5) -> List[int]:
        """
        Samples k negative words from the vocabulary for a given target word.

        Args:
            target_idx (int): Index of the target word.
            vocab_size (int): Size of the vocabulary.
            k (int): Number of negative samples to sample.

        Returns:
            List[int]: Indices of the negative samples.
        """
        negative_samples = []
        while len(negative_samples) < k:
            negative_word = random.choice(range(vocab_size))
            if negative_word != target_idx:
                negative_samples.append(negative_word)
        return negative_samples

class Word2VecTrainer:
    """Trainer for the Word2Vec model with Skip-Gram and Negative Sampling."""

    def __init__(self, model: Word2Vec, corpus: List[List[str]], vocab: List[str], 
                 batch_size: int = 1, epochs: int = 10, learning_rate: float = 0.01):
        """
        Initializes the trainer for training the Word2Vec model.

        Args:
            model (Word2Vec): The Word2Vec model to be trained.
            corpus (List[List[str]]): Tokenized and preprocessed text corpus.
            vocab (List[str]): Vocabulary list.
            batch_size (int): Size of the training batch.
            epochs (int): Number of training epochs.
            learning_rate (float): Learning rate for model optimization.
        """
        self.model = model
        self.corpus = corpus
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.word2idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx2word = {idx: word for idx, word in enumerate(vocab)}
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.optimizer = optim.SGD(model.parameters(), lr=self.learning_rate)

    def generate_batches(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generates batches for training.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Target indices and context indices for the batch.
        """
        for sentence in self.corpus:
            for i, target_word in enumerate(sentence):
                target_idx = self.word2idx[target_word]
                
                # Define context window: 2 words before and after the target
                context_window = sentence[max(0, i - 2):i] + sentence[i + 1:i + 3]
                
                context_indices = [self.word2idx[word] for word in context_window]
                
                yield torch.tensor([target_idx] * len(context_indices)), torch.tensor(context_indices)

    def train(self) -> None:
        """Trains the Word2Vec model on the provided corpus."""
        total_loss = 0
        for epoch in range(self.epochs):
            epoch_loss = 0
            for target_idx, context_idx in self.generate_batches():
                self.optimizer.zero_grad()
                
                # Forward pass to calculate loss
                loss = self.model(target_idx, context_idx)
                
                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss}")
            total_loss += epoch_loss

        print(f"Training finished, Total Loss: {total_loss}")


def preprocess_text(corpus: List[str]) -> List[List[str]]:
    """Preprocesses the text by tokenizing and filtering non-alphabetical words.

    Args:
        corpus (List[str]): Raw text corpus.

    Returns:
        List[List[str]]: Tokenized and cleaned corpus.
    """
    return [text.lower().split() for text in corpus]

def visualize_embeddings(model: Word2Vec, vocab: List[str]) -> None:
    """Visualizes the learned word embeddings in 2D using PCA.

    Args:
        model (Word2Vec): The trained Word2Vec model.
        vocab (List[str]): List of words in the vocabulary.
    """
    # Get the embeddings for all words in the vocabulary
    embeddings = model.W1.weight.data.cpu().numpy()
    
    # Reduce the embeddings to 2D using PCA
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)
    
    # Normalize the reduced embeddings
    reduced_embeddings = normalize(reduced_embeddings)
    
    # Plot the embeddings
    plt.figure(figsize=(10, 10))
    for i, word in enumerate(vocab):
        plt.scatter(reduced_embeddings[i, 0], reduced_embeddings[i, 1])
        plt.text(reduced_embeddings[i, 0], reduced_embeddings[i, 1], word, fontsize=12)
    
    plt.title("Word2Vec Word Embeddings Visualization")
    plt.show()


# Example usage:

# Sample corpus for training
corpus = [
    "This is a simple example of word2vec implementation",
    "Word2Vec is a technique to learn word representations",
    "Word embeddings can be used in many natural language processing tasks"
]

# Preprocess the text
processed_corpus = preprocess_text(corpus)

# Build the vocabulary
flattened_corpus = [word for sentence in processed_corpus for word in sentence]
word_counts = Counter(flattened_corpus)
vocab = list(word_counts.keys())

# Initialize Word2Vec model
embedding_dim = 50
model = Word2Vec(len(vocab), embedding_dim)

# Initialize trainer
trainer = Word2VecTrainer(model, processed_corpus, vocab, epochs=10, learning_rate=0.1)

# Train the model
trainer.train()

# Visualize embeddings
visualize_embeddings(model, vocab)
```