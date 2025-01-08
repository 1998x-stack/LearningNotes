# NNLM
通过神经网络，给定当前单词序列，预测下一个单词的概率。
能自动学习到单词的潜在语义特征，从而提高模型的泛化能力。
条件概率：P(wi|w1, w2, ..., wi-1) = f(wi, w1, w2, ..., wi-1)
模型结构：
1. 输入层：将单词映射为词向量（word embedding）
2. 隐藏层：将词向量拼接为一个长向量，然后通过一个非线性变换得到隐藏层的输出。
3. 输出层：通过softmax函数得到下一个单词的概率分布。
损失函数：交叉熵损失函数

局限性：
1. 计算资源消耗
2. 训练时间长
3. 容易过拟合

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List

class NNLM(nn.Module):
    def __init__(self, vocab_size: int, embedding_size: int, hidden_size: int, window_size: int):
        """
        Initialize the neural network components.

        Args:
            vocab_size (int): Size of the vocabulary.
            embedding_size (int): Dimensionality of the word embedding.
            hidden_size (int): Size of the hidden layer.
            window_size (int): The context window size.
        """
        super(NNLM, self).__init__()

        # Word embedding layer
        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        
        # Hidden layer weight matrix
        self.hidden_layer = nn.Linear(embedding_size, hidden_size)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size, vocab_size)
        
        # Activation function for hidden layer
        self.tanh = nn.Tanh()

    def forward(self, context_words: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: Compute the predicted word probabilities.

        Args:
            context_words (torch.Tensor): Tensor of word indices in the context.
        
        Returns:
            torch.Tensor: Predicted word probabilities.
        """
        # Get word embeddings for context words (average the context words)
        context_emb = self.embeddings(context_words).mean(dim=0)
        
        # Hidden layer activation
        hidden_output = self.tanh(self.hidden_layer(context_emb))
        
        # Output layer - probability distribution over vocabulary
        output = self.output_layer(hidden_output)
        
        return output

    def predict(self, context_words: torch.Tensor) -> torch.Tensor:
        """
        Predict the next word based on the context.

        Args:
            context_words (torch.Tensor): Tensor of word indices in the context.
        
        Returns:
            torch.Tensor: Predicted word probabilities after softmax.
        """
        output = self.forward(context_words)
        return torch.softmax(output, dim=0)

def train_model(model: NNLM, sentences: List[List[int]], epochs: int, learning_rate: float):
    """
    Train the NNLM using stochastic gradient descent.
    
    Args:
        model (NNLM): The NNLM model.
        sentences (List[List[int]]): The sentences, each sentence is a list of word indices.
        epochs (int): The number of training epochs.
        learning_rate (float): The learning rate for optimization.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        total_loss = 0.0
        
        for sentence in sentences:
            for i in range(model.window_size, len(sentence) - model.window_size):
                # Extract context words and target word
                context_words = torch.tensor(sentence[i - model.window_size:i] + sentence[i + 1:i + model.window_size + 1])
                target_word = torch.tensor([sentence[i]])
                
                # Zero gradients from the previous step
                optimizer.zero_grad()
                
                # Forward pass to get predicted probabilities
                output = model(context_words)
                
                # Compute loss
                loss = criterion(output.view(1, -1), target_word)
                total_loss += loss.item()
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
        
        # Print loss after each epoch
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss}")

# Example usage
if __name__ == "__main__":
    # Example sentences (using integer indices for simplicity)
    sentences = [
        [0, 1, 2, 3, 4],  # Sentence: I love programming in Python
        [4, 1, 5, 6, 7],  # Sentence: Python is a great programming language
        [0, 3, 2, 4, 1]   # Sentence: I enjoy solving problems with Python
    ]
    
    vocab_size = 8  # Vocabulary size (example)
    embedding_size = 5  # Embedding dimensionality
    hidden_size = 10  # Hidden layer size
    window_size = 2  # Context window size

    # Initialize the NNLM model
    model = NNLM(vocab_size, embedding_size, hidden_size, window_size)
    
    # Train the model
    train_model(model, sentences, epochs=100, learning_rate=0.01)
```