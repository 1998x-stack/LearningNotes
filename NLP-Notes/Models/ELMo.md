# ELMo
> WHAT: ELMo 通过上下文信息动态地生成词向量，这使得它能够为同一个词在不同的上下文中生成不同的向量表示，从而更好地捕捉语言的细节

> 核心思想：在上下文中生成词向量，而不是固定的词向量。

> 步骤
1. 用双向LSTM对语料进行预训练
2. 用预训练的LSTM对每个词生成向量表示
3. 将这些向量表示作为输入，训练一个分类器

> 模型结构
1. 预训练的双向语言模型（BiLM）：
    * 一个双向 LSTM 网络，处理从左到右和从右到左的文本序列。
2. 上下文相关的词嵌入层：
    * 从 LSTM 的每一层中提取的词向量，并通过加权平均生成最终的词嵌入。

> 优势
1. 上下文敏感：不同于 Word2Vec 和 GloVe，ELMo 能够为同一词在不同上下文中生成不同的词向量
2. 预训练与微调：ELMo 既可以用于预训练，也可以用于微调，适用于多种 NLP 任务


> 代码

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from typing import List

class CharacterLevelCNN(nn.Module):
    """
    A CNN for character-level processing of words.
    It creates word representations based on characters using a convolutional layer.
    """

    def __init__(self, vocab_size: int, embed_size: int, kernel_size: int = 5, num_filters: int = 100) -> None:
        """
        Initializes the CNN for character-level processing.
        
        Args:
            vocab_size (int): Size of the character vocabulary.
            embed_size (int): Size of the character embedding.
            kernel_size (int, optional): The size of the convolutional kernel. Default is 5.
            num_filters (int, optional): The number of filters in the convolution. Default is 100.
        """
        super(CharacterLevelCNN, self).__init__()
        
        # Character embedding layer
        self.character_embeddings = nn.Embedding(vocab_size, embed_size)
        
        # Convolutional layer with kernel size, padding, and number of filters
        self.conv = nn.Conv1d(embed_size, num_filters, kernel_size, padding=kernel_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CNN to process character-level information.
        
        Args:
            x (torch.Tensor): A tensor of shape [batch_size, word_length] containing character indices.
        
        Returns:
            torch.Tensor: The word representations of shape [batch_size, num_filters].
        """
        x = self.character_embeddings(x)  # [batch_size, word_length, embed_size]
        x = x.permute(0, 2, 1)  # [batch_size, embed_size, word_length]
        
        x = F.relu(self.conv(x))  # [batch_size, num_filters, word_length]
        x = F.max_pool1d(x, x.size(2))  # [batch_size, num_filters, 1]
        
        return x.squeeze(2)  # [batch_size, num_filters]


class BidirectionalLSTM(nn.Module):
    """
    A bidirectional LSTM to capture both forward and backward contextual information for each word.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2) -> None:
        """
        Initializes the bidirectional LSTM.
        
        Args:
            input_dim (int): The input dimension size (number of features).
            hidden_dim (int): The number of hidden units in the LSTM.
            num_layers (int, optional): Number of LSTM layers. Default is 2.
        """
        super(BidirectionalLSTM, self).__init__()
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, 
                            bidirectional=True, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the bidirectional LSTM.
        
        Args:
            x (torch.Tensor): The input tensor of shape [batch_size, seq_len, input_dim].
        
        Returns:
            torch.Tensor: The output of the LSTM of shape [batch_size, seq_len, hidden_dim * 2].
        """
        out, _ = self.lstm(x)
        return out


class ELMoModel(nn.Module):
    """
    The full ELMo model, which combines a character-level CNN and a bidirectional LSTM to generate contextual word embeddings.
    """

    def __init__(self, vocab_size: int, char_embed_size: int, lstm_hidden_size: int, num_layers: int = 2,
                 kernel_size: int = 5, num_filters: int = 100) -> None:
        """
        Initializes the ELMo model.
        
        Args:
            vocab_size (int): Size of the character vocabulary.
            char_embed_size (int): Size of the character embedding.
            lstm_hidden_size (int): Size of the hidden layer in the LSTM.
            num_layers (int, optional): Number of LSTM layers. Default is 2.
            kernel_size (int, optional): Size of the convolutional kernel. Default is 5.
            num_filters (int, optional): Number of filters in the convolution. Default is 100.
        """
        super(ELMoModel, self).__init__()
        
        # Character-level CNN for word embeddings
        self.char_cnn = CharacterLevelCNN(vocab_size, char_embed_size, kernel_size, num_filters)
        
        # Bidirectional LSTM for contextual embeddings
        self.bilstm = BidirectionalLSTM(num_filters, lstm_hidden_size, num_layers)

    def forward(self, char_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the ELMo model.
        
        Args:
            char_ids (torch.Tensor): A tensor of shape [batch_size, seq_len, word_length] containing character indices.
        
        Returns:
            torch.Tensor: The output word embeddings of shape [batch_size, seq_len, hidden_dim * 2].
        """
        # 1. Apply character-level CNN to obtain word embeddings
        word_embeddings = self.char_cnn(char_ids)  # [batch_size, seq_len, num_filters]
        
        # 2. Pass word embeddings through BiLSTM to obtain contextualized embeddings
        contextual_embeddings = self.bilstm(word_embeddings)  # [batch_size, seq_len, hidden_dim * 2]
        
        return contextual_embeddings


def test_elmo_model():
    """
    测试 ELMo 模型的前向传播过程。
    """
    # 参数设置
    vocab_size = 100  # 假设字符字典大小为 100
    char_embed_size = 30  # 字符级别的嵌入维度
    lstm_hidden_size = 256  # LSTM 的隐层维度
    batch_size = 2  # 批大小
    seq_len = 4  # 每个句子的词数
    word_length = 6  # 每个单词的字符数
    
    # 初始化模型
    elmo_model = ELMoModel(vocab_size, char_embed_size, lstm_hidden_size)
    
    # 随机生成字符索引作为输入
    char_ids = torch.randint(0, vocab_size, (batch_size, seq_len, word_length))
    
    # 前向传播，获得上下文化的词嵌入
    contextual_embeddings = elmo_model(char_ids)
    
    print("Contextual embeddings shape:", contextual_embeddings.shape)


if __name__ == "__main__":
    test_elmo_model()

```

---

```python
import torch
from allennlp.modules.elmo import Elmo, batch_to_ids
from typing import List, Tuple

class ElmoEmbedder:
    """
    A class to handle the embedding process using the ELMo model for contextual word embeddings.

    Attributes:
        elmo (Elmo): A pretrained ELMo model to generate contextual embeddings.
    """

    def __init__(self, options_file: str, weight_file: str, num_layers: int = 2, dropout: float = 0.5) -> None:
        """
        Initializes the ElmoEmbedder class with a pretrained ELMo model.

        Args:
            options_file (str): Path to the options file containing the model configuration.
            weight_file (str): Path to the weights file containing the trained parameters.
            num_layers (int, optional): The number of layers to use for ELMo. Default is 2.
            dropout (float, optional): The dropout rate for ELMo. Default is 0.5.
        """
        # 加载预训练的 ELMo 模型
        self.elmo = Elmo(options_file=options_file,
                         weight_file=weight_file,
                         num_layers=num_layers,
                         dropout=dropout)

    def embed_sentences(self, sentences: List[List[str]]) -> List[torch.Tensor]:
        """
        Embeds a list of sentences using the pretrained ELMo model.

        Args:
            sentences (List[List[str]]): A list of sentences, where each sentence is a list of words (strings).

        Returns:
            List[torch.Tensor]: A list of tensors, each containing the ELMo embeddings for each sentence.
        """
        # 将句子转换为字符 ID 格式，符合 ELMo 模型的输入要求
        character_ids = batch_to_ids(sentences)

        # 使用 ELMo 模型获取词向量，禁用梯度计算（推理模式）
        with torch.no_grad():
            embeddings = self.elmo(character_ids)

        # 返回 ELMo 嵌入的每个层的输出
        return embeddings['elmo_representations']

    def get_sentence_embedding(self, sentence: List[str]) -> torch.Tensor:
        """
        获取单个句子的 ELMo 嵌入。

        Args:
            sentence (List[str]): 输入句子，以单词的列表形式表示。

        Returns:
            torch.Tensor: 该句子的 ELMo 嵌入，形状为 [seq_len, embedding_dim]。
        """
        # 获取单个句子的嵌入
        sentence_embeddings = self.embed_sentences([sentence])
        
        # 这里只返回第一个句子的词向量，合并多个层的表示
        # 在实际应用中可以根据需求选择使用不同层的嵌入
        return sentence_embeddings[0]

    def get_average_embedding(self, sentence: List[str]) -> torch.Tensor:
        """
        获取一个句子的平均嵌入，即将所有单词的嵌入平均。

        Args:
            sentence (List[str]): 输入句子，以单词的列表形式表示。

        Returns:
            torch.Tensor: 该句子的平均 ELMo 嵌入，形状为 [embedding_dim]。
        """
        # 获取句子的 ELMo 嵌入
        sentence_embedding = self.get_sentence_embedding(sentence)

        # 对所有词的嵌入取平均
        # shape: [seq_len, embedding_dim]
        average_embedding = sentence_embedding.mean(dim=0)

        return average_embedding


def main() -> None:
    """
    主函数：使用 ElmoEmbedder 类进行 ELMo 嵌入。
    """
    # 定义预训练 ELMo 模型的路径
    options_file = 'https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn/2018.12.27.tar.gz'
    weight_file = 'https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn/2018.12.27.tar.gz'

    # 创建 ElmoEmbedder 实例
    elmo_embedder = ElmoEmbedder(options_file=options_file, weight_file=weight_file)

    # 示例句子
    sentences = [
        ['I', 'love', 'machine', 'learning'],
        ['Natural', 'language', 'processing', 'is', 'fun'],
        ['ELMo', 'embeddings', 'are', 'powerful']
    ]

    # 获取所有句子的 ELMo 嵌入
    elmo_embeddings = elmo_embedder.embed_sentences(sentences)

    # 输出每个句子的 ELMo 嵌入的维度
    for i, embedding in enumerate(elmo_embeddings):
        print(f"Sentence {i+1} ELMo embeddings shape: {embedding[0].shape}")

    # 获取句子的平均嵌入
    sentence = ['I', 'love', 'deep', 'learning']
    average_embedding = elmo_embedder.get_average_embedding(sentence)
    print(f"Average embedding for the sentence: {average_embedding.shape}")


if __name__ == "__main__":
    main()
```