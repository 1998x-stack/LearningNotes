# BERT
> WHAT:

一个预训练语言表示模型，广泛应用于自然语言处理（NLP）任务中。它的核心创新在于采用了Transformer架构，并且通过预训练和微调的方式，使得BERT在多个NLP任务中都能取得优异的性能。BERT的一个关键特点是双向性，即它不仅考虑一个词之前的上下文信息（左侧上下文），还考虑其之后的上下文信息（右侧上下文），这与传统的单向模型（如GPT）不同。

> 属性

1. 输入表示：
    * Token Embeddings：将输入的token映射为向量表示。
    * Segment Embeddings：区分不同句子的标识。
    * Position Embeddings：标识token在句子中的位置。
2. 双向自注意力机制: 在预训练时，利用**遮蔽语言模型**（Masked Language Model, MLM），使得BERT能够学习到双向上下文信息。
3. 训练目标：预训练时采用两个目标，分别是**遮蔽语言模型**（Masked Language Model, MLM）和**下一句预测**（Next Sentence Prediction, NSP）。

> bert的变种
- RoBERTa：一个没有NSP任务的BERT变体，它通过更长时间的训练和更大的batch size改进了BERT。
- DistilBERT：一种轻量级的BERT模型，减少了模型的大小，并且保持了较高的性能。
- ALBERT：通过共享参数和因式分解等方法减少BERT的参数量，同时保持较好的性能。
- ELECTRA：使用生成式的预训练方法，替代了BERT的MLM，进行更高效的训练。

