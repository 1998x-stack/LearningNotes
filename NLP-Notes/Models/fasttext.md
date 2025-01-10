# FastText
word2vec的扩展，采用ngram，更好改进词向量模型；
1. 核心思想：考虑词内的子词信息，处理低频词效果更好；
2. 假设：
    * 语义相似的词具有相似的上下文。
    * 通过上下文可以推测出一个词的含义。
3. 词内子词表示：apple采用3-gram表示：`<ap, app, ppl, ple, le>`；
4. 计算其所有子词的向量表示。然后，通过对所有子词的向量进行 聚合（加和或平均），得到该词的最终表示
5. FastText 与 Word2Vec 的区别
    * Word2Vec 只考虑词作为整体单位来学习词向量。
    * FastText 将词拆分为子词（n-gram），使得每个词的表示是由其子词表示组成。这使得 FastText 能够有效地处理未见过的词。

```python
!pip install fasttext
import fasttext

# 训练词向量模型
model = fasttext.train_unsupervised('data.txt', model='skipgram')

# 查看训练的词向量
word_vector = model.get_word_vector('machine')
print(word_vector)

# 保存模型
model.save_model("fasttext_model.bin")

# 加载模型
loaded_model = fasttext.load_model("fasttext_model.bin")

# 获取词向量
word_vector = loaded_model.get_word_vector('machine')
print(word_vector)

# 查找与 "machine" 相似的词
similar_words = model.get_nearest_neighbors('machine')

for similarity, word in similar_words:
    print(f"{word}: {similarity}")

```

```plaintext
__label__technology I love machine learning
__label__sports Soccer is great
__label__technology FastText is a tool for NLP```
```
```python
import fasttext

# 训练文本分类模型
classifier = fasttext.train_supervised('train_data.txt')

# 测试模型
test_text = "I enjoy playing football"
prediction = classifier.predict(test_text)

print(f"Predicted label: {prediction[0]}, with probability: {prediction[1]}")

# 使用调整后的参数进行训练
classifier = fasttext.train_supervised('train_data.txt', lr=0.5, epoch=25, dim=300, wordNgrams=2)
# 测试模型的准确率
result = classifier.test('test_data.txt')
print(f"Accuracy: {result[1]}")  # 输出准确率
# 保存模型
classifier.save_model('text_classification_model.bin')

# 加载模型
loaded_classifier = fasttext.load_model('text_classification_model.bin')

# 使用加载的模型进行预测
prediction = loaded_classifier.predict("I love playing basketball")
print(f"Predicted label: {prediction[0]}, with probability: {prediction[1]}")
```
