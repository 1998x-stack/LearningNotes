# TFIDF
term frequency * inverse document frequency

## Term Frequency
> 表示：某个词在文档中出现的频率
> 公式：TF(t) = (t出现在文档中的次数) / (文档中的词的总数)
> 例子：文档中有100个词，其中“apple”出现了5次，那么TF(apple) = 5 / 100 = 0.05

## Inverse Document Frequency
> 表示：某个词在所有文档中出现的频率
> 公式：IDF(t) = log_e(文档总数 / (含有t的文档总数+1))
> 例子：总共有1000篇文档，其中有100篇文档含有“apple”，那么IDF(apple) = log(1000 / 101) = 2.9

```python

def term_frequency(term, document):
    return document.count(term) / len(document)

def inverse_document_frequency(term, documents):
    count = sum([1 for document in documents if term in document])
    return math.log(len(documents) / (count + 1))
```

1. 优点：简单、易于理解；适用于大多数文本分类问题；
2. 缺点：无法处理一些特殊情况，比如：同义词、拼写错误等；