共现矩阵：反映了不同元素（例如单词或像素）之间的共同出现关系。
步骤：
1. 选择一个窗口大小，例如3
2. 从左到右扫描文本，每次取3个单词，构成一个窗口
3. 在窗口中的单词两两配对，构成一个共现对
4. 统计每个共现对出现的次数，构成共现矩阵


```python
def co_occurrence(text, window_size=3):
    words = text.split()
    co_occurrence_matrix = {}
    for i in range(len(words) - window_size + 1):
        window = words[i:i+window_size]
        for j in range(window_size):
            for k in range(j+1, window_size):
                word1, word2 = window[j], window[k]
                if word1 not in co_occurrence_matrix:
                    co_occurrence_matrix[word1] = {}
                if word2 not in co_occurrence_matrix[word1]:
                    co_occurrence_matrix[word1][word2] = 0
                co_occurrence_matrix[word1][word2] += 1
                if word2 not in co_occurrence_matrix:
                    co_occurrence_matrix[word2] = {}
                if word1 not in co_occurrence_matrix[word2]:
                    co_occurrence_matrix[word2][word1] = 0
                co_occurrence_matrix[word2][word1] += 1
    return co_occurrence_matrix
```