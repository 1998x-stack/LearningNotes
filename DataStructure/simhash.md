# Simhash

> WHAT: 近似相似度计算的算法

> 核心思想：将文本映射为一个固定长度的特征向量，通过计算两个文本的特征向量的汉明距离来判断文本的相似度。

> 步骤

1. 特征哈希：提取文本的特征，如n-gram、关键词等，然后将特征哈希为一个固定长度的二进制哈希值。
2. 加权投影：对特征哈希值进行加权，然后将其投影到一个固定长度的特征向量。
3. 符号映射：将特征向量中的每一位映射为1或0。
4. 汉明距离：计算两个文本的特征向量的汉明距离，通过汉明距离来判断文本的相似度。

> 优点: 高效、内存占用小

> 代码
```python
import hashlib
import numpy as np
from collections import Counter
import re

# Step 1: Tokenization and Feature Extraction
def tokenize(text):
    # Remove non-alphanumeric characters and split into words
    text = re.sub(r'\W+', ' ', text.lower())
    return text.split()

# Step 2: Hashing function to convert a token into a 64-bit binary hash
def hash_feature(feature):
    # Use MD5 or SHA1 to hash the feature (you can use any hash function)
    hash_value = hashlib.md5(feature.encode('utf-8')).hexdigest()
    # Convert the hash value to a binary string (64 bits)
    return bin(int(hash_value, 16))[2:].zfill(64)

# Step 3: Compute SimHash
def compute_simhash(text, num_bits=64):
    # Tokenize the input text
    tokens = tokenize(text)
    
    # Step 3: Accumulate weighted sum of hash values
    # Use a Counter to count token frequencies (this can be replaced with TF-IDF weights if desired)
    token_counts = Counter(tokens)
    
    # Initialize an array to hold the cumulative hash values (with `num_bits` dimensions)
    cumulative_hash = np.zeros(num_bits, dtype=int)
    
    for token, count in token_counts.items():
        # Get the binary hash for each token
        hash_bits = hash_feature(token)
        
        # Convert the hash to a numpy array of 1s and 0s
        hash_array = np.array([int(bit) for bit in hash_bits])
        
        # Add the weighted hash to the cumulative sum (weighted by the token count)
        cumulative_hash += count * (2 * hash_array - 1)  # +1 for bit=1, -1 for bit=0

    # Step 4: Convert cumulative hash sum to final SimHash
    simhash = ''.join('1' if x > 0 else '0' for x in cumulative_hash)
    
    return simhash

# Step 5: Compute Hamming Distance to measure similarity
def hamming_distance(hash1, hash2):
    return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))

# Example Usage
if __name__ == "__main__":
    # Example text documents
    text1 = "I love programming in Python!"
    text2 = "Python programming is awesome."

    # Compute SimHashes for both documents
    simhash1 = compute_simhash(text1)
    simhash2 = compute_simhash(text2)

    # Print the SimHash results
    print(f"SimHash for text 1: {simhash1}")
    print(f"SimHash for text 2: {simhash2}")

    # Compute Hamming distance to check similarity
    dist = hamming_distance(simhash1, simhash2)
    print(f"Hamming Distance: {dist}")
```