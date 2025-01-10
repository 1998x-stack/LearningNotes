# Bloom Filter
> WHAT： 

- 布隆过滤器是一个数据结构，用于检测一个元素是否在一个集合中。
- 它是一个空间效率的概率数据结构，通常用于检查一个元素是否是一个集合的成员。
- 它可能会误报，但不会漏报。
- 它是一个二进制向量和一组哈希函数的集合。

> 基本原理：

多个哈希函数和一个位向量。
* 当一个元素被加入集合时，通过多个哈希函数将这个元素映射成位向量中的多个点，把它们置为1。
* 检查一个元素是否在集合中时，同样地，将这个元素通过多个哈希函数映射成位向量中的多个点，只要这些点有一个为0，则被检元素一定不在集合中；若全部为1，则被检元素很可能在集合中。

> 操作流程：

1. 初始化：创建一个m位的位向量，和k个哈希函数。
2. 添加元素：将元素通过k个哈希函数映射成位向量中的k个点，将这k个点置为1。
3. 查询元素：将元素通过k个哈希函数映射成位向量中的k个点，只要这k个点有一个为0，则被检元素一定不在集合中；若全部为1，则被检元素很可能在集合中。
4. 删除元素：由于删除元素会影响其他元素的判断，所以布隆过滤器一般不支持删除操作。

> 优点：

1. 空间效率高：布隆过滤器只需要一个位向量和多个哈希函数。
2. 查询效率高：布隆过滤器只需要k次哈希计算，就能判断一个元素是否在集合中。
3. 误报率低：布隆过滤器不会漏报，只会误报。

> 数学分析：

误判率：当一个元素被检测时，如果这个元素不在集合中，那么这个元素被误判为在集合中的概率是多少？
公式：
$$
(1 - (1 - \frac{1}{m})^{kn})^k
$$
其中，m是位向量的长度，n是集合中元素的个数，k是哈希函数的个数。

理想哈希函数的个数：
$$
k = \frac{m}{n} \ln 2
$$

> 代码：
    
```python
import hashlib
import math
import os

class BloomFilter:
    def __init__(self, capacity: int, error_rate: float):
        """
        Initialize a Bloom Filter.
        
        :param capacity: The expected number of elements to be added to the filter.
        :param error_rate: The desired false positive probability (between 0 and 1).
        """
        # Calculate the size of the bit array (m) and the number of hash functions (k)
        self.capacity = capacity
        self.error_rate = error_rate
        self.bit_array_size = self._get_optimal_bit_array_size(capacity, error_rate)
        self.num_hash_functions = self._get_optimal_num_hash_functions(self.bit_array_size, capacity)
        
        # Initialize the bit array with all 0's
        self.bit_array = [0] * self.bit_array_size

    def _get_optimal_bit_array_size(self, n: int, p: float) -> int:
        """
        Calculate the optimal size of the bit array (m) using the desired error rate (p) and capacity (n).
        
        :param n: The expected number of elements to insert.
        :param p: The desired false positive probability.
        :return: The optimal size of the bit array (m).
        """
        m = -(n * math.log(p)) / (math.log(2) ** 2)
        return int(m)

    def _get_optimal_num_hash_functions(self, m: int, n: int) -> int:
        """
        Calculate the optimal number of hash functions (k).
        
        :param m: The size of the bit array.
        :param n: The expected number of elements to insert.
        :return: The optimal number of hash functions (k).
        """
        k = (m / n) * math.log(2)
        return int(k)

    def _hash(self, item: str, seed: int) -> int:
        """
        Generate a hash value for a given item using a seed for a specific hash function.
        
        :param item: The item to be hashed.
        :param seed: The seed value for the hash function.
        :return: A hash value between 0 and the size of the bit array.
        """
        return int(hashlib.md5((str(seed) + item).encode('utf-8')).hexdigest(), 16) % self.bit_array_size)

    def add(self, item: str):
        """
        Add an item to the Bloom filter.
        
        :param item: The item to be added.
        """
        for seed in range(self.num_hash_functions):
            hash_value = self._hash(item, seed)
            self.bit_array[hash_value] = 1

    def contains(self, item: str) -> bool:
        """
        Check whether an item is in the Bloom filter.
        
        :param item: The item to be checked.
        :return: True if the item is possibly in the Bloom filter, False if the item is definitely not in the filter.
        """
        for seed in range(self.num_hash_functions):
            hash_value = self._hash(item, seed)
            if self.bit_array[hash_value] == 0:
                return False
        return True

# Example usage:
if __name__ == "__main__":
    # Create a Bloom filter with an expected capacity of 1000 elements and an error rate of 1%
    bloom_filter = BloomFilter(capacity=1000, error_rate=0.01)

    # Add elements to the filter
    bloom_filter.add("apple")
    bloom_filter.add("banana")
    bloom_filter.add("cherry")

    # Check if elements are in the filter
    print("apple in Bloom Filter:", bloom_filter.contains("apple"))  # Should print: True
    print("banana in Bloom Filter:", bloom_filter.contains("banana"))  # Should print: True
    print("grape in Bloom Filter:", bloom_filter.contains("grape"))  # Should print: False
```