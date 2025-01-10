# Hash
> What: 通过某种算法，将任意长度的输入转换为固定长度的输出，这个输出就是哈希值。

> 特点：
1. 哈希值是固定长度的。
2. 不同的输入可能会产生相同的哈希值，这种现象称为哈希碰撞。
3. 哈希值是不可逆的，即无法通过哈希值推导出原始输入。
4. 高效、唯一

> 哈希 🆚 加密：
1. 哈希是不可逆的，加密是可逆的。

> 哈希碰撞：两个不同的输入，通过哈希算法得到相同的哈希值。

## 哈希表
> What: 通过哈希函数将键映射到哈希表中的位置，以加快查找速度。

> 实现：
1. 数组 + 哈希函数
2. 哈希冲突解决方法：链地址法、开放地址法

> 哈希函数：

* 除法法：通过对哈希表的大小取模来确定哈希值。例如，h(k) = k % m，其中 k 是键，m 是哈希表的大小。
* 乘法法：通过对一个常数进行乘法运算来生成哈希值。例如，h(k) = floor(m * (k * A % 1))，其中 A 是常数，m 是哈希表的大小。
* MurmurHash 和 SHA-256：更复杂的哈希函数，常用于处理复杂的数据类型或在分布式系统中应用。

> 哈希冲突：

1. 链地址法：将哈希值相同的键放在同一个链表中。
2. 开放地址法：当哈希冲突发生时，通过探测方法寻找下一个空槽。
    * 线性探测：逐个探测下一个空槽。
    * 二次探测：探测下一个空槽的位置是二次函数。
    * 双重哈希：使用第二个哈希函数来计算下一个空槽的位置。

> 哈希表的操作：

1. 插入：将键值对插入哈希表中。
2. 删除：删除哈希表中的键值对。
3. 查找：查找哈希表中的键值对。

> 代码：

```python
class HashTable:
    def __init__(self):
        self.size = 1000
        self.table = [[] for _ in range(self.size)]

    def hash(self, key):
        return key % self.size

    def insert(self, key, value):
        h = self.hash(key)
        for i, (k, v) in enumerate(self.table[h]):
            if k == key:
                self.table[h][i] = (key, value)
                return
        self.table[h].append((key, value))

    def delete(self, key):
        h = self.hash(key)
        for i, (k, v) in enumerate(self.table[h]):
            if k == key:
                self.table[h].pop(i)
                return

    def get(self, key):
        h = self.hash(key)
        for k, v in self.table[h]:
            if k == key:
                return v
        return None
```