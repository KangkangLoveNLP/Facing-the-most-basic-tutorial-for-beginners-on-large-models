FM模型（Factorization Machine，因子分解机）是一种**基于矩阵分解的机器学习模型**，由Steffen Rendle在2010年提出，专门用于解决**稀疏数据下的特征组合问题**。它结合了**线性模型**和**矩阵分解**的优点，能够高效地自动学习特征间的**二阶交互作用**，尤其适用于推荐系统、广告点击率（CTR）预估等场景。

---

### **核心思想**
FM模型的核心是通过**隐向量（latent vector）**来表示每个特征，并通过隐向量的内积来建模特征之间的交互关系，从而解决稀疏数据下特征组合难以直接学习的问题。

---

### **数学形式**
对于一个包含 \( n \) 个特征的样本 \( \mathbf{x} = [x_1, x_2, \dots, x_n] \)，FM模型的预测公式为：

\[
\hat{y}(\mathbf{x}) = w_0 + \sum_{i=1}^n w_i x_i + \sum_{i=1}^n \sum_{j=i+1}^n \langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j
\]

- **\( w_0 \)**: 全局偏置（截距项）。
- **\( w_i \)**: 第 \( i \) 个特征的权重（一阶项）。
- **\( \mathbf{v}_i \in \mathbb{R}^k \)**: 第 \( i \) 个特征的**隐向量**（\( k \) 是隐向量的维度，需手动设置）。
- **\( \langle \mathbf{v}_i, \mathbf{v}_j \rangle \)**: 隐向量的内积，表示特征 \( i \) 和 \( j \) 的二阶交互强度。

---

### **关键优势**
1. **解决稀疏数据问题**  
   在稀疏场景（如推荐系统），某些特征组合（如“用户=张三”和“商品=手机”）可能从未同时出现过，导致传统多项式模型无法学习。FM通过隐向量的内积泛化未观测到的组合。

2. **计算复杂度优化**  
   原始二阶项的计算复杂度为 \( O(n^2) \)，但通过数学变形可降至 **\( O(kn) \)**（\( k \ll n \)）：
   \[
   \sum_{i=1}^n \sum_{j=i+1}^n \langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j = \frac{1}{2} \sum_{f=1}^k \left[ \left( \sum_{i=1}^n v_{i,f} x_i \right)^2 - \sum_{i=1}^n v_{i,f}^2 x_i^2 \right]
   \]

3. **通用性**  
   支持任意实数特征（包括类别型特征的one-hot编码），可扩展至高阶交互（如FFM、DeepFM等）。

---

### **应用场景**
- **推荐系统**：预测用户对物品的评分或点击率（如Netflix、电商推荐）。
- **广告CTR预估**：预测用户点击广告的概率（如Google Ads、Facebook Ads）。
- **分类/回归任务**：任何需要特征组合的场景。

---

### **FM vs. 其他模型**
| **模型**       | **特点**                                                                 |
|-----------------|--------------------------------------------------------------------------|
| **线性模型（如LR）** | 无法自动学习特征交互，需人工构造组合特征。                                |
| **多项式模型**   | 直接建模高阶交互，但稀疏下参数过多，难以训练。                            |
| **FM**          | 通过隐向量高效学习二阶交互，泛化能力强。                                  |
| **深度学习（如DeepFM）** | 结合FM和神经网络，可捕捉高阶非线性关系，但需要更多数据。                   |

---

### **代码示例（Python + LibFM）**
```python
from pyfm import pylibfm
from sklearn.feature_extraction import DictVectorizer

# 构造稀疏特征（如用户ID、物品ID）
train_data = [
    {"user": "1", "item": "5", "age": 19},
    {"user": "2", "item": "43", "age": 33},
    # ...
]
y = [1, 0, ...]  # 标签（如是否点击）

# 特征向量化
v = DictVectorizer()
X = v.fit_transform(train_data)

# 训练FM模型
fm = pylibfm.FM(num_factors=10, num_iter=100, verbose=True)
fm.fit(X, y)
```

---

### **总结**
FM模型通过隐向量巧妙解决了稀疏数据下的特征组合问题，是推荐系统和CTR预估领域的经典基线模型。后续研究（如FFM、DeepFM、xDeepFM）均在其基础上扩展。