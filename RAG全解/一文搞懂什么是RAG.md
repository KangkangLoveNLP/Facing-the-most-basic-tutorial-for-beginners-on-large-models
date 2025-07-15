# 一文搞懂RAG是什么

## 1.什么是RAG

RAG（**Retrieval-Augmented Generation**，直译为**检索增强生成**）是一种结合了**检索技术与生成模型**的技术。

其核心流程为：**用户提问** -----> **系统从外部知识库（如文档、数据库）中检索相关文档片段** ----> **将检索结果作为上下文输入大语言模型（LLM）** ----> **LLM基于上下文生成最终答案**

一句话：RAG就是**检索知识，将知识加入prompt提示词，输入到LLM**。

与LLM**直接生成答案**的区别：
**RAG与传统生成模型**的主要区别在于**知识来源和实时性**，RAG模型可以**动态检索外部知识库**，支持**实时更新知识库**，而传统生成模型依赖于训练时的**静态知识，无法更新**。

## 2为什么需要RAG

### 2.1 缓解幻觉

大模型在**特定领域或知识密集型任务**中，特别是**在处理超出其训练数据或需要当前信息的查询**时，会产生 "**幻觉**"。大模型面对不知道的知识时会产生**错误的**答案，引用的文献等是虚构的。检索增强生成（RAG）通过**语义相似性计算**从**外部知识库**中检索相关文档块，从而增强了 LLM。通过引用外部知识，RAG 可**有效减少生成与事实不符内容**的问题。

### 2.2 增强对特定领域的能力

RAG 模型可以**增强特定领域的知识**，例如**医疗领域**、**法律领域**等。通过**知识库**的构建，RAG 模型可以**提供更准确的答案**，从而增强特定领域的知识。而且不需要**额外训练或者微调大模型**

### 2.3 保证数据安全

RAG 模型可以**保证数据安全**，因为**外部知识库**是**私有的**，只有**RAG 模型**才能访问。这样，**RAG 模型**可以**避免数据泄露**，从而保证数据的安全。**不需要将数据上传额外训练LLM**。

## 3.三种RAG范式

### 3.1朴素RAG（Naive RAG）

这是最基础的RAG版本，直接将检索到的信息用于生成。这种方法的优点是简单直接。

### 3.1.1 索引

还是举个例子：假设我们正在开发一个**问答系统**，该系统需要回答有关**历史事件的问题**。我们将使用**朴素RAG方法**来增强一个**生成式AI模型**的能力。

首先，我们需要收集一系列历史文档，例如百度百科上关于各种**历史事件的文章**。然后，我们将这些**文档分割成小块（例如，每个段落或句子作为一个块）**，并使用一个**预训练的嵌入模型（如BERT）**来将每个块转换为一个**向量**。

```python
from sentence_transformers import SentenceTransformer, util

# 加载预训练的BERT模型
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# 示例文档
documents = [
    "The French Revolution was a period of radical social and political upheaval in France and its colonies beginning in 1789.",
    "The American Revolution was a revolution against British rule in the thirteen colonies that formed the United States of America.",
    # 更多文档...
]

# 将文档转换为向量
doc_embeddings = model.encode(documents, convert_to_tensor=True)
```

### 3.1.2 检索

当用户提出一个问题时，例如“法国大革命是什么？”，我们首先将这个问题也**转换为向量**，然后计算**它与索引**中的每个**文档块的相似度**。

```python
# 用户问题
query = "What was the French Revolution?"

# 将问题转换为向量
query_embedding = model.encode(query, convert_to_tensor=True)

# 计算相似度
similarity_scores = util.pytorch_cos_sim(query_embedding, doc_embeddings)[0]

# 获取最相似的文档块索引
top_indices = torch.topk(similarity_scores, k=3).indices

```

### 3.1.3 生成

```python
import torch

# 获取最相似的文档块
retrieved_documents = [documents[i] for i in top_indices]

# 合并问题和检索到的文档块
prompt = f"Answer the question: {query}\nContext: {' '.join(retrieved_documents)}"

# 使用生成式AI模型生成答案
# 假设我们有一个生成式模型generate_answer，它接受提示并返回答案
answer = generate_answer(prompt)

print(answer)

```

这样，我们的**生成式AI模型**就可以**根据检索到的文档块**来回答问题。

## 3.2 高级RAG（Advanced RAG）

**高级RAG（Retrieval-Augmented Generation，检索增强生成）**是在基础RAG的基础上，通过**优化和增强各个阶段的处理步骤**，提升检索质量和生成效果的一种技术。以下是**实现高级RAG的关键步骤和方法**：

还是举个例子：
假设我们有一个包含**多篇关于人工智能和医疗领域的文档集合**，这些**文档可能来自学术论文、新闻报道或行业报告**。

```python
文档1：《人工智能在医疗影像诊断中的应用》
文档2：《AI如何助力医疗数据分析》
文档3：《智能医疗系统：未来医疗的希望》
文档4：《人工智能在药物研发中的作用》
文档5：《医疗AI的伦理和法律问题》
```

### 3.2.1 数据分块和向量化

将每篇文档**分割成固定大小的文本块（例如，每块约200-300字）**，确保每个块**包含完整的语义单元（如一个段落或一个主题）**。

使用一个高效的**嵌入模型（如E5 Embeddings）**将每个文本块转换为**向量**。

```python
文档1的块1 → 向量A
文档1的块2 → 向量B
文档2的块1 → 向量C
...
文档5的块N → 向量Z
```

### 3.2.2 索引优化

将所有向量存储到**一个高效的向量数据库（如FAISS或Milvus）**中，并构建索引以便快速检索。

### 3.2.3 用户查询处理

假如用户输入查询：“ **人工智能在医疗影像诊断中的应用** ”。

**查询转换（Query Transformations）**：使用一个**预训练语言模型（如GPT-3.5）对用户查询进行改写**，使其**更具体或更通用**。例如：

```python
原查询：人工智能在医疗影像诊断中的应用
转换后：AI技术如何提高医学影像诊断的准确性？
```

**还可以进行Step-back prompting**：生成更通用的查询，以便检索更**广泛**的上下文

```python
更通用的查询：人工智能在医疗领域的应用
```

### 3.2.4 检索增强

使用**向量检索算法（如FAISS的近似最近邻搜索）**找到与**查询向量最相似**的文档块。**结合混合搜索（Fusion Retrieval）**，同时使用**TF-IDF对关键词进行匹配，并通过Reciprocal Rank Fusion对结果进行重排序。**

```python
检索到的块1：文档1的块1（相关性0.95）
检索到的块2：文档2的块1（相关性0.85）
检索到的块3：文档4的块2（相关性0.75）
```

### 3.2.5 检索后处理

1. **上下文重排序（Context Reordering）**

```python
文档1的块1 → 文档2的块1 → 文档4的块2
```

2.**上下文筛选与压缩（Context Filtering & Compression）**

去除与**查询相关性较低**的内容，只保留**关键信息，避免上下文过长**。

### 3.2.6 模型生成

将筛选后的**上下文输入**到生成模型（如GPT-4）中，生成最终答案

```python
输入到生成模型：
“文档1的块1：人工智能在医疗影像诊断中的应用……
文档2的块1：AI如何助力医疗数据分析……
文档4的块2：人工智能在药物研发中的作用……”

生成的答案：
“人工智能在医疗诊断影像中主要通过深度学习算法分析医学影像，提高诊断的准确性和效率。同时，AI技术也广泛应用于医疗数据分析和药物研发，为医疗行业带来了诸多变革。”
```
