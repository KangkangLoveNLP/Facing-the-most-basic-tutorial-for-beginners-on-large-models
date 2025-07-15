# 从零开始的RAG库的搭建

什么是RAG？RAG是什么？RAG是**Retrieval Augmented Generation**的缩写，即**检索增强生成**。RAG是一种**信息检索**和**自然语言处理**的结合，它通过将文档和用户问题进行匹配，从而实现**信息检索**和**自然语言处理**的结合。

当我们输入一个问题，问题的答案会从文档中检索出来，然后将答案和问题一起输入到大模型中生成答案。

## 一、基本结构

要实现一个最简单的RAG，我们首先要了解**RAG的基础结构**。

- 1.**向量化结构**：能够将文档片段向量化
- 2.**文档切分和加载模块**：能够将文档切分为多个片段，并加载到内存中
- 3.**向量检索模块**：能够将向量检索到最相似的向量，并返回对应的文档片段
- 4.**向量数据库模块**：能够将向量存储到数据库中，并查询到最相似的向量
- 5.**问答模块**：能够将用户问题向量化，并查询到最相似的向量，并返回对应的答案

## 二、向量化结构

### 1.为什么需要向量化？

- 1.**高效检索**：将文本向量化后，可以利用**向量之间的距离**（如余弦距离）来衡量**文本的相似度**，从而快速**检索出与用户查询最相关的文档片段**
- 2.**语义理解**：向量化能够捕捉文本的语义信息，使得模型能够理解文本的含义，而不仅仅是基于关键词的匹配。

### 2.向量化的一般方法

- 1.**传统稀疏向量化**（Sparse Embedding）：映射成一个高维向量，维度通常与词汇表大小相同。向量的大部分元素为0，非零值表示特定单词在文档中的重要性。典型模型包括TF-IDF和BM25，适合关键词匹配任务。
- 2.**密集向量化**（Dense Embedding）：映射到一个相对低维的向量，所有维度都非零。典型模型包括基于BERT的模型（如BGE-v1.5）和Sentence Transformers，这些模型能够捕捉语义信息，适用于语义搜索任务

### 3. 代码实现

```python
from sentence_transformers import SentenceTransformer
'''
SentenceTransformer 是一个非常流行的 Python 库，用于将文本（句子或段落）转换为密集的向量表示（嵌入向量）。它是基于 Hugging Face 的 transformers 库和 PyTorch 的，提供了简单易用的接口来加载预训练的模型并生成文本嵌入。
'''
#下面实现一个基本类，然后继承这个类实现一个向量化的类
class BaseEmbeddings:
    """
    Base class for embeddings
    """
    def __init__(self, path: str, is_api: bool) -> None:
        self.path = path
        self.is_api = is_api
    # 下面这个函数是得到向量化的函数
    def get_embedding(self, text: str, model: str) -> List[float]:
        raise NotImplementedError
    #下面这个函数大户是计算余弦相似度
    @classmethod
    def cosine_similarity(cls, vector1: List[float], vector2: List[float]) -> float:
        """
        calculate cosine similarity between two vectors
        """
        dot_product = np.dot(vector1, vector2)
        magnitude = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        if not magnitude:
            return 0
        return dot_product / magnitude

#下面这个类是实现一个向量化的类
class MyEmbedding(BaseEmbeddings):
    """
    class for My embeddings
    """
    def __init__(self, path: str = '', is_api: bool = False) -> None:
        super().__init__(path, is_api)
        self.path = path
        self.model = SentenceTransformer(path)#这里是加载一个嵌入模型，可以是bert的模型

    def get_embedding(self, text: str) -> List[float]:
        embeddings  = self.model.encode(text)
        return embeddings
```

## 三、文档切分和加载模块

### 1. 为什么需要文档切分？

- 1.**提高检索效率**：将文档切分为多个片段，可以减少检索的文档数量，提高检索效率。
- 2.**提高检索准确性**：将文档切分为多个片段，可以提高检索的准确性，因为每个片段都包含一些与用户查询最相关的信息。

### 2.如何切分

- 1.**基于字符长度**：将文档切分为固定长度的片段。
- 2.**基于句子长度**：将文档切分为句子，然后根据句子长度进行切分。
- 3.**滑动窗口切分**：允许相邻块之间部分重叠，减少信息断裂，缓解上下文不连贯问题，提升检索相关性，但是会增加计算和内存开销
除此之外还有许多切分方法，这里不在赘述

### 3. 文档切分和加载模块代码实现

```python

def get_chunk(cls, text: str, max_token_len: int = 600, cover_content: int = 150):
    chunk_text = []

    curr_len = 0
    curr_chunk = ''

    lines = text.split('\n')  # 假设以换行符分割文本为行

    for line in lines:
        line = line.replace(' ', '')
        line_len = len(enc.encode(line))
        if line_len > max_token_len:
            print('warning line_len = ', line_len)
        if curr_len + line_len <= max_token_len:
            curr_chunk += line
            curr_chunk += '\n'
            curr_len += line_len
            curr_len += 1
        else:
            chunk_text.append(curr_chunk)
            curr_chunk = curr_chunk[-cover_content:]+line
            curr_len = line_len + cover_content

    if curr_chunk:
        chunk_text.append(curr_chunk)

    return chunk_text

```

## 四、数据库和向量检索模块

数据库和向量检索模块需要实现下面四个功能：

- persist：数据库持久化，本地保存
- load_vector：从本地加载数据库
- get_vector：获得文档的向量表示
- query：根据问题检索相关的文档片段

### 1.数据库和向量检索模块代码实现

```python

class VectorStore:
    def __init__(self, document: List[str] = ['']) -> None:
        self.document = document

    def get_vector(self, EmbeddingModel: BaseEmbeddings) -> List[List[float]]:
        
        self.vectors = []
        for doc in tqdm(self.document, desc="Calculating embeddings"):
            self.vectors.append(EmbeddingModel.get_embedding(doc).tolist())
        return self.vectors

    def persist(self, path: str = 'storage'):
        if not os.path.exists(path):
            os.makedirs(path)
        with open(f"{path}/doecment.json", 'w', encoding='utf-8') as f:
            json.dump(self.document, f, ensure_ascii=False)
        if self.vectors:
            with open(f"{path}/vectors.json", 'w', encoding='utf-8') as f:
                json.dump(self.vectors, f)

    def load_vector(self, path: str = 'storage'):
        with open(f"{path}/vectors.json", 'r', encoding='utf-8') as f:
            self.vectors = json.load(f)
        with open(f"{path}/doecment.json", 'r', encoding='utf-8') as f:
            self.document = json.load(f)

    def get_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        return BaseEmbeddings.cosine_similarity(vector1, vector2)

    def query(self, query: str, EmbeddingModel: BaseEmbeddings, k: int = 1) -> List[str]:
        query_vector = EmbeddingModel.get_embedding(query)
        result = np.array([self.get_similarity(query_vector, vector)
                          for vector in self.vectors])
        return np.array(self.document)[result.argsort()[-k:][::-1]].tolist()

```

## 五、问答模块

问答模块其实非常简答，将检索到的内容和用户的问题进行拼接，然后使用LLM进行回答即可。

```python
class BaseModel:
    def __init__(self, path: str = '') -> None:
        self.path = path

    def chat(self, prompt: str, history: List[dict], content: str) -> str:
        pass

    def load_model(self):
        pass


class InternLMChat(BaseModel):
    def __init__(self, path: str = '') -> None:
        super().__init__(path)
        self.load_model()

    def chat(self, prompt: str, history: List = [], content: str='') -> str:
        prompt = PROMPT_TEMPLATE['InternLM_PROMPT_TEMPALTE'].format(question=prompt, context=content)
        response, history = self.model.chat(self.tokenizer, prompt, history)
        return response


    def load_model(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.path, torch_dtype=torch.float16, trust_remote_code=True).cuda()

```
