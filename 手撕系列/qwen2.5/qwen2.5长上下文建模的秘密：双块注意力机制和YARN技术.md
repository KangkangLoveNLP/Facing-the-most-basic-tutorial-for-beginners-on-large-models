# 长上下文建模的秘密

**传统 Transformer：** 受限于固定长度的上下文窗口，通常只能处理较短的文本序列。

**Qwen2.5：** 采用**双块注意力机制（DCA）**和 **YARN 技术**，**能够处理长达 128K tokens 的上下文。**此外，Qwen2.5 使用**动态分辨率处理和绝对时间编码**，使其能够处理长视频和长文本。

## 1.双块注意力机制（DCA）

**Qwen2.5的双块注意力机制（Dual Chunk Attention，DCA）** 是一种用于提升长文本处理效率和性能的技术。其核心原理是将**长序列分割成多个可管理的块（chunk）**，并通过特定的注意力机制设计，实现对长序列的有效处理。

### 1.2 长文本分块

#### 分块的基本原则

**块的长度：**长文本序列被分割成固定长度的块，例如32,768个token

**块的数量：**根据输入序列的长度，整个序列被划分为多个块，每个块的长度不超过预训练的最大长度

### 1.3块内的注意力机制

#### 1.3.1 块内相对位置计算

**块内相对位置**是指在同一个块（chunk）内，任意两个token之间的相对距离。

假设我们有一个块，包含长度为 $C$ 的token序列，其中 $C$ 是块的固定长度（例如32,768个token）。对于块内的任意两个token，其相对位置计算公式为：
在块内，每个token与其他token之间的相对位置是直接计算的。对于块内的任意两个token，其相对位置计算公式为：   $$
   \text{Relative Position}_{\text{Intra}} = \text{Key Index} - \text{Query Index}
   $$
   其中，Key Index和Query Index是块内的索引。
    - **Key Index**：键（Key）token在块内的索引位置。
    - **Query Index**：查询（Query）token在块内的索引位置。

##### 1.3.2具体计算过程
假设块的长度为 \( C = 8 \)，块内的token索引从0到7。我们以两个token为例，计算它们的相对位置：

**示例** 1：
- Query Index = 2
- Key Index = 5

相对位置计算为：
$$
\text{Relative Position}_{\text{Intra}} = 5 - 2 = 3
$$

这意味着，对于Query Index为2的token，Key Index为5的token在它右边3个位置。

#### 示例2：
- Query Index = 6
- Key Index = 3

相对位置计算为：
$$
\text{Relative Position}_{\text{Intra}} = 3 - 6 = -3
$$

这意味着，对于Query Index为6的token，Key Index为3的token在它左边3个位置。

#### 1.3.3. 相对位置的范围
在块内，相对位置的范围是从 \(-C+1\) 到 \(C-1\)。例如，对于长度为8的块，相对位置的范围是从 \(-7\) 到 \(7\)。

- **正数**：表示Key在Query的右边。
- **负数**：表示Key在Query的左边。
- **零**：表示Key和Query是同一个token。

#### 1.3.4 块内相对位置的作用
在自注意力机制中，相对位置信息通常用于计算注意力权重。具体来说，相对位置信息可以作为额外的特征输入到注意力机制中，帮助模型更好地捕捉局部依赖关系。例如，Transformer-XL和Qwen2.5等模型通过相对位置编码（Relative Position Embedding）来增强自注意力机制。
