# 模型微调（Fine-tuning）

## 1.全量微调（Full Fine-tuning）

全量微调（Full Fine-tuning）是一种对预训练模型的所有参数进行更新和优化的微调方法。它是最直接的微调方式，目的是让模型在特定任务或数据集上达到最佳性能。

### 1.1 全量微调核心思想

全量微调的核心是利用预训练模型在大规模数据上学习到的通用特征，然后通过在特定任务的数据集上继续训练，调整模型的所有参数，使其更好地适应新任务。这种方法假设预训练模型的每一层和每个参数都对新任务有潜在价值，因此需要对它们进行重新优化。

### 2.2 **全量微调的优势**

- **性能潜力最大**：由于所有参数都可以调整，模型能够最大程度地适应新任务，理论上可以达到最高的性能。
- **灵活性高**：适用于各种任务类型，无论是文本分类、问答系统，还是图像识别、目标检测等。
- **无需额外设计**：不需要像参数高效微调那样设计适配器或低秩分解模块，直接对整个模型进行训练。

### 2.3 **全量微调的缺点**

- **计算资源需求高**：需要更新模型的所有参数，计算量大，训练时间长，对硬件要求高。
- **容易过拟合**：如果新任务的数据量有限，模型可能会过度拟合训练数据，导致泛化能力下降。
- **存储成本高**：需要保存整个模型的参数，占用大量存储空间。

### 2.4**全量微调的适用场景**

- **任务复杂且数据量充足**：当新任务的复杂度较高，且有足够的标注数据时，全量微调可以充分发挥模型的潜力。
- **目标是最高性能**：如果对模型性能要求极高，且有足够的资源支持，全量微调是最佳选择。
- **预训练模型与新任务相关性较低**：当预训练任务和新任务差异较大时，全量微调可以更彻底地调整模型。

---

## 2. **全量微调的步骤**

以下是全量微调的一般步骤：

### （1）**加载预训练模型**

从开源平台（如Hugging Face、TensorFlow Hub）加载预训练模型及其权重。

### （2）**准备任务数据**

对新任务的数据进行预处理，包括数据清洗、分词（NLP）、归一化（CV）等，并将其格式化为模型可接受的形式。

### （3）**设置训练环境**

选择合适的优化器（如Adam、AdamW）、损失函数（如交叉熵损失、均方误差）和学习率调度策略。

### （4）**训练模型**

在新任务的数据集上对模型的所有参数进行训练。通常需要设置多个epoch，并在训练过程中监控验证集的性能。

### （5）**评估与优化**

通过验证集评估模型性能，根据需要调整学习率、正则化参数等，以防止过拟合。

### （6）**保存模型**

训练完成后，保存模型权重，用于后续的推理或部署。

---

## 3. **全量微调的注意事项**

- **学习率设置**：由于预训练模型的参数已经经过优化，建议使用较低的学习率（如1e-5或1e-6），避免破坏预训练模型的特征。
- **数据增强**：通过数据增强技术（如文本的同义词替换、图像的随机裁剪）扩充数据集，提升模型的泛化能力。
- **早停机制**：在验证集性能不再提升时停止训练，防止过拟合。
- **正则化**：使用Dropout、L2正则化等技术，减少模型复杂度。

---

## 4. **全量微调的案例**

假设我们使用一个预训练的Transformer模型（如BERT）进行文本分类任务，以下是代码示例（基于PyTorch）：

```python
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 准备数据
texts = ["样本文本1", "样本文本2", ...]  # 文本数据
labels = [0, 1, ...]  # 标签
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
encoded_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

# 划分训练集和验证集
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2)
train_encodings = tokenizer(train_texts, padding=True, truncation=True, return_tensors="pt")
val_encodings = tokenizer(val_texts, padding=True, truncation=True, return_tensors="pt")

# 创建数据加载器
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = TextDataset(train_encodings, train_labels)
val_dataset = TextDataset(val_encodings, val_labels)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# 加载预训练模型
model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
optimizer = AdamW(model.parameters(), lr=1e-5)

# 训练模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(3):  # 训练3个epoch
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")

    model.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(batch["labels"].cpu().numpy())
    print(f"Validation Accuracy: {accuracy_score(val_labels, val_preds)}")

# 保存模型
torch.save(model.state_dict(), "fine_tuned_model.pth")
```

---

### 8. **总结**

全量微调是一种强大的微调方法，适用于对性能要求极高且资源充足的情况。它通过调整预训练模型的所有参数，能够最大程度地适应新任务，但也需要注意过拟合和计算成本等问题。

## 5 部分参数微调：对全量微调的调整

部分参数微调（Partial Fine-Tuning）是一种在预训练模型基础上，仅调整部分参数（如顶层或特定中间层参数）的微调方法。这种方法旨在保留预训练模型的通用特征，同时以较低的成本适应新任务。

### **定义**

部分参数微调通过冻结预训练模型的大部分参数，仅对部分层（通常是靠近输出的顶层）或特定模块进行训练。这种方法适用于新任务与预训练任务相关性较高，或者希望保留预训练模型泛化能力的场景。

### **优势**

1. **计算成本低**：仅更新少量参数，训练速度快，计算资源需求低。
2. **泛化能力强**：保留预训练模型的底层通用特征，避免过度修改，减少过拟合风险。
3. **多任务共享**：可使用同一底层预训练模型，针对不同任务微调顶层参数，实现知识共享。

### **缺点**

1. **适应性有限**：对于复杂或高度专业化的任务，仅调整顶层可能无法充分捕捉任务特有的细微差异。
2. **性能上限**：相比全量微调，部分参数微调的性能提升可能有限。

### **适用场景**

- **资源有限**：计算资源或数据量有限时，部分参数微调是一种高效的选择。
- **任务相关性高**：新任务与预训练任务较为相关，底层特征可复用。
- **多任务学习**：需要同时处理多个相关任务时，可共享底层模型。

### **技巧**

1. **冻结策略**：冻结底层参数，仅训练顶层或特定中间层。
2. **结合其他方法**：可以与适配器（Adapter）或低秩分解（LoRA）等参数高效微调方法结合使用。
3. **任务导向**：根据任务复杂度选择合适的微调层，复杂任务可适当增加微调层数。

部分参数微调是一种高效且灵活的微调策略，特别适合在资源受限或任务相关性较高的场景中使用。
