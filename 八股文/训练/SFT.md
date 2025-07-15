# 大模型的微调 SFT、 指令微调

## 为什么需要指令微调

大模型所有的能力都是在预训练阶段获得的，但是大模型还不知道如何表达自己的知识，预训练完成后的大模型只会根据提供的上文续写下文，指令微调能让大模型学会如何表达自己的知识回答人提出的问题。

## Chat Tempalte对话模板

预训练和指令微调只在训练数据上有本质区别，开源的大模型都会有预训练版本和指令微调版本，每个厂都会有自己的对话模板

以llama3.1为例子

```python
[{"role": "system", "content": "You are a helpful assistant."},
 {"role": "user", "content": "天空为什么是蓝色的？"},
 {"role": "assistant", "content": "这是由于光的散射引起的。"}]
```

实际输入到大模型的对话模板为

```python
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>
天空为什么是蓝色的？<|eot_id|><|start_header_id|>assistant<|end_header_id|>
这是由于光的散射引起的。<|eot_id|>
```

将角色放入<|start_header_id|> <|end_header_id|>中，将内容用<|eot_id|>分隔开。
"system"表示系统设定
"user"表示用户输入
"assistant"表示模型输出

当我们要推理时：

```python
[{"role": "system", "content": "You are a helpful assistant."},
 {"role": "user", "content": "天空为什么是蓝色的？"}]
```

对话模板

```python
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful assistant.
<|eot_id|><|start_header_id|>user<|end_header_id|>
天空为什么是蓝色的？
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```

大模型输出

```python
天空为什么是蓝色的？<|eot_id|>
```

## completions only

我们在训练过程中为了让大模型把注意力集中在如何回答问题上，我们只对答案计算loss。

```python
天空为什么是蓝色的？<|eot_id|>
```

我们可以使用一个**loss mask** 来解决这个问题.

对不用计算loss的token标记为0，对需要计算loss的token标记为1。

最后我们在计算loss时对每个位置的loss乘以loss mask，就可以忽略不需要计算的token的loss。

## NEEFTune(Noisy Embedding Fine-Tuning)

对Embedding进行扰动，进行数据增强

## SFT代码实现

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = r'D:\work\models\Meta-Llama-3.1-8B-Instruct'

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_path).to("cuda")
optimizer = torch.optim.AdamW(model.parameters())

dialog = [{"role": "system", "content": "You are a helpful assistant."},
          {"role": "user", "content": "天空为什么是蓝色的？"},
          {"role": "assistant", "content": "这是由于光的散射引起的。"}]

input = tokenizer.apply_chat_template(dialog, return_tensors="pt")
input = {k: v.to("cuda") for k, v in input.items()}

#设置labels和inputs一致
input["labels"] = input["input_ids"].clone()

output = model(**input)

#获取模型的loss
loss = output.loss
loss.backward()
optimizer.step()
optimizer.zero_grad()

#保存模型
model.save_pretrained("output_dir")
```
