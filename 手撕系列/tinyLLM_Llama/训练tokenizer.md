# 自己训练一个tokenizer

## tokenizer需要的模块

- **encode**: 将句子转换为token
- **decode**: 将token转换为句子

## SentencePiece 库

是由 Google 开发的一种开源的**文本分词和标记化工具**，广泛应用于自然语言处理（NLP）任务中。它支持多种子词分词算法，如 字节对编码 **（BPE） 和 Unigram 语言模型**，能够将文本分割成子词单元（subwords），从而提高模型的泛化能力和任务效率

SentencePieceProcessor 是 SentencePiece 库的**核心类**，用于加载和使用训练好的 SentencePiece 模型，执行分词（Tokenization）、编码（Encoding）和解码（Decoding）等操作。它是 SentencePiece 模型的主要接口，提供了丰富的功能来处理文本数据。

## tokenizer类中的初始化函数

```python
def __init__(self, tokenizer_model=None):
        """
        初始化分词器。加载预训练的SentencePiece模型，并设置一些特殊的token ID。

        参数:
        tokenizer_model: str, 可选，分词器模型的路径，如果不指定则使用默认路径 TOKENIZER_MODEL。
        """
        # 如果提供了分词器模型路径，使用该路径；否则使用默认模型路径
        model_path = tokenizer_model if tokenizer_model else TOKENIZER_MODEL
        # 确保模型文件存在
        assert os.path.isfile(model_path), model_path

        # 加载 SentencePiece 模型
        self.sp_model = SentencePieceProcessor(model_file=model_path)
        self.model_path = model_path

        # 获取分词器的特殊token和词汇表大小
        self.n_words: int = self.sp_model.vocab_size()  # 词汇表大小
        self.bos_id: int = self.sp_model.bos_id()       # 句子开头 (BOS) 的ID
        self.eos_id: int = self.sp_model.eos_id()       # 句子结尾 (EOS) 的ID
        self.pad_id: int = self.sp_model.pad_id()       # 填充 (PAD) 的ID

```

解释：

- **特殊token**：BOS是句子开头的标记，EOS是句子结尾的标记，PAD是填充的标记。它们在分词器中用于标记句子的开头和结尾，以及填充空白位置。
- **词汇表大小**：SentencePiece模型中的词汇表大小，表示模型可以处理的最大词汇数量。
- **SentencePiece** ： 加载预训练过的分词模型

## tokenizer类中的encode函数

```python
def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        """
        将字符串编码为词元ID列表。可以选择是否添加句子开头 (BOS) 和句子结尾 (EOS) 标记。

        参数:
        s: str, 要编码的字符串。
        bos: bool, 是否在编码的词元列表前添加 BOS 标记。
        eos: bool, 是否在编码的词元列表末尾添加 EOS 标记。

        返回:
        List[int]: 编码后的词元ID列表。
        """
        # 确保输入是字符串类型
        assert type(s) is str
        # 使用SentencePiece将字符串编码为词元ID
        t = self.sp_model.encode(s)
        # 如果需要BOS标记，将其添加到词元列表开头
        if bos:
            t = [self.bos_id] + t
        # 如果需要EOS标记，将其添加到词元列表末尾
        if eos:
            t = t + [self.eos_id]
        return t
```

## tokenizer类中的decode函数

```python
def decode(self, t: List[int]) -> str:
        """
        将词元ID列表解码为字符串。

        参数:
        t: List[int], 词元ID列表。

        返回:
        str: 解码后的字符串。
        """
        return self.sp_model.decode(t)
```

## 完整代码

```python
import os
import struct
from sentencepiece import SentencePieceProcessor
from typing import List

TOKENIZER_MODEL = "./data/tok4096.model"

class Tokenizer:
    def __init__(self, tokenizer_model=None):
        """
        初始化分词器。加载预训练的SentencePiece模型，并设置一些特殊的token ID。

        参数:
        tokenizer_model: str, 可选，分词器模型的路径，如果不指定则使用默认路径 TOKENIZER_MODEL。
        """
        # 如果提供了分词器模型路径，使用该路径；否则使用默认模型路径
        model_path = tokenizer_model if tokenizer_model else TOKENIZER_MODEL
        # 确保模型文件存在
        assert os.path.isfile(model_path), model_path

        # 加载 SentencePiece 模型
        self.sp_model = SentencePieceProcessor(model_file=model_path)
        self.model_path = model_path

        # 获取分词器的特殊token和词汇表大小
        self.n_words: int = self.sp_model.vocab_size()  # 词汇表大小
        self.bos_id: int = self.sp_model.bos_id()       # 句子开头 (BOS) 的ID
        self.eos_id: int = self.sp_model.eos_id()       # 句子结尾 (EOS) 的ID
        self.pad_id: int = self.sp_model.pad_id()       # 填充 (PAD) 的ID

        # 验证分词器词汇表大小是否正确
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        """
        将字符串编码为词元ID列表。可以选择是否添加句子开头 (BOS) 和句子结尾 (EOS) 标记。

        参数:
        s: str, 要编码的字符串。
        bos: bool, 是否在编码的词元列表前添加 BOS 标记。
        eos: bool, 是否在编码的词元列表末尾添加 EOS 标记。

        返回:
        List[int]: 编码后的词元ID列表。
        """
        # 确保输入是字符串类型
        assert type(s) is str
        # 使用SentencePiece将字符串编码为词元ID
        t = self.sp_model.encode(s)
        # 如果需要BOS标记，将其添加到词元列表开头
        if bos:
            t = [self.bos_id] + t
        # 如果需要EOS标记，将其添加到词元列表末尾
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        """
        将词元ID列表解码为字符串。

        参数:
        t: List[int], 词元ID列表。

        返回:
        str: 解码后的字符串。
        """
        return self.sp_model.decode(t)

```

## 训练函数

```python
def train_vocab(vocab_size: int=32000, num_shards: int=20):
    """
    vocab_size: int, 词汇表的大小，决定分词器的词汇量。
    num_shards: int, 用于加快词汇表训练的效率，指定要处理的分片数量。
    """
    # 确保词汇表大小为正数
    assert vocab_size > 0, "Vocab size must be positive"

    # SentencePiece 模型的前缀路径，将用于保存分词器
    prefix = os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}")

    # 1) 将多个分片中的文本导出为单个文本文件 tiny.txt
    tiny_file = os.path.join(DATA_CACHE_DIR, "tiny.txt")
    data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))

    # 创建 tiny.txt 文件并写入指定数量的分片中的文本
    print(f"Writing temporary file {tiny_file} with {num_shards} shards...")
    with open(tiny_file, "w", encoding="utf-8") as of:
        # 遍历前 num_shards 个分片
        for shard in tqdm(shard_filenames[:num_shards]):
            with open(shard, "r") as f:
                data = json.load(f)  # 读取分片中的JSON数据
            # 遍历每个例子，将其中的故事文本写入 tiny.txt 文件
            for example in data:
                text = example["story"]
                text = text.strip()  # 去除文本首尾的空白字符
                of.write(text + "\n")  # 每个文本写入一行

    # 输出生成的 tiny.txt 文件的大小
    print(f"Size is: {os.path.getsize(tiny_file) / 1024 / 1024:.2f} MB")

    # 2) 使用 SentencePiece 训练分词器
    print("Will now train the vocab...")
    spm.SentencePieceTrainer.train(
        input=tiny_file,         # 输入文件为之前生成的 tiny.txt
        model_prefix=prefix,     # 模型前缀路径
        model_type="bpe",        # 使用 Byte-Pair Encoding (BPE) 训练分词器
        vocab_size=vocab_size,   # 词汇表大小
        self_test_sample_size=0, # 自测样本大小设置为 0
        input_format="text",     # 输入文件格式为纯文本
        character_coverage=1.0,  # 覆盖所有字符（包括非常见字符）
        num_threads=os.cpu_count(),  # 使用 CPU 的线程数
        split_digits=True,       # 拆分数字
        allow_whitespace_only_pieces=True,  # 允许仅由空格组成的词元
        byte_fallback=True,      # 启用字节级回退
        unk_surface=r" \342\201\207 ",  # UNK token 表示未知字符的方式
        normalization_rule_name="identity"  # 使用“identity”归一化规则
    )

    # 3) 可选的清理操作，询问用户是否删除临时文件 tiny.txt
    dec = input(f"Delete the temporary file {tiny_file}? [y/N] ")
    if dec.lower() == "y":
        os.remove(tiny_file)  # 删除临时文件
        print(f"Deleted {tiny_file}")

    # 输出模型保存的路径
    print(f"Trained tokenizer is in {prefix}.model")
    print("Done.")
```

### 数据分片

数据被分割成多个分片文件（.json 格式），每个分片文件包含多个文本样本。通过 num_shards 参数控制处理的分片数量，可以加快训练速度，减少每次加载到内存中的数据量，提高训练效率。

### 临时文件

tiny.txt 是一个临时文件，用于将多个分片中的文本合并成一个文件，便于 SentencePiece 训练。训练完成后，可以选择删除该临时文件以节省磁盘空间。

### SentencePiece 训练参数

model_type="bpe"：使用 BPE 算法进行分词。vocab_size：指定词汇表大小。character_coverage=1.0：覆盖所有字符，包括非常见字符。byte_fallback=True：启用字节级回退，确保所有字符都能被处理。num_threads=os.cpu_count()：使用 CPU 的所有线程加速训练。
