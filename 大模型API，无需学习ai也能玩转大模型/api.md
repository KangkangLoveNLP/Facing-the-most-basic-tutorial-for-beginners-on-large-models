# 大模型api，无需学习ai知识也能玩转大模型

## 1. 什么是api

**大模型API**是一种允许开发者在自己的应用程序、服务或研究中整合和**使用大型预训练模型功能的接口**。通过这些API，开发者无需自己训练模型或拥有强大的计算资源，**即可轻松地调用大模型的能力**。

说白话就是，**大模型API**可以让开发者**远程调用大模型**，而无需自己训练模型或拥有强大的计算资源。

许多主流大模型都提供了API接口，开发者可以**通过这些接口**调用大模型，实现**文本生成、图像生成、语音合成等**等功能。

**注意以下表格仅供参考**：

**OpenAI（美元，USD）**：

| 模型名称 | 输入价格（USD/1M tokens） | 输出价格（USD/1M tokens） | 备注 |
| --- | --- | --- | --- |
| GPT-4.5 | 270.00 | 1080.00 | 最新聊天模型，高性能 |
| GPT-4o | 18.13 | 72.50 | 多模态，支持图像、文本 |
| GPT-4o mini | 1.09 | 4.35 | 成本效益高，适合轻量任务 |
| GPT-4 Turbo | 72.50 | 217.50 | 优化对话，高性能 |
| GPT-4 | 217.50 | 435.00 | 经典高性能模型 |
| GPT-3.5 Turbo (0125) | 3.63 | 10.88 | 对话优化，普及型 |
| o1 | 21.75 | 87.00 | 复杂推理，顶级性能 |
| o1 mini | 21.75 | 87.00 | 推理任务，成本效益高 |

---

**Anthropic（美元，USD）**：

| 模型名称 | 输入价格（USD/1M tokens） | 输出价格（USD/1M tokens） | 备注 |
| --- | --- | --- | --- |
| Claude 3.5 Sonnet | 21.75 | 108.75 | 高性能，平衡速度与成本 |
| Claude 3 Opus | 108.75 | 543.75 | 最强大模型，高级分析 |
| Claude 3 Haiku | 1.81 | 9.05 | 轻量级，成本最低 |
| Claude 3.7 Sonnet | 21.75 | 108.75 | 支持代码运行 |

---

**DeepSeek（人民币，CNY）**：

| 模型名称 | 输入价格（USD/1M tokens） | 输出价格（USD/1M tokens） | 备注 |
| --- | --- | --- | --- |
| DeepSeek-V3 (deepseek-chat) | 5.07 | 81.00 | 通用对话，性价比高 |
| DeepSeek-R1 (deepseek-reasoner) | 10.14 | 162.00 | 高级推理，含CoT |

---

**阿里云（人民币，CNY）**：

| 模型名称 | 输入价格（CNY/1M tokens） | 输出价格（CNY/1M tokens） | 备注 |
| --- | --- | --- | --- |
| Qwen-Max | 40.00 | 120.00 | 通用模型 |
| Qwen-Long | 40.00 | 120.00 | 长文本优化 |
| Qwen-Turbo | 40.00 | 120.00 | 对话优化 |

---

**百度（人民币，CNY）**：

| 模型名称 | 输入价格（CNY/1M tokens） | 输出价格（CNY/1M tokens） | 备注 |
| --- | --- | --- | --- |
| ERNIE 4.0 | 12.00 | 12.00 | 通用模型 |
| ERNIE Speed/Lite/Tiny | 免费 | 免费 | 轻量级模型 |

---

**科大讯飞（人民币，CNY）**：

| 模型名称 | 输入价格（CNY/1M tokens） | 输出价格（CNY/1M tokens） | 备注 |
| --- | --- | --- | --- |
| Spark Lite | 免费 | 免费 | 轻量级模型 |

---

**智谱（人民币，CNY）**：

| 模型名称 | 输入价格（CNY/1M tokens） | 输出价格（CNY/1M tokens） | 备注 |
| --- | --- | --- | --- |
| GLM-4-Flash | 免费 | 免费 | 轻量级模型 |

---

**其他（美元，USD）**：

| 模型名称 | 输入价格（USD/1M tokens） | 输出价格（USD/1M tokens） | 备注 |
| --- | --- | --- | --- |
| Gemini 1.5 Flash | 5.07 | 15.21 | 低延迟，成本效益高 |
| Gemini 1.5 Pro | 25.35 | 76.05 | 高性能，超长上下文 |

我们可以看到deepseek的模型性能和GPT相当，但是token的价格比GPT低很多，难怪deepseek在国外也能大受欢迎。

**输入价格（Input Price）**：

输入价格是指用户向模型发送的输入数据的费用。输入数据通常包括用户的问题、上下文信息或其他需要模型处理的内容。费用是根据输入数据的tokens数量来计算的。

**Token**：在自然语言处理中，一个token可以是一个单词、一个子词（如“sub-word”）或一个字符，具体取决于模型的分词方式。例如，句子“Hello, how are you?”可能被分词为多个tokens。

**输出价格（Output Price）**：

输出价格是指模型生成的输出数据的费用。输出数据是模型根据输入生成的响应，例如回答、生成的文本、翻译结果等。费用同样是根据输出数据的tokens数量来计算的。

**一句话，token相当于字或者词，输入价格和输出价格是计算token数量的。，输入价格是按照你输入到模型中的token数量来计算，输出价格是按照模型生成的token数量来计算**。

## 2. 如何调用API

我们以百度智能云示例：[百度智能云](https://cloud.baidu.com/doc/API/index.html)

以python代码为例子，事实上它也支持java,go,c++,PHP,C#等语言。

```python
# 导入必要的模块
import requests  # 用于发送 HTTP 请求
import json  # 用于处理 JSON 数据

# 定义主函数
def main():
    # 设置 API 请求的 URL
    # 注意：这里的 URL 是百度 Qianfan API 的接口地址，需要确保是正确的
    url = "https://qianfan.baidubce.com/v2/chat/completions"
    
    # 构造请求的 JSON 数据（payload）
    # 这里定义了要发送给模型的内容，包括模型名称、用户消息等
    payload = json.dumps({
        "model": "ernie-4.0-turbo-8k",  # 指定使用的模型版本
        "messages": [  # 用户的消息列表
            {
                "role": "user",  # 指定消息的角色为用户
                "content": "您好"  # 用户发送的消息内容
            }
        ],
        "web_search": {  # 控制是否启用网络搜索功能
            "enable": False,  # 不启用网络搜索
            "enable_citation": False,  # 不启用引用功能
            "enable_trace": False  # 不启用跟踪功能
        }
    }, ensure_ascii=False)  # 确保 JSON 数据中可以包含中文字符

    # 设置 HTTP 请求的头部信息
    headers = {
        'Content-Type': 'application/json',  # 告诉服务器请求体的格式是 JSON
        'appid': '',  # 需要从百度云平台获取的 appid,需要购买，在控制台中创建和查看
        'Authorization': 'Bearer '  # 需要从百度云平台获取的访问令牌，格式为 "Bearer <你的Token>"
    }
    
    # 发送 HTTP POST 请求
    # 使用 requests.request 方法发送 POST 请求，包含 URL、头部信息和请求数据
    response = requests.request("POST", url, headers=headers, data=payload.encode("utf-8"))
    
    # 打印响应内容
    # response.text 是服务器返回的响应内容，通常是 JSON 格式的字符串
    print(response.text)

# 确保直接运行此脚本时才会执行 main 函数
if __name__ == '__main__':
    main()
```

假设可以运行就会邮以下输出

```python
请求的 URL: https://qianfan.baidubce.com/v2/chat/completions
发送的 payload: {"model": "ernie-4.0-turbo-8k", "messages": [{"role": "user", "content": "您好"}], "web_search": {"enable": false, "enable_citation": false, "enable_trace": false}}
响应状态码: 200
响应内容: {"result": "您好！很高兴为您服务。", "status": "success"}
```

类似的，我们可以在其他的API供应商中购买其他API，比如阿里云、腾讯云、华为云等。
也可以调用其他模型，比如文生图等。

新用户有新手大礼包，一些模型免费用，有几百万token的免费额度来学习使用。
我们可以看到，仅仅只需几行代码，就可以轻松调用大模型。

**作者码字不易，觉得有用的话不妨点个赞吧，关注我，持续为您更新AI的优质内容**。
