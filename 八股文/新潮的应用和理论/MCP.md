# MCP 的前世今生

## 一、MCP的由来

MCP（**Model Context Protocol，模型上下文协议**）是由 **Anthropic** 公司于 **2024 年 11 月** 正式提出并开源的。Anthropic 是一家由前 **OpenAI 研究人员创立的公司**，专注于可解释性和安全 AI 系统的研究。

MCP 的提出背景是为了解决**大型语言模型（LLM）与外部数据源及工具之间缺乏标准化交互接口的问题**。此前，虽然 **Function Call 等机制已经被广泛采用，但不同厂商的实现方式存在差异**，导致集成成本高、缺乏通用性。MCP 的目标是创建一个通用的、标准化的协议，使得 AI 模型能够以一致的方式连接各种数据源和工具。

MCP 的发明者包括 Anthropic 的工程师 **Justin Spahr-Summers** 和 **David Soria Parra**。他们希望 MCP 能够成为一个真正的开放项目，吸引更多的 AI 实验室和公司参与，而不仅仅被视为“Anthropic 的协议”。

## 二、MCP比之其他的协议好在哪里？

MCP干的事情的本质其实就是相当于秦始皇干的事情。他做到的标准化和模块化，MCP本质是一种技术协议，是智能体开发中的共同约定的一种规范，就好比“书同文，车同轨”。在统一的规范下，大家协作的效率就能大幅度提高。

总的来说，MCP解决的最大痛点，就是Agent开发中调用外部工具的技术门槛过高的问题

## 三、其他的解决方案

### 1. Function Call

Function Call 是一种允许 AI 模型通过调用外部函数来扩展其能力的技术机制，它使AI模型能够跳出自身的知识边界，通过调用外部API、数据库或其他服务来完成任务。例如，AI模型可以通过Function Call查询实时天气、预订机票、发送邮件或执行其他复杂的操作

#### function call 的本质

**Natural Language Interface**，自然语言连接一切，通过编写函数和函数描述，让模型调用函数，实现各种功能。我们需要编写函数和函数定义，让大模型知道函数定义，使得模型能够调用外部工具，得到函数的响应后输出结果。但是传统的function call 没有统一的标准，都是由厂商自定义，新增的工具需要适配

**特点**：

动态调用：根据用户需求调用合适的函数。
上下文感知：结合对话上下文选择最佳函数。
扩展性强：支持多种API和服务集成。

#### function call和MCP的对比

**MCP**:

- 适用场景： 涉及复杂、多步骤的交互，需要长时间维持上下文，任务需要在创造力与控制力之间取得平衡
- 执行方式： 异步调用，发送请求后程序不会等待结果，会继续执行其他代码，等结果出来再处理
- 技术实现：协议标准，提供开放标准（遵循 JSON-RPC 2.0），统一的服务访问方式
- 拓展性： 高，支持即插即用，新增数据源或工具只需符合协议即可接入

**Function Call**：

- 适用场景： 适合需要快速响应和高效执行的简单任务，例如实时翻译、情感分析、数据提取等
- 执行方式： 同步调用，调用函数后程序会一直等待函数执行完返回结果，才继续执行后续代码
- 技术实现：厂商自定义（如 OpenAI 的 JSON 格式），无统一标准
- 拓展性： 受限于厂商接口，新增工具需重新适配

#### funnction call 的具体流程

##### 1. **函数定义和函数编写**

开发者以 JSON 格式定义函数的元数据，包括函数名称、参数描述、返回值类型等。例如，定义一个查询天气的函数：

```json
{
  "name": "get_weather",
  "description": "获取指定地区的当前天气",
  "parameters": {
    "type": "object",
    "properties": {
      "location": {
        "type": "string",
        "description": "城市或地区名称，例如：北京"
      },
      "unit": {
        "type": "string",
        "enum": ["celsius", "fahrenheit"],
        "description": "温度单位"
      }
    },
    "required": ["location"]
  }
}
```

当然我们在外部要编写好**get_weather**这个函数，设定好函数的参数，以及函数的返回值类型。

##### 2. **处理用户输入请求**

用户通过自然语言提出请求，例如：“北京今天的天气如何？”，从中提取出响应的槽位，作为函数调用的参数。我们可以编写函数来处理文本得到结构化数据。

##### 3. **模型解析意图**

AI 模型（如 GPT-4）分析用户的请求，判断是否需要调用外部函数。如果需要，模型会生成一个包含函数名称和参数的 JSON 对象。例如：

```json
{
  "tool_calls": [
    {
      "function": {
        "name": "get_weather",
        "arguments": "{\"location\": \"北京\", \"unit\": \"celsius\"}"
      }
    }
  ]
}
```

##### 4. **执行外部函数**

开发者的后端系统接收模型生成的函数调用请求，并执行对应的函数逻辑。例如，调用天气 API 获取北京的天气数据。

##### 5. **返回函数结果**

外部函数执行完成后，将结果返回给 AI 模型。例如：

```json
{
  "location": "北京",
  "temperature": "25",
  "unit": "celsius",
  "forecast": ["sunny", "windy"]
}
```

##### 6. **模型生成最终响应**

AI 模型根据函数返回的结果，生成自然语言的最终回复，例如：“北京今天晴天，25℃”。

##### 7. **消息结构**

在 Function Call 的场景中，消息历史需要包含完整的对话上下文，包括用户输入、模型生成的工具调用请求（`tool_calls`），以及工具执行后的返回结果。例如：

```json
[
  {
    "role": "user",
    "content": "北京今天的天气如何？"
  },
  {
    "role": "assistant",
    "content": null,
    "tool_calls": [
      {
        "id": "call_abc123",
        "type": "function",
        "function": {
          "name": "get_weather",
          "arguments": "{\"location\": \"北京\", \"unit\": \"celsius\"}"
        }
      }
    ]
  },
  {
    "role": "function",
    "name": "get_weather",
    "content": "{\"location\": \"北京\", \"temperature\": \"25\", \"unit\": \"celsius\", \"forecast\": [\"sunny\", \"windy\"]}"
  }
]
```

### 8. **代码示例**

以下是一个基于 OpenAI 的 Python 代码示例：

```python
import openai
import json

openai.api_key = "your-api-key"

def get_weather(location, unit="celsius"):
    # 模拟天气 API 调用
    return json.dumps({
        "location": location,
        "temperature": "25",
        "unit": unit,
        "forecast": ["sunny", "windy"]
    })

def run_conversation():
    messages = [{"role": "user", "content": "北京今天的天气如何？"}]
    functions = [
        {
            "name": "get_weather",
            "description": "获取指定地区的当前天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["location"]
            }
        }
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=messages,
        functions=functions,
        function_call="auto"
    )

    response_message = response.choices[0].message
    if "function_call" in response_message:
        function_name = response_message["function_call"]["name"]
        function_args = json.loads(response_message["function_call"]["arguments"])

        if function_name == "get_weather":
            function_response = get_weather(
                location=function_args["location"],
                unit=function_args.get("unit", "celsius")
            )

        messages.append(response_message)
        messages.append({
            "role": "function",
            "name": function_name,
            "content": function_response
        })

        final_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=messages
        )
        return final_response.choices[0].message['content']
    else:
        return response_message['content']

print(run_conversation())
```

## 四、MCP的核心组件

MCP的核心组件包括：

- 主机（Host）：运行LLM的应用程序（如Claude Desktop），负责发起与MCP服务器的连接。作为用户直接交互的 AI 应用程序，例如 Claude Desktop 或增强型 IDE（如 Cursor）。它负责管理整个系统，包括启动和管理 MCP 客户端，控制客户端的连接权限和生命周期。
- 客户端（Client）：在主机应用程序内部运行，与MCP服务器建立1:1连接。作为宿主应用和 MCP Server 之间的中间件，负责建立和维护与服务器的连接。
- 服务器（Server）：提供对外部数据源和工具的访问，响应客户端的请求，通过标准化接口暴露给 AI 模型。
- LLM：大型语言模型，通过MCP获取上下文并生成输出。

**工作流程**：

主机启动客户端。
客户端连接到MCP服务器。
服务器提供资源、提示或工具。
LLM使用这些信息生成响应。

## 五、 编写一个简单的MCP程序
