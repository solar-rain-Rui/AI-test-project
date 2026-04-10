# AI辅助接口自动化测试工具

<div align="center">

🤖 **AI-Powered API Testing Tool**

基于大模型的智能测试用例生成与自动化执行工具

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-orange.svg)](https://streamlit.io/)
[![AutoGen](https://img.shields.io/badge/AutoGen-0.4+-green.svg)](https://microsoft.github.io/autogen/)

</div>

---

##  项目简介

这是一个**AI辅助的接口自动化测试工具**，实现了从需求输入到测试执行的完整闭环。通过大模型（DeepSeek）自动生成测试用例，结合多智能体协作（Writer + Reviewer）进行质量评审，最终执行接口测试并输出结果。

**核心价值**：
-  **效率提升**：自动生成测试用例，覆盖正常/异常/边界场景
-  **质量保障**：多智能体评审 + 规则校验，过滤不合理用例
-  **自动化闭环**：从需求到测试报告的完整流程
-  **数据驱动**：支持pytest参数化测试，可集成CI/CD

---

##  核心功能

| 功能模块 | 说明 |
|---------|------|
| **AI生成测试用例** | 根据需求描述，自动生成JSON格式测试用例（normal/abnormal/boundary） |
| **多智能体协作** | Writer生成 + Reviewer评审，双重质量保障 |
| **JSON容错解析** | 自动修复AI输出中的格式问题（代码块、非法表达式、不可见字符） |
| **用例校验过滤** | 字段完整性校验 + 反常识判断，过滤AI幻觉 |
| **接口自动化执行** | requests发送HTTP请求，自动断言响应状态 |
| **数据持久化** | 保存JSON文件，支持pytest参数化测试 |

---

##  技术架构

```
用户输入需求
    ↓
AI生成测试用例（Writer Agent）
    ↓
AI评审用例质量（Reviewer Agent）
    ↓
JSON解析与容错处理
    ↓
用例校验与过滤
    ↓
接口自动化执行
    ↓
结果展示 + 数据持久化
```

**技术亮点**：
- 多层JSON容错机制，解析成功率95%+
- Writer + Reviewer双Agent协作模式
- 三层质量保障（提示词约束 + AI评审 + 规则校验）
- 数据驱动测试，支持CI/CD集成

---

##  快速启动

### 1. 克隆项目

```bash
git clone https://github.com/your-username/AutoGenTestCase.git
cd AutoGenTestCase
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置API Key

编辑 `config.ini` 文件，填入你的DeepSeek API Key：

```ini
[deepseek]
api_key = sk-your-api-key-here
base_url = https://api.deepseek.com
model = deepseek-chat
```

>  获取API Key：访问 [DeepSeek官网](https://platform.deepseek.com/) 注册并获取

### 4. 启动Web界面

```bash
streamlit run run.py
```

### 5. 开始使用

1. 在界面输入需求描述（如："用户注册接口，需要用户名、密码、邮箱"）
2. 点击"生成并执行测试"按钮
3. 查看AI生成的测试用例和执行结果
4. 测试用例自动保存为 `test_data.json`

### 6. 运行pytest（可选）

```bash
# 先通过Web界面生成 test_data.json
pytest test_api.py -v
```

---

##  示例输入输出

### 输入示例

```
用户注册接口，需要用户名（3-20字符）、密码（至少8位）、邮箱（有效格式）
```

### 输出示例

**AI生成的测试用例（JSON）**：
```json
{
    "normal": [
        {
            "name": "正常注册",
            "input": {"username": "testuser", "password": "Pass1234", "email": "test@example.com"},
            "expected": "注册成功"
        }
    ],
    "abnormal": [
        {
            "name": "用户名为空",
            "input": {"username": "", "password": "Pass1234", "email": "test@example.com"},
            "expected": "注册失败"
        }
    ],
    "boundary": [
        {
            "name": "用户名最小长度",
            "input": {"username": "abc", "password": "Pass1234", "email": "test@example.com"},
            "expected": "注册成功"
        }
    ]
}
```


## 🛠️ 技术栈

| 类别 | 技术 | 说明 |
|------|------|------|
| **Web框架** | Streamlit | 快速构建Web界面 |
| **AI框架** | AutoGen | 微软开源多智能体框架 |
| **大模型** | DeepSeek | 国产大模型，性价比高 |
| **HTTP库** | requests | 发送接口请求 |
| **测试框架** | pytest | 参数化测试，支持CI/CD |
| **数据处理** | json + re | JSON解析与正则处理 |

---

##  项目结构

```
AutoGenTestCase/
├── run.py              # 启动入口
├── page.py             # 核心逻辑（AI生成 + 测试执行）
├── test_api.py         # pytest自动化测试
├── llms.py             # 模型配置
├── config.ini          # API配置
├── requirements.txt    # 依赖列表
├── test_data.json      # 生成的测试用例（运行后生成）
└── README.md           # 项目说明
```



