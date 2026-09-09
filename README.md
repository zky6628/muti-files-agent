# multi-files-agent

基于阿里云 DashScope（deepseek-v3 + text-embedding）的多文档 RAG 问答示例项目，用 LangChain、LlamaIndex、qwen-agent 三种主流框架，分别实现了同一业务场景（公司制度文档问答），用于框架横向对。

## 项目结构

```
muti-files-agent/
├── docs/                            # 知识库文档（公司报销/假期/办公用品申领制度 .docx）
├── langchain-agent-multi-files.py   # 方案 A：LangChain RAG 问答链（FAISS）
├── llamaindex-agent-multi-files.py  # 方案 B：LlamaIndex ReAct Agent
├── qwen-agent-multi-files.py        # 方案 C：qwen-agent Assistant（TUI / Gradio WebUI）
├── requirements.txt                 # 依赖清单（锁定版本）
├── storage/                         # LlamaIndex 索引持久化（方案 B）
├── langchain_storage/               # FAISS 索引持久化（方案 A）
├── reports/                         # 方案 D 生成的分析报告输出目录
├── workspace/                       # qwen-agent 工具运行时产物（方案 C）
└── llamaindex-agent-multi-files-logic.html  # 方案 B 逻辑图（浏览器打开查看）
```

## 运行配置

### 环境要求

- Windows / Python 3.10 ~ 3.12（本项目虚拟环境为 3.12.10）
- 阿里云 DashScope API Key（在[百炼控制台](https://bailian.console.aliyun.com/)获取）

### 安装步骤

```powershell

# 或 创建并激活虚拟环境（用 venv，不装全局）
python -m venv .venv
.venv\Scripts\activate

# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 API Key（PowerShell，会话级）
$env:DASHSCOPE_API_KEY = "sk-你的key"
# 或永久设置：
# setx DASHSCOPE_API_KEY "sk-你的key"   （设置后重开终端生效）

# 3. 运行方案：
# 如方案 A：LangChain RAG 问答链（FAISS）
python langchain-agent-multi-files.py
```
