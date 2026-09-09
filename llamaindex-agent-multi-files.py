#!/usr/bin/env python
# coding: utf-8

import os
import asyncio
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
    load_index_from_storage,
    PromptTemplate,
)
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.dashscope import DashScope
from llama_index.embeddings.dashscope import (
    DashScopeEmbedding,
    DashScopeTextEmbeddingModels,
)


# 步骤 1：配置 LLM 和 Embedding
def setup_llm_and_embedding():
    """配置 LLM 和 Embedding，使用 DashScope"""
    api_key = os.getenv('DASHSCOPE_API_KEY')
    
    if not api_key:
        raise ValueError("请设置环境变量 DASHSCOPE_API_KEY")
    
    # 使用 DashScope LLM
    llm = DashScope(
        model="deepseek-v3",
        api_key=api_key,
        temperature=0.7,
        top_p=0.8,
    )
    
    # 使用 DashScope Embedding（自动从环境变量读取 API key）
    embed_model = DashScopeEmbedding(
        model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V1,
    )
    
    return llm, embed_model


# 步骤 2：加载文档并创建索引
def load_documents_and_create_index(file_dir: str = './docs'):
    """加载文档文件夹中的所有文件并创建向量索引"""
    # 检查索引是否已存在
    persist_dir = "./storage"
    
    if os.path.exists(persist_dir):
        try:
            # 从存储中加载索引
            storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
            index = load_index_from_storage(storage_context)
            print("从存储加载索引成功")
            return index
        except Exception as e:
            print(f"加载索引失败: {e}，将重新创建索引")
    
    # 如果索引不存在，创建新索引
    if not os.path.exists(file_dir):
        print(f"文档目录 {file_dir} 不存在")
        return None
    
    # 读取文档
    reader = SimpleDirectoryReader(file_dir)
    documents = reader.load_data()
    
    if not documents:
        print("没有找到任何文档")
        return None
    
    print(f"加载了 {len(documents)} 个文档")
    
    # 创建向量索引
    index = VectorStoreIndex.from_documents(documents)
    
    # 保存索引
    index.storage_context.persist(persist_dir=persist_dir)
    print(f"索引已保存到 {persist_dir}")
    
    return index


# 步骤 3：创建智能体
def create_agent(index, llm):
    """创建 ReAct 智能体"""
    # 创建检索器
    retriever = index.as_retriever(similarity_top_k=5)
    
    # 创建查询引擎（用于检索工具）
    # 拒答指令下沉到 query_engine 的 QA 提示词。
    # 原因：retrieve_documents 工具返回的是内层模型已合成的答案，
    # Agent 外层的 system_prompt 管不到 query_engine 内部这次真正的生成调用，
    # 拒答约束必须写进这一层才会生效
    qa_prompt = PromptTemplate(
        "以下是文档中的上下文内容：\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "请仅根据上述上下文内容回答用户的问题，禁止使用你自身的知识编造或补充答案。"
        "如果上下文中没有与问题相关的信息，请直接回答：文档中没有相关信息。"
        "请用中文回复。\n"
        "问题: {query_str}\n"
        "回答: "
    )
    query_engine = index.as_query_engine(
        similarity_top_k=5,
        # text_qa_template=qa_prompt, # 放开后 LLM会明确要求只能基于工具返回的内容作答
    )
    
    # 定义系统提示词
    system_instruction = '''你是一个乐于助人的AI助手。你总是用中文回复用户。
回答任何问题都必须遵守以下规则：
1. 收到问题后必须先调用 retrieve_documents 工具检索文档，禁止跳过工具直接回答。
2. 只能基于工具返回的内容作答。
3. 当工具返回"文档中没有相关信息"时，你的最终回答必须且只能是：文档中没有相关信息。
4. 禁止使用你自身的知识编造或补充答案。'''
    
    # 创建检索工具（用于查询文档）
    def retrieve_documents(query: str) -> str:
        """从文档中检索相关信息"""
        response = query_engine.query(query)
        return str(response)
    
    retrieve_tool = FunctionTool.from_defaults(fn=retrieve_documents)
    
    # 创建智能体（新版 API）
    agent = ReActAgent(
        tools=[retrieve_tool],
        llm=llm,
        system_prompt=system_instruction,
    )
    
    return agent, retriever


# 步骤 4：主函数
async def main():
    """主函数"""
    # 配置 LLM 和 Embedding
    llm, embed_model = setup_llm_and_embedding()
    Settings.llm = llm
    Settings.embed_model = embed_model
    
    # 加载文档并创建索引
    index = load_documents_and_create_index()
    if index is None:
        print("无法创建索引，程序退出")
        return
    
    # 创建智能体
    agent, retriever = create_agent(index, llm)
    
    # 执行查询
    # query = "介绍下公司报销章程"

    # 交互式循环：接收用户问题，输入空行或 q/exit/quit 退出
    while True:
        query = input("\n请输入问题 (输入 q 退出): ").strip()
        if not query or query.lower() in ("q", "exit", "quit"):
            print("已退出")
            break
        print(f"\n用户查询: {query}\n")
    
        # 显示召回的文档内容
        print("\n===== 召回的文档内容 =====")
        retrieved_nodes = retriever.retrieve(query)
        if retrieved_nodes:
            for i, node in enumerate(retrieved_nodes):
                print(f"\n文档片段 {i+1}:")
                # 处理特殊字符，避免 Windows 控制台编码问题
                text_preview = node.text[:200].encode('gbk', errors='replace').decode('gbk')
                print(f"内容: {text_preview}...")  # 只显示前200个字符
                print(f"元数据: {node.metadata}")
                if hasattr(node, 'score'):
                    print(f"相似度分数: {node.score}")
        else:
            print("没有召回任何文档内容")
        print("===========================\n")
        
        # 使用智能体回答问题（新版 API 使用 run 方法，是异步的）
        print("\n===== 智能体回复 =====")
        response = await agent.run(query)
        # 处理特殊字符，避免 Windows 控制台编码问题
        response_str = str(response).encode('gbk', errors='replace').decode('gbk')
        print(response_str)
        print("======================\n")


if __name__ == "__main__":
    asyncio.run(main())