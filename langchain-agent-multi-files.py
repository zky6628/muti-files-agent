#!/usr/bin/env python
# coding: utf-8
"""
基于 LangChain 的多文件 RAG 应用
支持加载 docs 文件夹下的多种格式文件进行问答
"""

import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 获取 API Key
DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY')
if not DASHSCOPE_API_KEY:
    raise ValueError("请设置环境变量 DASHSCOPE_API_KEY")


# 步骤 1：加载文档并创建索引
def load_documents_and_create_index(file_dir: str = './docs', persist_dir: str = './langchain_storage'):
    """加载文档文件夹中的所有文件并创建向量索引"""
    
    # 创建嵌入模型
    embeddings = DashScopeEmbeddings(
        model="text-embedding-v1",
        dashscope_api_key=DASHSCOPE_API_KEY,
    )
    
    # 检查索引是否已存在
    if os.path.exists(persist_dir):
        try:
            # 从存储中加载索引
            vector_store = FAISS.load_local(
                persist_dir, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            print("从存储加载索引成功")
            return vector_store
        except Exception as e:
            print(f"加载索引失败: {e}，将重新创建索引")
    
    # 如果索引不存在，创建新索引
    if not os.path.exists(file_dir):
        print(f"文档目录 {file_dir} 不存在")
        return None
    
    # 加载目录下的所有 docx 文件
    loader = DirectoryLoader(
        file_dir,
        glob="**/*.docx",
        loader_cls=Docx2txtLoader,
        show_progress=True,
    )
    documents = loader.load()
    print(f"加载了 {len(documents)} 个文档")
    
    if not documents:
        print("没有找到任何文档")
        return None
    
    # 文本分割
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"文本被分割成 {len(chunks)} 个块")
    
    # 创建向量索引
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    # 保存索引
    os.makedirs(persist_dir, exist_ok=True)
    vector_store.save_local(persist_dir)
    print(f"索引已保存到 {persist_dir}")
    
    return vector_store


# 步骤 2：创建问答链
def create_qa_chain(llm):
    """创建 QA 问答链 (LangChain 1.x LCEL 写法)"""
    
    # QA Prompt 模板
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个乐于助人的AI助手。
根据以下上下文内容回答用户的问题。如果上下文中没有相关信息，请如实说明。
你总是用中文回复用户。

上下文内容:
{context}"""),
        ("human", "{question}")
    ])
    
    # 创建问答链 (LCEL 管道语法)
    qa_chain = qa_prompt | llm | StrOutputParser()
    
    return qa_chain


# 步骤 3：主函数
def main():
    """主函数：交互式循环接收用户问题"""
    # 配置 LLM
    llm = ChatTongyi(
        model_name="deepseek-v3",
        dashscope_api_key=DASHSCOPE_API_KEY
    )

    # 加载文档并创建索引
    vector_store = load_documents_and_create_index()
    if vector_store is None:
        print("无法创建索引，程序退出")
        return

    # 创建问答链
    qa_chain = create_qa_chain(llm)

    # 交互式循环：接收用户问题，输入空行或 q/exit/quit 退出
    while True:
        query = input("\n请输入问题 (输入 q 退出): ").strip()
        if not query or query.lower() in ("q", "exit", "quit"):
            print("已退出")
            break

        print(f"\n用户查询: {query}\n")

        # 相似度搜索，找到相关文档
        docs = vector_store.similarity_search(query, k=5)

        # 显示召回的文档内容
        print("===== 召回的文档内容 =====")
        if docs:
            for i, doc in enumerate(docs):
                print(f"\n文档片段 {i+1}:")
                print(f"内容: {doc.page_content[:200]}...")
                print(f"来源: {doc.metadata.get('source', '未知')}")
        else:
            print("没有召回任何文档内容")
        print("===========================\n")

        # 格式化上下文
        context = "\n\n".join(doc.page_content for doc in docs)

        # 执行问答链
        print("===== AI 回复 =====")
        response = qa_chain.invoke({"context": context, "question": query})
        print(response)
        print("===================")


if __name__ == "__main__":
    main()
