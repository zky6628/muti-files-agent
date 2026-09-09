import urllib.parse
import json5
from qwen_agent.agents import Assistant
from qwen_agent.tools.base import BaseTool, register_tool
from qwen_agent.gui import WebUI
import os

# 步骤 1：添加一个名为 `my_image_gen` 的自定义工具。
@register_tool('my_image_gen')
class MyImageGen(BaseTool):
    # `description` 用于告诉智能体该工具的功能。
    description = 'AI 绘画（图像生成）服务，输入文本描述，返回基于文本信息绘制的图像 URL。'
    # `parameters` 告诉智能体该工具有哪些输入参数。
    parameters = [{
        'name': 'prompt',
        'type': 'string',
        'description': '期望的图像内容的详细描述',
        'required': True
    }]

    def call(self, params: str, **kwargs) -> str:
        # `params` 是由 LLM 智能体生成的参数。
        prompt = json5.loads(params)['prompt']
        prompt = urllib.parse.quote(prompt)
        return json5.dumps(
            {'image_url': f'https://image.pollinations.ai/prompt/{prompt}'},
            ensure_ascii=False)


# 步骤 2：配置您所使用的 LLM。
llm_cfg = {
    # 使用 DashScope 提供的模型服务：
    'model': 'deepseek-v3',
    'model_server': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
    'api_key': os.getenv('DASHSCOPE_API_KEY'),  # 从环境变量获取API Key
    'generate_cfg': {
        'top_p': 0.8
    }
}

# 步骤 3：定义系统提示词和工具列表
system_instruction = '''你是一个乐于助人的AI助手。
根据以下上下文内容回答用户的问题。如果上下文中没有相关信息，请如实说明。
在收到用户的请求后，你应该：
- 首先绘制一幅图像，得到图像的url，
- 然后运行代码`requests.get`以下载该图像的url，
- 最后从给定的文档中选择一个图像操作进行图像处理。
用 `plt.show()` 展示图像。
你总是用中文回复用户。'''
# tools = ['my_image_gen', 'code_interpreter']  # `code_interpreter` 是框架自带的工具，用于执行代码。
tools = ['my_image_gen']

# 获取文件夹下所有文件
def get_doc_files():
    """获取 docs 文件夹下的所有文件"""
    file_dir = os.path.join('./', 'docs')
    files = []
    if os.path.exists(file_dir):
        # 遍历目录下的所有文件
        for file in os.listdir(file_dir):
            file_path = os.path.join(file_dir, file)
            if os.path.isfile(file_path):  # 确保是文件而不是目录
                files.append(file_path)
    print('加载的文件:', files)
    return files


# ====== 初始化智能体服务 ======
def init_agent_service():
    """初始化智能体服务"""
    try:
        # 获取文档文件列表
        files = get_doc_files()
        print('files=', files)
        bot = Assistant(
            llm=llm_cfg,
            system_message=system_instruction,
            function_list=tools,
            files=files
        )
        print("智能体初始化成功！")
        return bot
    except Exception as e:
        print(f"智能体初始化失败: {str(e)}")
        raise


def app_tui():
    """终端交互模式
    
    提供命令行交互界面，支持：
    - 连续对话
    - 文件输入
    - 实时响应
    """
    try:
        # 初始化助手
        bot = init_agent_service()

        # 对话历史
        messages = []
        while True:
            try:
                # 获取用户输入
                query = input('\n用户问题: ')
                
                # 输入验证
                if not query:
                    print('用户问题不能为空！')
                    continue
                    
                # 构建消息
                messages.append({'role': 'user', 'content': query})

                print("正在处理您的请求...")
                # 运行助手并处理响应
                response = []
                current_index = 0
                for response in bot.run(messages=messages):
                    if current_index == 0:
                        # 尝试获取并打印召回的文档内容
                        if hasattr(bot, 'retriever') and bot.retriever:
                            print("\n===== 召回的文档内容 =====")
                            retrieved_docs = bot.retriever.retrieve(query)
                            if retrieved_docs:
                                for i, doc in enumerate(retrieved_docs):
                                    print(f"\n文档片段 {i+1}:")
                                    print(f"内容: {doc.page_content[:200]}...")
                                    print(f"元数据: {doc.metadata}")
                            else:
                                print("没有召回任何文档内容")
                            print("===========================\n")
                    
                    current_response = response[0]['content'][current_index:]
                    current_index = len(response[0]['content'])
                    print(current_response, end='')
                
                # 将机器人的回应添加到聊天历史
                messages.extend(response)
                print("\n")
            except KeyboardInterrupt:
                print("\n\n退出程序")
                break
            except Exception as e:
                print(f"处理请求时出错: {str(e)}")
                print("请重试或输入新的问题")
    except Exception as e:
        print(f"启动终端模式失败: {str(e)}")


def app_gui():
    """图形界面模式，提供 Web 图形界面"""
    try:
        print("正在启动 Web 界面...")
        # 初始化助手
        bot = init_agent_service()
        
        # 配置聊天界面，列举一些典型问题
        chatbot_config = {
            'prompt.suggestions': [
                '介绍下公司报销章程',
                '帮我生成一幅关于春天的图像',
                '分析一下文档中的关键信息',
            ]
        }
        
        print("Web 界面准备就绪，正在启动服务...")
        # 启动 Web 界面
        WebUI(
            bot,
            chatbot_config=chatbot_config
        ).run()
    except Exception as e:
        print(f"启动 Web 界面失败: {str(e)}")
        print("请检查网络连接和 API Key 配置")


if __name__ == '__main__':
    # 运行模式选择
    app_gui()          # 图形界面模式（默认）
    # app_tui()        # 终端交互模式（可选）

