# ollama_qa.py
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import readline

# 向量数据库目录
VECTOR_DIR = "./vector_store"
# 模型名称
MODEL_NAME = "deepseek-r1:7b"

# 构建检索链流程
def build_qa_chain():
    # 1. 初始化向量数据库
    vector_store = Chroma(
        persist_directory=VECTOR_DIR,
        embedding_function=OllamaEmbeddings(model=MODEL_NAME),
    )

    # 2. 初始化 Ollama 对话模型
    llm = ChatOllama(
        model=MODEL_NAME,
        temperature=0.3,
        # 开启流式响应输出，与下面的回调搭配使用
        streaming=True,
        # 流式响应回调
        callbacks=[StreamingStdOutCallbackHandler()],
        
    )

    # 3. 初始化检索器，并设置检索参数
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 5,
            "fetch_k": 20,
            "lambda_mult": 0.5,
            "score_threshold": 0.4,
        },
    )

    # 4. 设置提示词模板
    system_template = """
        您是一名文章阅读助手，是一个设计用于査询文档来回答问题的代理。
        您可以使用文档检索工具，并基于检索内容来回答问题。
        您可能不查询文档就知道答案，但是您仍然应该查询文档来获得答案。
        如果您从文档中找不到任何信息用于回答问题，则只需返回“抱歉，这个问题我还不知道。”作为答案。
        如果有人提问等关于您的名字的问题，您就回答：“我是超级牛逼哄哄的小天才助手”作为答案。
        上下文：{context}
        """
    prompt = ChatPromptTemplate(
        [
            ("system", system_template),
            ("human", "{question}"),
        ]
    )
    # 构建 LangChain 检索链
    return (
        {
            "context": retriever,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

# 控制台聊天对话
def console_qa():
    print("初始化知识库系统...")

    # 初始化检索链
    chain = build_qa_chain()
    # 交互界面
    print("系统就绪，输入问题开始对话（输入 'exit' 退出）")
    while True:
        try:
            query = input("\n问题：").strip()
            if not query or query.lower() in ("exit", "quit"):
                break

            print("回答：", end="", flush=True)
            response = ""

            # 回答采用流式输出，invoke 将问题传入到 Runnables 管道中
            for chunk in chain.invoke(query):
                response += chunk

            print("\n\n")
            print("==== 请继续对话（输入 'exit' 退出）====")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"出错：{str(e)}")

if __name__ == "__main__":
    console_qa()
    print("对话结束")

  







exit()
#多轮对话
# from langchain_chroma import Chroma
# from langchain_ollama import ChatOllama, OllamaEmbeddings
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser
# from langchain.memory import ConversationBufferMemory
# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
# import readline

# VECTOR_DIR = "./vector_store"
# MODEL_NAME = "deepseek-r1:7b"

# # 初始化会话记忆缓冲区，用于存储对话历史，保存在内存中
# memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")

# def build_qa_chain():

#     vector_store = Chroma(
#         persist_directory=VECTOR_DIR,
#         embedding_function=OllamaEmbeddings(model=MODEL_NAME),
#     )

#     llm = ChatOllama(
#         model=MODEL_NAME,
#         temperature=0.3,
#         callbacks=[StreamingStdOutCallbackHandler()],
#         streaming=True,
#     )

#     retriever = vector_store.as_retriever(
#         search_type="mmr",
#         search_kwargs={
#             "k": 5,
#             "fetch_k": 20,
#             "lambda_mult": 0.5,
#             "score_threshold": 0.4,
#         },
#     )

#     system_template = """
#         您是一名超级牛逼哄哄的小天才助手，是一个设计用于査询文档来回答问题的代理。
#         您可以使用文档检索工具，并基于检索内容来回答问题。
#         您可能不查询文档就知道答案，但是您仍然应该查询文档来获得答案。
#         如果您从文档中找不到任何信息用于回答问题，则只需返回“抱歉，这个问题我还不知道。”作为答案。
#         如果有人提问等关于您的名字的问题，您就回答：“我是超级牛逼哄哄的小天才助手”作为答案。
#         上下文：{context}
#         """
#     prompt = ChatPromptTemplate(
#         [
#             ("system", system_template),
#             MessagesPlaceholder("chat_history"),  # 将历史对话插入到模板中
#             ("human", "{question}"),
#         ]
#     )

#     return (
#         {
#             "question": RunnablePassthrough(),
#             "context": retriever,
#             "chat_history": lambda x: memory.load_memory_variables({})["chat_history"],
#         }
#         | prompt
#         | llm
#         | StrOutputParser()
#     )

# def console_qa():
#     print("初始化知识库系统...")
#     chain = build_qa_chain()
#     print("系统就绪，输入问题开始对话（输入 'exit' 退出）")
#     while True:
#         try:
#             query = input("\n问题：").strip()
#             if not query or query.lower() in ("exit", "quit"):
#                 break

#             print("回答：", end="", flush=True)
#             response = ""
            
#             for chunk in chain.invoke(query):
#                 response += chunk

#             # 截取 </think> 后面的字符串
#             split_string = lambda str: (
#                 str.split("</think>", 1)[1] if "</think>" in str else str
#             )
#             # 将当前对话的问题和回答，保存到记忆缓冲区中
#             memory.save_context({"inputs": query}, {"outputs": split_string(response)})

#             print("\n\n")
#             print("==== 请继续对话（输入 'exit' 退出）====")

#         except KeyboardInterrupt:
#             break


# if __name__ == "__main__":
#     # console_qa()
#     # print("对话结束")


#     from fastapi import FastAPI
#     from pydantic import BaseModel
#     print("初始化知识库系统...")
#     chain = build_qa_chain()
#     def init_system():
#         # 初始化代码（同上）
#         return chain

#     app = FastAPI()
#     qa_system = init_system()

#     class Question(BaseModel):
#         text: str

#     @app.post("/ask")
#     def ask(question: Question):
#         result = qa_system({"query": question.text})
#         return {"answer": result["result"]}



