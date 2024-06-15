import telebot
from dotenv import load_dotenv
from telebot.types import Message
from qdrant_client import QdrantClient

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine.retriever_query_engine import RetrieverQueryEngine
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, get_response_synthesizer, PromptTemplate
from llama_index.llms.together import TogetherLLM#, LlamaCPP
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.tools import ToolMetadata

from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool
import os

load_dotenv()

llm = TogetherLLM(
        # model="meta-llama/Llama-3-8b-chat-hf",
        model="Qwen/Qwen2-72B-Instruct",
        api_key=os.getenv("together_api"))

Settings.llm = llm

def load_vectorized(path):
    """
    Загружаем из векторной базы данные
    """

    client = QdrantClient(path=path)

    Settings.embed_model = HuggingFaceEmbedding(
        model_name = 'paraphrase-multilingual-mpnet-base-v2'
    )

    vector_store = QdrantVectorStore(collection_name="deep_collection", client=client)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)

    return index


def agent():
    buhg_engine = load_vectorized('admin_db').as_query_engine(similarity_top_k=3)
    econ_engine = load_vectorized('econ_db').as_query_engine(similarity_top_k=3)


    query_engine_tools = [
        QueryEngineTool(
            query_engine=buhg_engine,
            metadata=ToolMetadata(
                name="admin",
                description="""Административные вопросы.""",
                return_direct=False,
            ),
        ),
        QueryEngineTool(
            query_engine=econ_engine,
            metadata=ToolMetadata(
                name="economic",
                description="""Зарплатные, денежные, экономические вопросы.""",
                return_direct=False,
            ),
        ),
    ]

    # initialize ReAct agent
    agent = ReActAgent.from_tools(query_engine_tools, llm=llm, verbose=True)


    react_system_header_str = """\

    You are designed to find answers to users' questions on various topics

    ## Tools
    You have access to a wide variety of tools. You are responsible for using
    the tools in any sequence you deem appropriate to complete the task at hand.
    This may require breaking the task into subtasks and using different tools
    to complete each subtask. Ahswer on russian. 

    You have access to the following tools:
    {tool_desc}

    ## Output Format
    To answer the question, please use the following format.

    ```
    Thought: I need to use a tool to help me answer the question.
    Action: tool name (one of {tool_names}) if using a tool.
    Action Input: the input to the tool, in a JSON format representing the kwargs (e.g. {{"input": "hello world", "num_beams": 5}})
    ```

    Please ALWAYS start with a Thought.

    Please use a valid JSON format for the Action Input. Do NOT do this {{'input': 'hello world', 'num_beams': 5}}.

    If this format is used, the user will respond in the following format:

    ```
    Observation: tool response
    ```

    You should keep repeating the above format until you have enough information
    to answer the question without using any more tools. At that point, you MUST respond
    in the one of the following two formats:

    ```
    Thought: I can answer without using any more tools.
    Answer: [your answer here]
    ```

    ```
    Thought: I cannot answer the question with the provided tools.
    Answer: Извините, у меня нет ответа на этот вопрос. Переформулируйте его пожалуйста
    ```

    ## Additional Rules
    - The answer MUST contain a sequence of bullet points that explain how you arrived at the answer. This can include aspects of the previous conversation history.
    - You MUST obey the function signature of each tool. Do NOT pass in no arguments if the function expects arguments.
    - Provide the page number where the information was found in the response

    ## Current Conversation
    Below is the current conversation consisting of interleaving human and assistant messages.

    """
    react_system_prompt = PromptTemplate(react_system_header_str)
    agent.update_prompts({"agent_worker:system_prompt": react_system_prompt})

    return agent 

'''
def save_vectorized():
    """
    Разбиваем документы на чанки по 1024 и сохраняем в векторную базу данных qdrant
    """
    docs = SimpleDirectoryReader(f"data").load_data()

    # db = chromadb.PersistentClient(path=f"{dir_name}\\db")
    client = QdrantClient(path="db")


    Settings.embed_model = HuggingFaceEmbedding(
        model_name = 'paraphrase-multilingual-mpnet-base-v2'
    )

    vector_store = QdrantVectorStore(collection_name="deep_collection", client=client)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    splitter = SentenceSplitter(
        chunk_size=1024,
        chunk_overlap=48
    )
    nodes =splitter.get_nodes_from_documents(docs)

    Settings.chunk_size = 1024


    index = VectorStoreIndex(nodes, storage_context=storage_context, show_progress=True)
'''



'''
def request_settings(index):
    """
    Настраиваем LLM, прописываем промт
    """
    # для быстрого инференса используется llama3 развернутая на together
    #########################################################
    Settings.llm = TogetherLLM(
        # model="meta-llama/Llama-3-8b-chat-hf",
        model="Qwen/Qwen2-72B-Instruct",
        api_key='b1ff1047cf3750cc14ae886eed55381caba5be6989bcf35349deb7f1bd15cc61')
    Settings.context_window = 4096

    # тоже самое (и даже лучше) можно развернуть локально, но нужны мощности для адекватного отклика
    #########################################################
    # Settings.llm = LLamaCPP(
    #     model_path='saiga3',
    #     temperature=0.1,
    #     max_new_tokens=256,
    #     context_window=8196,
    #     model_kwargs={"n_gpu_layers": -1},
    #     verbose=True
    # )
    #########################################################
    
    resoinse_synthesizer = get_response_synthesizer()

    query_engine = RetrieverQueryEngine(
        retriever=index.as_retriever(similarity_top_k=5),
        response_synthesizer=resoinse_synthesizer,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.1)]
    )
    # service_context = ServiceContext.from_defaults(callback_manager=CallbackManager([cl.LlamaIndexCallbackHandler()]))

    promt = (
        """
        Ты помощник службы поддержки в РосАтом. Используй информацию из контекста чтобы помочь пользователям найти ответ на их вопрос.
        Отвечай на русском, будь вежлив, сокращай ответ.
        Если контекст пустой - проси переформулировать вопрос
        Добавляй в ответ вырезку из документа с номером страницы
        Образец вопроса:
        Образец ответа:
        Контекст: {context_str}
        Вопрос: {query_str}
        Ответ:
        """
    )

    query_engine.update_prompts({"response_synthesizer:text_qa_template": PromptTemplate(promt)})

    return query_engine

# пробуем создать индекс из векторизованной базы. Если не получается, значит запуск первый и нужно её создать
try:
    index = load_vectorized()
except Exception as e:
    print(e.args)
    print("БД не найдена. Создаём из файлов папки data")
    save_vectorized()
    index = load_vectorized()
'''

agent = agent()





load_dotenv()

# Логика работы через бота
bot = telebot.TeleBot(os.getenv("tg_token"))

@bot.message_handler(commands=["start", "help"])
def send_welcom(message: Message):
    bot.send_message(message.chat.id, "Здравствуйте. Я Атомбот. Я помогаю Вам")

user_message_history = {}
# query_engine = request_settings(index)

@bot.message_handler(content_types=["text"])
def message_handler(message:Message):
    chat_id = message.chat.id
    user_id = message.from_user.id

    bot.send_chat_action(chat_id, "typing")

    msg = message.text
    response = agent.chat(msg)
    reply = response.response

    bot.send_message(chat_id, reply)


bot.infinity_polling()
