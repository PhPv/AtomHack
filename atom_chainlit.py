import chainlit as cl
import os
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, PromptTemplate
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from llama_index.llms.together import TogetherLLM
from llama_index.core.query_engine.retriever_query_engine import RetrieverQueryEngine
# from llama_index.llms.llama_cpp import LlamaCPP




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


def load_vectorized():
    """
    Загружаем из векторной базы данные
    """

    client = QdrantClient(path="db")

    Settings.embed_model = HuggingFaceEmbedding(
        model_name = 'paraphrase-multilingual-mpnet-base-v2'
    )

    vector_store = QdrantVectorStore(collection_name="deep_collection", client=client)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)

    return(index)


@cl.on_chat_start
async def start():

    load_dotenv()
    # пробуем создать индекс из векторизованной базы. Если не получается, значит запуск первый и нужно её создать
    try:
        index = load_vectorized()
    except Exception as e:
        print(e.args)
        print("БД не найдена. Создаём из файлов папки data")
        save_vectorized()
        index = load_vectorized()

    # для быстрого инференса используется Qwen2-72b развернутая на together
    #########################################################
    Settings.llm = TogetherLLM(
        # model="meta-llama/Llama-3-8b-chat-hf",
        model="Qwen/Qwen2-72B-Instruct",
        api_key=os.getenv("together_api"))
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
    

    query_engine = index.as_query_engine(streaming=True, similarity_top_k=5)

    promt = (
        """
        Ты помощник службы поддержки в РосАтом. Используй инфомрацию из контекста чтобы помочь пользователям найти ответ на их вопрос.
        Отвечай на русском, будь вежлив, сокращай ответ.
        Если контекст пустой - проси переформулировать вопрос
        Добавляй в ответ вырезку из документа
        Образец вопроса:
        Образец ответа:
        Контекст: {context_str}
        Вопрос: {query_str}
        Ответ:
        """
    )

    query_engine.update_prompts({"response_synthesizer:text_qa_template": PromptTemplate(promt)})
    cl.user_session.set("query_engine", query_engine)

    await cl.Message(
        author="АтомБот", content="Привет! Я АтомБот. Чем могу помочь?"
    ).send()


@cl.on_message
async def main(message: cl.Message):
    query_engine = cl.user_session.get("query_engine") # type: RetrieverQueryEngine

    msg = cl.Message(content="", author="АтомБот")

    res = await cl.make_async(query_engine.query)(message.content)

    for token in res.response_gen:
        await msg.stream_token(token)
    await msg.send()


