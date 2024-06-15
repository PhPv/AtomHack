import telebot
from dotenv import load_dotenv
from telebot.types import Message
from qdrant_client import QdrantClient
import os

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine.retriever_query_engine import RetrieverQueryEngine
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, get_response_synthesizer, PromptTemplate
from llama_index.llms.together import TogetherLLM
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.llms.llama_cpp import LlamaCPP


def save_vectorized():
    """
    Разбиваем документы на чанки по 1024 и сохраняем в векторную базу данных qdrant
    """
    docs = SimpleDirectoryReader(f"data").load_data()
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

    return index


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

    promt = (
        """
        Ты помощник службы поддержки в РосАтом. Используй информацию из контекста чтобы помочь пользователям найти ответ на их вопрос.
        Отвечай на русском, будь вежлив, сокращай ответ.
        Если контекст пустой - проси переформулировать вопрос
        Добавляй в ответ вырезку из документа с номером страницы
        Контекст: {context_str}
        Вопрос: {query_str}
        Ответ:
        """
    )

    query_engine.update_prompts({"response_synthesizer:text_qa_template": PromptTemplate(promt)})

    return query_engine


def main():

    load_dotenv()

    # пробуем создать индекс из векторизованной базы. Если не получается, значит запуск первый и нужно её создать
    try:
        index = load_vectorized()
    except Exception as e:
        print(e.args)
        print("БД не найдена. Создаём из файлов папки data")
        save_vectorized()
        index = load_vectorized()


    # Логика работы через бота
    bot = telebot.TeleBot(os.getenv("tg_token"))

    @bot.message_handler(commands=["start", "help"])
    def send_welcome(message: Message):
        bot.send_message(message.chat.id, "Здравствуйте. Я Атомбот. Я помогаю Вам")

    user_message_history = {}
    query_engine = request_settings(index)

    @bot.message_handler(content_types=["text"])
    def message_handler(message:Message):
        chat_id = message.chat.id
        user_id = message.from_user.id

        bot.send_chat_action(chat_id, "typing")

        msg = message.text
        response = query_engine.query(msg)
        reply = response.response

        bot.send_message(chat_id, reply)


    bot.infinity_polling()

main()