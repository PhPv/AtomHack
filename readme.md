# Запуск:

1. Установить зависимости

```pip instal -r requirements.txt```

2. Создать файл .env в корне проекта

3. Запуск версии с UI на chainlit

    3.1 Для сервинга модели через together пройти регистрацию и получить ключ с бесплатным демо периодом
    https://api.together.xyz/settings/api-keys и Прописать его в .env переменной together_api 
    
    3.2 Для сервинга локальной модели убрать комментирование llamaCPP

    3.3 При первой загрузке должны скачать эмбединги с huggingface и llm (если выбрана LlamaCPP)

```chainlit run atom_chainlit.py -w```

3. Запуск версии на telegram

    3.1 Перед запуском через [botfath](https://t.me/BotFather) создать нового бота 

    3.2 Прописать tg_token в .env
    
```python3 atom_tg.py```
    