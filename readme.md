# Запуск:

1. Установить зависимости

```pip instal -r requirements.txt```

2. Запуск версии с UI на chainlit

    2.1 Для сервинга модели через together пройти регистрацию и получить ключ с бесплатным демо периодом
    https://api.together.xyz/settings/api-keys

    2.2 Для сервинга локальной модели убрать комментирование llamaCPP

    2.3 При первой загрузке должны скачать эмбединги с huggingface и llm (если выбрана LlamaCPP)

```chainlit run atom_chainlit.py -w```

3. Запуск версии на telegram

    3.1 Перед запуском 
```python3 atom_tg.py```
    