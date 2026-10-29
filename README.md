# API Integration — RAG-агент с Pinecone и Telegram

Учебный проект: умный помощник на LangChain/LangGraph с векторной памятью Pinecone, обработкой URL и вызовом внешнего API. Интерфейс — Telegram-бот (PyTelegramBotAPI).

## Домашнее задание

1. **Настроить базовый RAG-агент:** подключить векторное хранилище Pinecone и выполнить тестовый запрос для проверки интеграции.
2. **Реализовать обработку URL:** парсинг страницы → чанки → эмбеддинги → сохранение в векторную БД.
3. **Модифицировать API-запрос:** заменить заглушку на реальный GET к открытому API и корректно обработать ответ.

## Как проект закрывает требования

| Требование ДЗ | Реализация в проекте |
|---|---|
| RAG + Pinecone + тестовый запрос | Класс `RAGAgent` в `rag_agent.py`: создание/подключение индекса, эмбеддинги через ProxyAPI (`OPENAI_BASE_URL`), поиск `similarity_search`. Точка входа `python rag_agent.py` вызывает `test_connection()`. |
| Обработка URL | `fetch_url_text` (парсинг HTML, в т.ч. Википедия) → `RecursiveCharacterTextSplitter` → эмбеддинги → `ingest_url` / tool `ingest_url_to_memory`. Поддержаны кириллические URL (percent-encoding). |
| Реальный GET вместо заглушки | Tool `get_random_zodiac_fact`: GET `https://birthdex.net/api/zodiac.json`, случайный знак Зодиака, разбор JSON и текстовый ответ. |

Дополнительно (сверх минимального ДЗ): Telegram-бот (`bot.py`), эвристическая память о пользователе, препоиск в Pinecone перед ответом, агент с `@tool`.

## Структура

```
├── rag_agent.py      # RAG-агент: Pinecone, URL, tools, тест подключения
├── bot.py            # Telegram-интерфейс
├── requirements.txt  # зависимости
├── .env.example      # образец переменных окружения
├── LangChain_Example # справочный пример актуального API LangChain
└── README.md
```

## Быстрый старт

1. Создайте виртуальное окружение и установите зависимости:

```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

2. Скопируйте `.env.example` → `.env` и заполните ключи:

- `OPENAI_KEY`, `OPENAI_BASE_URL` — ProxyAPI
- `PINICONE_TOKEN`, `PINECONE_INDEX_NAME` — Pinecone
- `BOT_TOKEN` — токен бота от @BotFather

3. Проверка интеграции с Pinecone (п.1 ДЗ):

```bash
python rag_agent.py
```

Ожидается сообщение вида `Подключение OK` (число hits может быть 0, если индекс пуст).

4. Запуск бота:

```bash
python bot.py
```

### Примеры проверки в боте

- **Память / RAG:** `Привет! Я люблю астрономию`
- **URL (п.2 ДЗ):** `Прочитай https://ru.wikipedia.org/wiki/Плутон и кратко перескажи`
- **GET API (п.3 ДЗ):** `Расскажи случайный факт о Зодиаке`
- **Память пользователя:** `/memory`

## Основные компоненты

### `RAGAgent` (`rag_agent.py`)

- Подключение к Pinecone, namespaces: `shared` (документы/URL) и `user_{id}` (факты о пользователе)
- Эмбеддинги: `OpenAIEmbeddings` с внешним `base_url`
- Чат-модель: `init_chat_model` + агент `create_agent` (как в `LangChain_Example`)
- Tools: `search_knowledge_base`, `remember_user_fact`, `add_knowledge`, `ingest_url_to_memory`, `get_random_zodiac_fact`

### Telegram (`bot.py`)

- Команды: `/start`, `/help`, `/memory`
- Текстовые сообщения → `rag.ask(...)` с таймаутом и устойчивой отправкой ответа

## Примечание по окружению

На Python 3.14 пакет `langchain-pinecone` недоступен (`Requires-Python <3.14`), поэтому работа с Pinecone идёт через официальный SDK `pinecone` — на требования ДЗ это не влияет.
