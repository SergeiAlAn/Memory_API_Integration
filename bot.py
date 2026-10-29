"""Telegram-интерфейс умного RAG-помощника на PyTelegramBotAPI."""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from pathlib import Path

import telebot
from dotenv import load_dotenv
from telebot import types

from rag_agent import RAGAgent

load_dotenv(Path(__file__).resolve().parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("rag-bot")

BOT_TOKEN = os.getenv("BOT_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN")
if not BOT_TOKEN:
    raise ValueError("Не найден BOT_TOKEN в .env")

ASK_TIMEOUT_SEC = int(os.getenv("ASK_TIMEOUT_SEC", "120"))

HELP_TEXT = (
    "Я умный помощник с памятью.\n\n"
    "Умею:\n"
    "• отвечать на вопросы\n"
    "• искать данные в векторной базе перед ответом\n"
    "• запоминать важные факты о вас (эвристика + команда «запомни …»)\n"
    "• читать HTML-страницы по URL и сохранять их в память\n"
    "• добавлять заметки через агента\n\n"
    "Примеры:\n"
    "• Меня зовут Алекс, я изучаю Python\n"
    "• Запомни: я предпочитаю короткие ответы\n"
    "• Прочитай https://example.com и кратко перескажи\n"
    "• Что ты обо мне помнишь?"
)

bot = telebot.TeleBot(BOT_TOKEN, threaded=True)
executor = ThreadPoolExecutor(max_workers=4)

logger.info("Инициализация RAGAgent...")
rag = RAGAgent()
logger.info("RAGAgent готов, индекс=%s", rag.index_name)


def _split_message(text: str, limit: int = 4000) -> list[str]:
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    current = ""
    for line in text.splitlines(keepends=True):
        if len(current) + len(line) > limit:
            if current:
                chunks.append(current)
            current = line
        else:
            current += line
    if current:
        chunks.append(current)
    return chunks or [text[:limit]]


def _safe_send(chat_id: int, text: str, *, reply_to: int | None = None) -> None:
    for part in _split_message(text):
        try:
            bot.send_message(chat_id, part, reply_to_message_id=reply_to)
        except telebot.apihelper.ApiTelegramException:
            logger.exception("Не удалось отправить сообщение в chat_id=%s", chat_id)
            try:
                bot.send_message(chat_id, part)
            except Exception:
                logger.exception("Повторная отправка тоже не удалась")


def _update_status(chat_id: int, message_id: int, text: str) -> None:
    try:
        bot.edit_message_text(text, chat_id=chat_id, message_id=message_id)
    except telebot.apihelper.ApiTelegramException as exc:
        logger.warning("edit_message_text failed: %s", exc)


@bot.message_handler(commands=["start", "help"])
def handle_start(message: types.Message) -> None:
    bot.reply_to(message, f"Привет! {HELP_TEXT}")


@bot.message_handler(commands=["memory"])
def handle_memory(message: types.Message) -> None:
    user_id = message.from_user.id
    bot.send_chat_action(message.chat.id, "typing")
    try:
        hits = rag.similarity_search(
            "факты о пользователе имя предпочтения работа учёба",
            k=8,
            user_id=user_id,
        )
    except Exception:
        logger.exception("Ошибка /memory user=%s", user_id)
        bot.reply_to(message, "Не удалось прочитать память. Попробуйте позже.")
        return

    user_hits = [h for h in hits if h.get("kind") == "user_fact"] or hits
    if not user_hits:
        bot.reply_to(
            message,
            "Пока ничего не запомнил. Расскажите о себе или напишите «запомни …».",
        )
        return

    lines = ["Что нашлось в памяти:"]
    for i, hit in enumerate(user_hits[:8], start=1):
        lines.append(f"{i}. {hit['text']}")
    bot.reply_to(message, "\n".join(lines))


@bot.message_handler(content_types=["text"])
def handle_text(message: types.Message) -> None:
    text = (message.text or "").strip()
    if not text:
        return

    user_id = message.from_user.id
    chat_id = message.chat.id
    logger.info("user=%s text=%r", user_id, text[:200])

    try:
        status = bot.reply_to(message, "Думаю: ищу в памяти и готовлю ответ…")
    except telebot.apihelper.ApiTelegramException:
        logger.exception("Не удалось отправить статус")
        status = None

    bot.send_chat_action(chat_id, "typing")

    future = executor.submit(rag.ask, text, user_id=user_id)
    try:
        answer = future.result(timeout=ASK_TIMEOUT_SEC)
        logger.info("user=%s answer_len=%s", user_id, len(answer or ""))
    except FuturesTimeout:
        logger.error("Таймаут ask() user=%s after %ss", user_id, ASK_TIMEOUT_SEC)
        answer = (
            "Слишком долго жду ответ от модели/памяти. "
            "Попробуйте ещё раз чуть позже."
        )
    except Exception:
        logger.exception("Ошибка обработки сообщения user=%s", user_id)
        answer = "Не удалось обработать сообщение. Попробуйте ещё раз."

    if not answer or not str(answer).strip():
        answer = "Получил пустой ответ от модели. Попробуйте переформулировать вопрос."

    if status is not None:
        _update_status(chat_id, status.message_id, str(answer)[:4000])
        rest = _split_message(str(answer))[1:]
        for part in rest:
            _safe_send(chat_id, part)
    else:
        _safe_send(chat_id, str(answer), reply_to=message.message_id)


if __name__ == "__main__":
    logger.info("Бот запущен (polling), timeout ask=%ss", ASK_TIMEOUT_SEC)
    try:
        bot.infinity_polling(skip_pending=True, timeout=60, long_polling_timeout=50)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
