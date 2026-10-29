"""RAG-агент: Pinecone + tools + память пользователя + чтение URL."""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit, urlunsplit
from uuid import uuid4

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import InMemorySaver
from pinecone import Pinecone, ServerlessSpec

load_dotenv(Path(__file__).resolve().parent / ".env")

URL_PATTERN = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)
MAX_URL_CHUNKS = 40  # чтобы большие статьи (Википедия) не срывали таймаут бота

# Эвристики: фразы, по которым стоит запомнить факт о пользователе.
MEMORY_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bзапомни\b",
        r"\bremember\b",
        r"\bменя зовут\b",
        r"\bмое имя\b",
        r"\bмоё имя\b",
        r"\bя живу\b",
        r"\bя из\b",
        r"\bя работаю\b",
        r"\bя учусь\b",
        r"\bя изучаю\b",
        r"\bмне нравится\b",
        r"\bя люблю\b",
        r"\bмой любимый\b",
        r"\bмоя любимая\b",
        r"\bмне\s+\d+\s*лет\b",
        r"\bя\s+\d+\s*лет\b",
        r"\bмой telegram\b",
        r"\bмой email\b",
        r"\bмоя профессия\b",
        r"\bя программист\b",
        r"\bя разработчик\b",
        r"\bпредпочитаю\b",
    )
]

SYSTEM_PROMPT = """Ты умный помощник с долговременной памятью и инструментами.

## Возможности
- `search_knowledge_base`: поиск по векторной базе (документы и память пользователя).
- `remember_user_fact`: сохранить важный факт о пользователе в память.
- `add_knowledge`: добавить произвольный текст/заметку в базу знаний.
- `ingest_url_to_memory`: скачать HTML/текст по URL, разбить на чанки и сохранить.
- `get_random_zodiac_fact`: GET-запрос случайного факта о созвездии Зодиака.

## Правила
1. В сообщении уже может быть блок «Контекст из векторной базы» — используй его.
   Вызывай `search_knowledge_base` только если этого контекста явно недостаточно.
2. Факты о пользователе и URL часто уже сохранены до твоего ответа — не дублируй без нужды.
3. Не выдумывай содержимое документов: опирайся на контекст и tools.
4. Если пользователь просит факт/интересное про Зодиак или созвездия знаков — вызови `get_random_zodiac_fact`.
5. Отвечай кратко и по делу, на языке пользователя. Для простого приветствия tools не нужны.
"""

ZODIAC_API_URL = "https://birthdex.net/api/zodiac.json"

ZODIAC_NAMES_RU = {
    "aries": "Овен",
    "taurus": "Телец",
    "gemini": "Близнецы",
    "cancer": "Рак",
    "leo": "Лев",
    "virgo": "Дева",
    "libra": "Весы",
    "scorpio": "Скорпион",
    "sagittarius": "Стрелец",
    "capricorn": "Козерог",
    "aquarius": "Водолей",
    "pisces": "Рыбы",
}


class RAGAgent:
    """RAG-агент для Telegram-бота: поиск, память, URL, tools."""

    SHARED_NAMESPACE = "shared"

    def __init__(
        self,
        *,
        index_name: str | None = None,
        chat_model: str | None = None,
        embedding_model: str | None = None,
        top_k: int = 5,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
    ) -> None:
        self.openai_api_key = os.getenv("OPENAI_KEY") or os.getenv("OPENAI_API_KEY")
        self.openai_base_url = os.getenv("OPENAI_BASE_URL")
        self.pinecone_api_key = (
            os.getenv("PINICONE_TOKEN")
            or os.getenv("PINECONE_API_KEY")
            or os.getenv("PINECONE_TOKEN")
        )

        if not self.openai_api_key:
            raise ValueError("Не найден OPENAI_KEY / OPENAI_API_KEY в .env")
        if not self.openai_base_url:
            raise ValueError("Не найден OPENAI_BASE_URL в .env")
        if not self.pinecone_api_key:
            raise ValueError("Не найден PINICONE_TOKEN / PINECONE_API_KEY в .env")

        os.environ.setdefault("OPENAI_API_KEY", self.openai_api_key)
        os.environ.setdefault("PINECONE_API_KEY", self.pinecone_api_key)

        self.index_name = index_name or os.getenv("PINECONE_INDEX_NAME", "rag-agent")
        self.chat_model_name = chat_model or os.getenv("CHAT_MODEL", "gpt-4o-mini")
        self.embedding_model_name = embedding_model or os.getenv(
            "EMBEDDING_MODEL", "text-embedding-3-small"
        )
        self.top_k = top_k
        self.embedding_dimension = 1536
        self._current_user_id: str = "default"

        self.embeddings = OpenAIEmbeddings(
            model=self.embedding_model_name,
            api_key=self.openai_api_key,
            base_url=self.openai_base_url,
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        self.pc = Pinecone(api_key=self.pinecone_api_key)
        self._ensure_index()
        self.index = self.pc.Index(self.index_name)

        self.llm = init_chat_model(
            self.chat_model_name,
            model_provider="openai",
            api_key=self.openai_api_key,
            base_url=self.openai_base_url,
            temperature=0.2,
            timeout=90,
            max_tokens=2000,
        )
        self.checkpointer = InMemorySaver()
        self.tools = self._build_tools()
        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=SYSTEM_PROMPT,
            checkpointer=self.checkpointer,
        )

    @staticmethod
    def user_namespace(user_id: str | int) -> str:
        return f"user_{user_id}"

    def _ensure_index(self) -> None:
        existing = {item["name"] for item in self.pc.list_indexes()}
        if self.index_name in existing:
            return

        self.pc.create_index(
            name=self.index_name,
            dimension=self.embedding_dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not self.pc.describe_index(self.index_name).status["ready"]:
            time.sleep(1)

    def _build_tools(self) -> list[Any]:
        agent = self

        @tool
        def search_knowledge_base(query: str) -> str:
            """Ищет релевантные фрагменты в памяти пользователя и общей базе знаний."""
            hits = agent.similarity_search(
                query,
                k=agent.top_k,
                user_id=agent._current_user_id,
            )
            if not hits:
                return "В базе ничего релевантного не найдено."
            parts = []
            for i, hit in enumerate(hits, start=1):
                kind = hit.get("kind") or "unknown"
                source = hit.get("source") or "unknown"
                score = hit.get("score")
                score_txt = f"{score:.3f}" if isinstance(score, float) else "n/a"
                parts.append(
                    f"[{i}] kind={kind} score={score_txt} source={source}\n{hit['text']}"
                )
            return "\n\n".join(parts)

        @tool
        def remember_user_fact(fact: str) -> str:
            """Сохраняет важный факт о текущем пользователе в векторную память."""
            try:
                record_id = agent.add_user_memory(agent._current_user_id, fact)
            except Exception as exc:  # noqa: BLE001
                return f"Не удалось сохранить факт: {exc}"
            return f"Факт сохранён (id={record_id})."

        @tool
        def add_knowledge(text: str, source: str = "manual") -> str:
            """Добавляет произвольный текст/заметку в общую базу знаний."""
            try:
                count = agent.add_texts([text], source=source, kind="note")
            except Exception as exc:  # noqa: BLE001
                return f"Не удалось добавить знание: {exc}"
            return f"Добавлено фрагментов: {count}."

        @tool
        def ingest_url_to_memory(url: str) -> str:
            """Скачивает страницу по URL, делает чанки, эмбеддинги и пишет в Pinecone."""
            try:
                count = agent.ingest_url(url, user_id=agent._current_user_id)
            except Exception as exc:  # noqa: BLE001
                return f"Не удалось обработать URL: {exc}"
            return f"Страница сохранена в память: {count} чанков из {url}"

        @tool
        def get_random_zodiac_fact() -> str:
            """Делает GET-запрос и возвращает случайный факт о созвездии Зодиака."""
            try:
                return agent.fetch_random_zodiac_fact()
            except Exception as exc:  # noqa: BLE001
                return f"Не удалось получить факт о Зодиаке: {exc}"

        return [
            search_knowledge_base,
            remember_user_fact,
            add_knowledge,
            ingest_url_to_memory,
            get_random_zodiac_fact,
        ]

    def similarity_search(
        self,
        query: str,
        k: int | None = None,
        *,
        user_id: str | int | None = None,
    ) -> list[dict[str, Any]]:
        """Поиск по namespace пользователя и общей базе."""
        top_k = k or self.top_k
        vector = self.embeddings.embed_query(query)
        namespaces = [self.SHARED_NAMESPACE]
        if user_id is not None:
            namespaces.append(self.user_namespace(user_id))

        merged: list[dict[str, Any]] = []
        for ns in namespaces:
            response = self.index.query(
                vector=vector,
                top_k=top_k,
                include_metadata=True,
                namespace=ns,
            )
            for match in response.get("matches", []):
                metadata = match.get("metadata") or {}
                text = metadata.get("text") or ""
                if not text:
                    continue
                merged.append(
                    {
                        "id": match.get("id"),
                        "score": match.get("score"),
                        "text": text,
                        "source": metadata.get("source"),
                        "kind": metadata.get("kind"),
                        "namespace": ns,
                        "metadata": metadata,
                    }
                )

        merged.sort(key=lambda item: item.get("score") or 0.0, reverse=True)
        return merged[:top_k]

    def upsert_records(
        self,
        records: list[dict[str, Any]],
        *,
        namespace: str,
    ) -> int:
        batch_size = 100
        for start in range(0, len(records), batch_size):
            self.index.upsert(
                vectors=records[start : start + batch_size],
                namespace=namespace,
            )
        return len(records)

    def add_texts(
        self,
        texts: list[str],
        *,
        source: str = "manual",
        kind: str = "note",
        user_id: str | int | None = None,
        namespace: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> int:
        clean = [t.strip() for t in texts if t and t.strip()]
        if not clean:
            return 0

        vectors = self.embeddings.embed_documents(clean)
        ns = namespace or (
            self.user_namespace(user_id) if user_id is not None else self.SHARED_NAMESPACE
        )
        records = []
        for text, vector in zip(clean, vectors):
            meta = {
                "text": text[:3500],
                "source": source,
                "kind": kind,
            }
            if user_id is not None:
                meta["user_id"] = str(user_id)
            if extra_metadata:
                meta.update(extra_metadata)
            records.append(
                {
                    "id": f"{kind}-{uuid4().hex}",
                    "values": vector,
                    "metadata": meta,
                }
            )
        return self.upsert_records(records, namespace=ns)

    def add_user_memory(self, user_id: str | int, fact: str) -> str:
        """Сохраняет факт о пользователе в его namespace."""
        fact = fact.strip()
        if not fact:
            raise ValueError("Пустой факт")
        vectors = self.embeddings.embed_documents([fact])
        record_id = f"memory-{user_id}-{uuid4().hex[:12]}"
        self.upsert_records(
            [
                {
                    "id": record_id,
                    "values": vectors[0],
                    "metadata": {
                        "text": fact[:3500],
                        "source": "user_message",
                        "kind": "user_fact",
                        "user_id": str(user_id),
                    },
                }
            ],
            namespace=self.user_namespace(user_id),
        )
        return record_id

    @classmethod
    def extract_memory_candidates(cls, text: str) -> list[str]:
        """Эвристика: вытаскивает предложения/фразы, похожие на факты о пользователе."""
        if not text or not text.strip():
            return []

        candidates: list[str] = []
        # Явная команда «запомни ...»
        remember_match = re.search(
            r"(?:запомни|remember)\s*[:\-]?\s*(.+)",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if remember_match:
            fact = remember_match.group(1).strip()
            if fact:
                candidates.append(fact)

        sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
        for sentence in sentences:
            s = sentence.strip()
            if len(s) < 8:
                continue
            if any(pattern.search(s) for pattern in MEMORY_PATTERNS):
                if s not in candidates:
                    candidates.append(s)

        return candidates[:5]

    @staticmethod
    def normalize_url(url: str) -> str:
        """Percent-encode path/query (нужно для кириллических URL вроде Википедии)."""
        cleaned = url.strip().rstrip(".,;:!?)»\"'")
        parts = urlsplit(cleaned)
        # Не трогаем уже закодированные %XX-последовательности.
        path = quote(parts.path, safe="/%")
        query = quote(parts.query, safe="=&%+")
        fragment = quote(parts.fragment, safe="%")
        return urlunsplit((parts.scheme, parts.netloc.encode("idna").decode("ascii"), path, query, fragment))

    def fetch_random_zodiac_fact(self) -> str:
        """GET https://birthdex.net/api/zodiac.json → случайный знак Зодиака."""
        req = urllib.request.Request(
            ZODIAC_API_URL,
            headers={"User-Agent": "Mozilla/5.0 (compatible; rag-agent/1.0)"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"GET {ZODIAC_API_URL} failed: {exc}") from exc

        signs = payload.get("signs") or []
        if not signs:
            raise RuntimeError("API вернул пустой список знаков Зодиака")

        sign = random.choice(signs)
        slug = str(sign.get("slug") or "").lower()
        name_en = sign.get("name") or slug.title() or "Unknown"
        name_ru = ZODIAC_NAMES_RU.get(slug, name_en)
        symbol = sign.get("symbol") or ""
        date_range = sign.get("dateRange") or "—"
        element = sign.get("element") or "—"
        modality = sign.get("modality") or "—"
        planet = sign.get("rulingPlanet") or "—"
        birthstone = sign.get("birthstone") or "—"

        return (
            f"Случайный факт о созвездии Зодиака {name_ru} ({name_en}) {symbol}\n"
            f"• Период: {date_range}\n"
            f"• Стихия: {element}\n"
            f"• Модальность: {modality}\n"
            f"• Управляющая планета: {planet}\n"
            f"• Камень: {birthstone}\n"
            f"Источник: {ZODIAC_API_URL}"
        )

    def fetch_url_text(self, url: str) -> str:
        """Скачивает URL и возвращает очищенный текст страницы."""
        safe_url = self.normalize_url(url)
        req = urllib.request.Request(
            safe_url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/122.0.0.0 Safari/537.36"
                ),
                "Accept-Language": "ru,en;q=0.9",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read()
                content_type = resp.headers.get("Content-Type", "")
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Fetch failed: {exc}") from exc
        except UnicodeEncodeError as exc:
            raise RuntimeError(f"URL encoding failed: {exc}") from exc

        text = raw.decode("utf-8", errors="replace")
        if "html" in content_type.lower() or "<html" in text[:500].lower():
            soup = BeautifulSoup(text, "html.parser")
            for tag in soup(
                [
                    "script",
                    "style",
                    "noscript",
                    "iframe",
                    "nav",
                    "footer",
                    "header",
                ]
            ):
                tag.decompose()
            # Для Википедии/статей берём основной контент, без меню и служебных блоков.
            main = (
                soup.select_one("#mw-content-text .mw-parser-output")
                or soup.select_one("#mw-content-text")
                or soup.select_one("article")
                or soup.select_one("main")
                or soup.body
                or soup
            )
            for selector in (
                ".navbox",
                ".vertical-navbox",
                ".mw-editsection",
                ".reference",
                ".noprint",
                "#toc",
                ".toc",
                ".catlinks",
                ".mw-authority-control",
                "table.infobox",
            ):
                for node in main.select(selector):
                    node.decompose()
            text = main.get_text(separator="\n")

        lines = [line.strip() for line in text.splitlines()]
        # Убираем очень короткие «кнопочные» строки навигации.
        lines = [line for line in lines if line and (len(line) > 2)]
        return "\n".join(lines)

    def ingest_url(self, url: str, *, user_id: str | int | None = None) -> int:
        """Парсит URL → чанки → эмбеддинги → upsert в shared namespace."""
        safe_url = self.normalize_url(url)
        page_text = self.fetch_url_text(safe_url)
        if not page_text.strip():
            raise RuntimeError("Страница пустая или не удалось извлечь текст")

        chunks = self.text_splitter.split_text(page_text)
        if not chunks:
            raise RuntimeError("После разбиения не осталось чанков")

        total_chunks = len(chunks)
        chunks = chunks[:MAX_URL_CHUNKS]

        # Эмбеддим батчами — надёжнее на ProxyAPI для длинных статей.
        vectors: list[list[float]] = []
        batch_size = 16
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            vectors.extend(self.embeddings.embed_documents(batch))

        url_hash = hashlib.sha1(safe_url.encode("utf-8")).hexdigest()[:12]
        records = []
        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            meta: dict[str, Any] = {
                "text": chunk[:3500],
                "source": safe_url,
                "kind": "url_chunk",
                "chunk_index": i,
                "total_chunks": total_chunks,
            }
            if user_id is not None:
                meta["user_id"] = str(user_id)
            records.append(
                {
                    "id": f"{url_hash}-{i}-{uuid4().hex[:8]}",
                    "values": vector,
                    "metadata": meta,
                }
            )

        saved = self.upsert_records(records, namespace=self.SHARED_NAMESPACE)
        if total_chunks > saved:
            # Сигнал вызывающему коду через обычный return; подробность пишется в ask().
            return saved
        return saved

    @staticmethod
    def extract_urls(text: str) -> list[str]:
        urls = []
        for match in URL_PATTERN.findall(text):
            cleaned = match.rstrip(".,;:!?)»\"'")
            if cleaned and cleaned not in urls:
                urls.append(cleaned)
        return urls

    def _format_hits(self, hits: list[dict[str, Any]]) -> str:
        if not hits:
            return "Ничего не найдено."
        parts = []
        for i, hit in enumerate(hits, start=1):
            kind = hit.get("kind") or "unknown"
            source = hit.get("source") or "unknown"
            score = hit.get("score")
            score_txt = f"{score:.3f}" if isinstance(score, float) else "n/a"
            parts.append(
                f"[{i}] kind={kind} score={score_txt} source={source}\n{hit['text']}"
            )
        return "\n\n".join(parts)

    @staticmethod
    def _message_to_text(content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    texts.append(block.get("text") or "")
                elif isinstance(block, str):
                    texts.append(block)
                else:
                    text = getattr(block, "text", None)
                    if text:
                        texts.append(str(text))
            return "\n".join(t for t in texts if t) or str(content)
        return str(content)

    def _answer_with_llm(self, user_content: str) -> str:
        """Запасной путь без tool-loop — если агент завис/упал."""
        result = self.llm.invoke(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]
        )
        return self._message_to_text(getattr(result, "content", result)).strip()

    def ask(
        self,
        message: str,
        *,
        user_id: str | int = "default",
        thread_id: str | None = None,
    ) -> str:
        """
        Главный вход для бота:
        1) эвристика памяти пользователя
        2) ingest URL
        3) препоиск в векторной БД
        4) ответ агента с tools (с fallback на обычный LLM)
        """
        self._current_user_id = str(user_id)
        thread = thread_id or f"tg-{user_id}"
        notes: list[str] = []

        # 1. Эвристическое запоминание фактов о пользователе.
        for fact in self.extract_memory_candidates(message):
            try:
                record_id = self.add_user_memory(user_id, fact)
                notes.append(f"Запомнен факт о пользователе ({record_id}): {fact}")
            except Exception as exc:  # noqa: BLE001
                notes.append(f"Не удалось запомнить факт: {exc}")

        # 2. URL → чанки → Pinecone.
        for url in self.extract_urls(message):
            try:
                count = self.ingest_url(url, user_id=user_id)
                notes.append(
                    f"URL {url} сохранён ({count} чанков, лимит {MAX_URL_CHUNKS})."
                )
            except Exception as exc:  # noqa: BLE001
                notes.append(f"URL {url} не удалось сохранить: {exc}")

        # 3. Препоиск перед ответом (мягко: при сбое API всё равно отвечаем).
        try:
            pre_hits = self.similarity_search(message, k=self.top_k, user_id=user_id)
            context_block = self._format_hits(pre_hits)
        except Exception as exc:  # noqa: BLE001
            context_block = f"Препоиск недоступен: {exc}"

        user_content = (
            "Контекст из векторной базы (уже найден до ответа):\n"
            f"{context_block}\n\n"
        )
        if notes:
            user_content += (
                "Служебные действия до ответа:\n"
                + "\n".join(notes)
                + "\n\n"
            )
        user_content += f"Сообщение пользователя:\n{message}"

        try:
            result = self.agent.invoke(
                {"messages": [{"role": "user", "content": user_content}]},
                config={
                    "configurable": {"thread_id": thread},
                    "recursion_limit": 12,
                },
            )
            last = result["messages"][-1]
            answer = self._message_to_text(getattr(last, "content", last)).strip()
            if answer:
                return answer
        except Exception:
            # Агент/tools могли упасть или зависнуть на стороне API — отвечаем напрямую.
            pass

        return self._answer_with_llm(user_content) or "Готово, но текст ответа пустой. Попробуйте ещё раз."

    def test_connection(self, query: str = "тестовый запрос к базе знаний") -> dict[str, Any]:
        """Проверка, что эмбеддинги и Pinecone отвечают."""
        hits = self.similarity_search(query, k=1, user_id="default")
        return {
            "ok": True,
            "index": self.index_name,
            "query": query,
            "hits": len(hits),
            "sample": hits[0] if hits else None,
        }


if __name__ == "__main__":
    print("Инициализация RAGAgent...", flush=True)
    rag = RAGAgent()
    print(f"Индекс Pinecone: {rag.index_name}", flush=True)
    print("Тестовый поиск в векторной БД...", flush=True)
    report = rag.test_connection()
    print("Подключение OK:" if report["ok"] else "Подключение FAIL:", report, flush=True)
