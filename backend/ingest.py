import asyncio
import json
import uuid
import os
from datetime import datetime

import numpy as np
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore

from pydantic import BaseModel, Field, ValidationError
from typing import Optional, Literal

from . import models
from .mcp_client import get_mcp_tools


# ==========================================
#               КОНФИГУРАЦИЯ
# ==========================================
LM_STUDIO_URL = "http://localhost:1234/v1"
GENERATION_MODEL = "qwen/qwen3-4b-2507"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))    # Путь к корню проекта (ai-agent/)
FAISS_INDEX = os.path.join(PROJECT_ROOT, "faiss_index")
BRAIN_FILE = os.path.join(PROJECT_ROOT, "brain.json")

# ==========================================
#         ИНИЦИАЛИЗАЦИЯ КОМПОНЕНТОВ
# ==========================================
print("ОК: Загрузка компонентов BrAIn...")



class IngestSchema(BaseModel):
    """Схема для парсинга входящих знаний"""
    text: str = Field(..., description="Основной смысл, структурированный текст")
    topic: str = Field(..., description="Общая тема (например: ML.classification)")
    subtopic: Optional[str] = Field(None, description="Уточнение темы")
    tags: list[str] = Field(default_factory=list, description="Список тегов")
    level: Literal["beginner", "intermediate", "advanced"] = Field("intermediate", description="Уровень сложности")
    status: Literal["draft"] = "draft"
    related_ids: list[str] = Field(default_factory=list, description="ID связанных записей")

embeddings = models.embeddings

llm = ChatOpenAI(
    base_url=LM_STUDIO_URL,
    api_key="lm-studio",
    model=GENERATION_MODEL,
    temperature=0.5,
    max_tokens=1024
)
mcp_tools = get_mcp_tools()
router_llm = llm.bind_tools(mcp_tools)

output_parser = StrOutputParser()

print("ОК: Компоненты BrAIn...")

# ==========================================
#                  ПРОМПТЫ
# ==========================================

PARSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Ты — парсер знаний. Извлеки информацию из сообщения пользователя.
Верни ТОЛЬКО валидный JSON без markdown и пояснений. Если информации нет, верни пустой объект {{}}.

Требуемая структура JSON:
{{ 
  "text": "основная мысль, можно структурировать, но не удалять важные детали",
  "topic": "общая тема, например: ML.classification.RandomForest (макс. 2 уровня, через точку)",
  "subtopic": "уточнение темы, можно микс русского и английского",
  "tags": ["тег1", "тег2", "тег3"],
  "level": "beginner|intermediate|advanced",
  "status": "draft"
}}"""),
    ("human", "{input}"),
])

VALIDATE_PROMPT = ChatPromptTemplate.from_messages([
("system", """Ты — редактор базы знаний. Сравни НОВЫЙ текст с СУЩЕСТВУЮЩИМИ записями.

Если НОВЫЙ текст просто перефразирует существующий (смысл тот же) -> 'duplicate'
Если НОВЫЙ текст добавляет детали, уточнения или факты, которых нет в старых -> 'complement'  
Если НОВЫЙ текст о совершенно другом -> 'new'

Ответь ТОЛЬКО одним словом: duplicate, complement или new."""),

("human", """СУЩЕСТВУЮЩИЕ ЗАПИСИ:
{context}

НОВЫЙ ТЕКСТ:
"{new_text}"

Решение:""")
])


# ==========================================
#                  CHAINS
# ==========================================

parse_chain = PARSE_PROMPT | llm.with_structured_output(IngestSchema)
validate_chain = VALIDATE_PROMPT | llm | output_parser


# ==========================================
#         ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==========================================

def load_brain() -> list:
    """Загружает brain.json или возвращает пустой список"""
    try:
        with open(BRAIN_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_brain(entries: list):
    """Сохраняет в brain.json"""
    with open(BRAIN_FILE, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)


def rebuild_faiss_index() -> FAISS:
    brain = load_brain()

    index = faiss.IndexFlatIP(768)  # 768 = размерность all-mpnet-base-v2

    # 2. Создаём пустой FAISS-объект с этим индексом
    db = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore({}),
        index_to_docstore_id={}
    )

    if not brain:
        print("⚠️ brain.json пуст. Возвращаю пустой индекс.")
        return db

    # 3. Готовим данные
    texts = [item["text"] for item in brain]
    metadatas = [{"id": item["id"], "topic": item.get("topic", "")} for item in brain]
    ids = [item["id"] for item in brain]

    print("🔄 Векторизация + добавление в индекс...")

    # 4. Векторизуем ВРУЧНУЮ (чтобы контролировать нормализацию)
    vectors = embeddings.embed_documents(texts)
    vectors_np = np.array(vectors, dtype=np.float32)

    faiss.normalize_L2(vectors_np)

    # 6. Добавляем векторы в индекс напрямую (быстро, без лишней абстракции)
    db.index.add(vectors_np)

    # 7. Синхронизируем docstore и index_to_docstore_id (это делает LangChain внутри add_texts, но мы делаем вручную для контроля)
    from langchain_core.documents import Document
    for i, (text, meta, uid) in enumerate(zip(texts, metadatas, ids)):
        db.docstore._dict[uid] = Document(page_content=text, metadata=meta)
        db.index_to_docstore_id[i] = uid

    db.save_local(FAISS_INDEX)
    print("✅ Индекс перестроен (Cosine/IP, dim=768)")
    return db


def find_similar_lc(query_text: str, db: FAISS, threshold: float = 0.75, top_k: int = 5) -> list:
    if db.index.ntotal == 0:
        print("⚠️ Индекс пуст. Поиск невозможен.")
        return []

    actual_k = min(top_k, db.index.ntotal)
    docs_with_scores = db.similarity_search_with_score(query_text, k=actual_k)

    print(f"\nТОП-{len(docs_with_scores)} наиболее релевантных:")
    similar = []
    for doc, score in docs_with_scores:
        preview = doc.page_content[:60].replace('\n', ' ').strip()
        print(f"  • [{score:.3f}] {preview}...")

        if score > threshold:
            similar.append({
                "id": doc.metadata.get("id"),
                "similarity": score,
                "text": doc.page_content
            })
    print("-" * 50)
    return sorted(similar, key=lambda x: x["similarity"], reverse=True)


async def unpack_input_async(raw_input: str) -> str:
    """Асинхронная версия роутера. Вызывает MCP-инструменты через await."""
    system_prompt = """Ты — помощник по подготовке данных.
Если ввод выглядит как имя файла (например: 'lecture.pdf', 'notes.docx', '1.txt'), 
вызови соответствующий инструмент (read_pdf, read_docx, read_text).
Если это обычный текст для сохранения в базу — просто верни его в поле 'extracted_text'."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=raw_input)
    ]

    response = await router_llm.ainvoke(messages)

    if hasattr(response, 'tool_calls') and response.tool_calls:
        tool_call = response.tool_calls[0]
        tool = next((t for t in mcp_tools if t.name == tool_call['name']), None)
        if tool:
            print(f"🔧 Вызываю инструмент: {tool_call['name']}({tool_call['args']})")
            result = await tool.ainvoke(tool_call['args'])
            if isinstance(result, str) and result.startswith("❌"):
                raise ValueError(result)
            return result

    return response.content if hasattr(response, 'content') else str(response)


def decide_with_llm_lc(new_text: str, existing_entries: list) -> str:
    """LangChain-версия валидации через LLM"""
    context = "\n".join([f"- {e['text']}" for e in existing_entries])

    try:
        response = validate_chain.invoke({
            "context": context,
            "new_text": new_text
        })
        decision = response.strip().lower()
        return decision if decision in ["duplicate", "complement", "new"] else "new"
    except Exception as e:
        print(f"⚠️ Ошибка валидации: {e}")
        return "new"


def save_knowledge_lc(parsed: IngestSchema, db: FAISS) -> str:
    """
    Сохраняет запись в brain.json и обновляет FAISS индекс.
    Возвращает ID новой записи.
    """
    entry_id = str(uuid.uuid4())

    # Схема записи в brain
    entry = {
        "id": entry_id,
        "text": parsed.text,
        "topic": parsed.topic,
        "subtopic": parsed.subtopic,
        "tags": parsed.tags,
        "level": parsed.level,
        "status": parsed.status,
        "created": datetime.now().isoformat(),
        "related_ids": parsed.related_ids
    }

    # Сохраняем в brain.json
    brain = load_brain()
    brain.append(entry)
    save_brain(brain)

    # Добавляем в векторный индекс (FAISS)
    db.add_texts(
        texts=[parsed.text],
        metadatas=[{"id": entry_id, "topic": parsed.topic}],
        ids=[entry_id]
    )
    db.save_local(FAISS_INDEX)

    print(f"OK: Сохранено: {entry['text'][:40]}...")
    return entry_id


# ==========================================
#                 Контракт
# ==========================================

# Helper к основному контракту
async def process_ingest_async(raw_input: str) -> dict:
    """Асинхронная обёртка для FastAPI. Логика инжеста та же, только распаковка async."""
    print(f'raw_input: {raw_input}')
    if not raw_input or not raw_input.strip():
        return {"status": "error", "message": "Пустой ввод", "data": None}

    try:
        print(f"🔍 Анализирую ввод: {raw_input[:50]}...")
        # 🔹 Async-распаковка
        text = await unpack_input_async(raw_input)
        print(f'text: {text}, type: {type(text)}')
        text = text[0]['text']

        if not text or len(text.strip()) < 10:
            return {"status": "error", "message": "Не удалось извлечь текст", "data": None}

        print(f"✅ Извлечено {len(text)} символов. Передаю в пайплайн...")

        # 🔹 Далее вызываем твою старую sync-функцию, она отлично работает внутри async
        return process_ingest(text)

    except Exception as e:
        print(f"❌ Async Ingest Error: {e}")
        return {"status": "error", "message": str(e), "data": None}


def process_ingest(text: str) -> dict:
    if not text or not text.strip():
        return {"status": "error", "message": "Пустой ввод", "data": None}

    try:
        parsed = parse_chain.invoke({"input": text})
        if not parsed or not parsed.text:
            return {"status": "error", "message": "Модель не смогла извлечь текст", "data": None}

        if not os.path.exists(FAISS_INDEX):
            db = rebuild_faiss_index()
        else:
            db = FAISS.load_local(FAISS_INDEX, embeddings, allow_dangerous_deserialization=True)

        similar = find_similar_lc(parsed.text, db, threshold=0.75)
        best_match = None

        if not similar:
            decision = "new"
        else:
            best_match = similar[0]
            score = best_match['similarity']

            if score > 0.90:
                decision = decide_with_llm_lc(parsed.text, similar)
            elif score > 0.75:
                decision = "complement"
                if not parsed.related_ids:
                    parsed.related_ids = []
                parsed.related_ids.append(best_match["id"])
            else:
                decision = "new"

        if decision == "duplicate":
            return {
                "status": "duplicate",
                "message": "Это похоже на дубликат. Запись пропущена.",
                "data": {"match": best_match["text"][:50] + "..."}
            }

        entry_id = save_knowledge_lc(parsed, db)

        return {
            "status": "success",
            "message": f"Запись сохранена ({decision}).",
            "data": {
                "id": entry_id,
                "topic": parsed.topic,
                "decision": decision,
                "related_to": best_match["text"][:50] + "..." if best_match and decision == "complement" else None
            }
        }

    except ValidationError as e:
        return {"status": "error", "message": f"Ошибка структуры данных: {e}", "data": None}
    except Exception as e:
        print(f"❌ Ingest Error: {e}")
        return {"status": "error", "message": str(e), "data": None}


# ==========================================
#                   main
# ==========================================

# def main():
#     print("🧠 BrAIn Ingest (LangChain Edition)")
#     print("Вводи текст (или 'q' для выхода)")
#     print("-" * 30)
#
#     # Загружаем векторы
#     if not os.path.exists(FAISS_INDEX):
#         db = rebuild_faiss_index()
#     else:
#         db = FAISS.load_local(FAISS_INDEX, embeddings, allow_dangerous_deserialization=True)
#         print("OK: Существующий FAISS-индекс загружен.")
#
#     while True:
#         user_input = input("\nuser: ").strip()
#         if user_input.lower() == 'q':
#             break
#         if not user_input:
#             continue
#
#         print("⏳ Парсю смысл...")
#         try:
#             raw_response = parse_chain.invoke({"input": user_input})
#             clean = raw_response.strip().replace("```json", "").replace("```", "").strip()
#             parsed = json.loads(clean)
#
#             # Валидация результата парсинга
#             if not parsed or not all(k in parsed for k in ["text", "topic", "tags"]):
#                 print("❌ Не удалось распарсить JSON. Попробуй чётче.")
#                 print(f"Debug: {parsed}")
#                 continue
#
#         except json.JSONDecodeError as e:
#             print(f"❌ Ошибка парсинга JSON: {e}")
#             print(f"Raw ответ модели: {raw_response[:200]}...")
#             continue
#         except Exception as e:
#             print(f"❌ Ошибка парсинга: {e}")
#             continue
#
#         print("Векторизую и проверяю связи...")
#
#         similar = find_similar_lc(parsed["text"], db)
#
#         if not similar:
#             print("OK: Похожих тем не найдено.")
#             decision = "new"
#         else:
#             best_match = similar[0]
#             score = best_match['similarity']
#             print(f"Прикреплено к: {best_match['text'][:30]}... (сходство: {score:.3f})")
#             print("=" * 50)
#
#
#             # Логика зон
#             if score > 0.90:
#                 print("Высокое сходство. Проверяю на дубликат...")
#                 decision = decide_with_llm_lc(parsed["text"], similar)
#             elif score > 0.75:
#                 print("Интересная связь. Это дополнение к существующей теме.")
#                 decision = "complement"
#
#                 if "related_ids" not in parsed:
#                     parsed["related_ids"] = []
#                 parsed["related_ids"].append(best_match["id"])
#             else:
#                 decision = "new"
#
#         if decision == "duplicate":
#             print("OK: Это дубликат. Пропускаем сохранение.")
#             continue
#         elif decision == "complement":
#             print(f"OK: Сохраняю как дополнение к {best_match['id']}")
#         else:
#             print("OK: Сохраняю как новую запись.")
#
#         save_knowledge_lc(parsed, db)
#
#
# if __name__ == "__main__":
#     main()