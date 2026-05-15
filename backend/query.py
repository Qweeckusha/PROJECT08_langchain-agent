import json
from operator import itemgetter
import os
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from . import models
from .skills import detect_skills, get_skills_prompt
from .ingest import rebuild_faiss_index


# ==========================================
#                 КОМПОНЕНТЫ
# ==========================================

# --- Векторная база ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))    # Путь к корню проекта (ai-agent/)
FAISS_INDEX = os.path.join(PROJECT_ROOT, "faiss_index")

embeddings = models.embeddings

if not os.path.exists(FAISS_INDEX) or not os.path.exists(os.path.join(FAISS_INDEX, "index.faiss")):
    print("⚠️ FAISS-индекс не найден. Создаю новый...")
    db = rebuild_faiss_index()
else:
    db = FAISS.load_local(FAISS_INDEX, embeddings, allow_dangerous_deserialization=True)

retriever = db.as_retriever(search_kwargs={"k": 5})
print("ОК: База загружена из 'faiss_index'.")



# --- LLM (Через LM Studio) ---
llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    model="qwen/qwen3-4b-2507",
    temperature=0.5
)


# ==========================================
#            Chain и интерфейс
# ==========================================

prompt = ChatPromptTemplate.from_template("""Ты — опытный наставник по Machine Learning. 
Твоя задача — объяснять концепции простым языком, опираясь СТРОГО на предоставленный контекст.
Записи из контекста нужно будет связать между собой, если это возможно.
Ты можешь использовать весь контекст, но важно определить релевантные вопросу части контекста. То есть обсуждение двух разных тем просто недопустимо.

ПРАВИЛА ОТВЕТА:
Объясняй своими словами, не копируй текст дословно.
Используй аналогии и примеры, если они уместны.
Структурируй ответ: короткое определение → суть → пример/применение.
Если в контексте нет ответа, честно скажи: "В базе пока нет информации об этом".
Не выдумывай факты и не добавляй знания извне.
Старайся дать максимально полный ответ.
Если пользователь просит ответить кратко, то постарайся ужать информацию.

СТОП-СЛОВА:
- Не начинай с фраз вроде "На основе контекста...", "В предоставленных данных...".
- Не перечисляй источники списком.
- Избегай сложных терминов без пояснений.

ДОПОЛНИТЕЛЬНЫЕ ИНСТРУКЦИИ (активны только если указаны ниже):
{skills}

КОНТЕКСТ (база знаний):
{context}

ВОПРОС:
{question}

Ответ:""")

def format_docs(docs) -> str:
    """Превращает список Document в строку с буллитами (как в твоём сыром коде)"""
    return "\n\n".join([f"• {doc.page_content}" for doc in docs])

chain = (
    {
        # 1. Берем строку из ключа "question" -> отдаем в retriever -> форматируем в строку
        "context": itemgetter("question") | retriever | RunnableLambda(format_docs),

        # 2. Берем ту же строку из ключа "question" -> отдаем в переменную {question} промпта
        "question": itemgetter("question"),
        "skills": itemgetter("skills")
    }
    | prompt
    | llm
    | StrOutputParser()
)

# ==========================================
#                 Контракт
# ==========================================

def stream_query(question: str):
    """Генератор для потоковой отдачи ответа и источников"""
    if not question or not question.strip():
        yield f'event: error\ndata: {{"message": "П пустой запрос"}}\n\n'
        return

    try:

        active_skills = detect_skills(question)
        skills_block = get_skills_prompt(active_skills)

        if active_skills:
            print(f"🔧 Активированы скиллы: {active_skills}")

        docs = retriever.invoke(question)
        sources = []
        for doc in docs:
            # 🛡 Безопасное извлечение текста: работаем и с Document, и с dict
            raw_content = doc.page_content if hasattr(doc, 'page_content') else doc.get("page_content", "")
            text = str(raw_content) if raw_content else ""

            # Безопасное извлечение темы
            meta = doc.metadata if hasattr(doc, 'metadata') else doc.get("metadata", {})
            topic = meta.get("topic", "info")

            sources.append({
                "topic": topic,
                "preview": text[:100].replace("\n", " ") + "..."
            })

        # Отдаём источники первым событием
        yield f'event: sources\ndata: {json.dumps(sources, ensure_ascii=False)}\n\n'

        if not docs:
            yield f'event: answer\ndata: В базе пока нет информации об этом.\n\n'
            yield 'event: done\ndata: {}\n\n'
            return

        input_data = {
            "question": question,
            "skills": skills_block
        }

        # Стримим ответ модели по чанкам (~токенам)
        for chunk in chain.stream(input_data):
            if chunk:
                yield f'event: answer\ndata: {chunk}\n\n'

    except Exception as e:
        print(f"❌ Stream Error: {e}")
        yield f'event: error\ndata: {{"message": "Ошибка генерации ответа"}}\n\n'
        return

    yield 'event: done\ndata: {}\n\n'
