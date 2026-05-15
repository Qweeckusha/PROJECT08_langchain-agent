from pathlib import Path
from mcp.server.fastmcp import FastMCP

mcp = FastMCP(name="BrAIn File Tools", port=8001, host='127.0.0.1')

# Песочница: сервер видит ТОЛЬКО эту папку
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = PROJECT_ROOT / "brain_docs"
BASE_DIR.mkdir(exist_ok=True)

def _safe_resolve(filename: str) -> Path | None:
    """Защищает от выхода за пределы BASE_DIR"""
    target = (BASE_DIR / filename).resolve()
    return target if target.is_relative_to(BASE_DIR.resolve()) else None

# ==========================================
# 🛠 ИНСТРУМЕНТЫ (Tools)
# ==========================================

@mcp.tool()
async def read_text(filename: str) -> str:
    """Читает .txt, .md, .json файлы. Идеально для статей, заметок, документации."""
    path = _safe_resolve(filename)
    if not path or not path.exists():
        return "❌ Файл не найден или доступ запрещён."
    if path.suffix not in [".txt", ".md", ".json"]:
        return "⚠️ Поддерживаются только .txt, .md, .json"
    return path.read_text(encoding="utf-8")

@mcp.tool()
async def read_pdf(filename: str) -> str:
    """Извлекает текст из PDF. Работает с текстовыми PDF (не сканами)."""
    import pypdf
    path = _safe_resolve(filename)
    if not path or not path.exists() or path.suffix != ".pdf":
        return "❌ PDF не найден."
    try:
        reader = pypdf.PdfReader(path)
        text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
        return text.strip() if text.strip() else "⚠️ Текст не извлечён (файл может быть сканом или защищён)."
    except Exception as e:
        return f"❌ Ошибка чтения PDF: {str(e)}"

@mcp.tool()
async def read_docx(filename: str) -> str:
    """Извлекает текст из документов Word (.docx, 2007+)."""
    import docx
    path = _safe_resolve(filename)
    if not path or not path.exists() or path.suffix != ".docx":
        return "❌ DOCX не найден."
    try:
        doc = docx.Document(path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(paragraphs)
    except Exception as e:
        return f"❌ Ошибка чтения DOCX: {str(e)}"

if __name__ == "__main__":
    print("🚀 MCP-сервер запущен: http://127.0.0.1:8001/sse")
    # ✅ Настройки сети передаём сюда
    mcp.run(transport="sse")
    print(f"📁 Рабочая папка: {BASE_DIR}")