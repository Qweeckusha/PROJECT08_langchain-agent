from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from . import query, ingest

app = FastAPI(title="BrAIn Web", version="0.1.0")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")

class IngestRequest(BaseModel):
    text: str

@app.get("/")
async def index():
    return FileResponse("frontend/index.html")

@app.post("/api/query")
async def api_query(request: Request):
    data = await request.json()
    question = data.get("question", "")
    # media_type="text/event-stream" говорит браузеру, что это SSE поток
    return StreamingResponse(query.stream_query(question), media_type="text/event-stream")

@app.post("/api/ingest")
async def api_ingest(request: IngestRequest):
    """Асинхронный эндпоинт: корректно работает с MCP-инструментами"""
    result = await ingest.process_ingest_async(request.text)
    return result