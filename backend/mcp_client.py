# backend/mcp_client.py
import asyncio
from concurrent.futures import ThreadPoolExecutor
from langchain_mcp_adapters.client import MultiServerMCPClient

# 🔹 Кэшируем инструменты, чтобы не переподключаться при каждом запросе
_cached_tools = None
_executor = ThreadPoolExecutor(max_workers=1)

def _load_mcp_async():
    """Запускает MCP-клиент в изолированном event loop"""
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        client = MultiServerMCPClient({
            "brain_files": {
                "url": "http://127.0.0.1:8001/sse",
                "transport": "sse"
            }
        })
        return loop.run_until_complete(client.get_tools())
    finally:
        loop.close()

def get_mcp_tools():
    global _cached_tools
    if _cached_tools is None:
        # Первый вызов загрузит тулзы (~0.5-1с). Дальше — мгновенно.
        _cached_tools = _executor.submit(_load_mcp_async).result()
    return _cached_tools