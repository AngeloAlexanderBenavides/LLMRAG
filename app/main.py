import os

from agent import check_ollama_available, consultar_agente, consultar_agente_stream
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

try:
    from app.history_store import get_chat_messages, list_chats
except Exception:
    from history_store import get_chat_messages, list_chats

app = FastAPI()

base_dir = os.path.dirname(__file__)
templates = Jinja2Templates(directory=os.path.join(base_dir, "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(base_dir, "static")), name="static")


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    context = {
        "request": request,
        "user_name": "Prueba",
        "user_role": "Example",
        "app_name": "IA_Bot",
    }

    try:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context=context,
        )
    except TypeError:
        return templates.TemplateResponse("index.html", context)


@app.post("/api/chat")
async def chat_endpoint(request: Request):
    data = await request.json()
    pregunta = data.get("message", "")
    chat_id = data.get("chat_id")
    if not pregunta:
        return JSONResponse({"error": "empty message"}, status_code=400)

    resultado = consultar_agente(pregunta, chat_id=chat_id)
    return JSONResponse(resultado)


import json

@app.post("/api/chat/stream")
async def chat_stream_endpoint(request: Request):
    data = await request.json()
    pregunta = data.get("message", "")
    chat_id = data.get("chat_id")
    if not pregunta:
        return JSONResponse({"error": "empty message"}, status_code=400)

    def event_generator():
        try:
            for event in consultar_agente_stream(pregunta, chat_id=chat_id):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as e:
            err_event = {"type": "error", "content": str(e)}
            yield f"data: {json.dumps(err_event, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/chats")
def chats_list():
    return JSONResponse({"chats": list_chats()})


@app.get("/api/chats/{chat_id}")
def chat_history(chat_id: str):
    return JSONResponse({"chat_id": chat_id, "messages": get_chat_messages(chat_id)})


@app.get("/api/ollama_check")
def ollama_check():
    ok, msg = check_ollama_available()
    return JSONResponse({"ok": ok, "message": msg})
