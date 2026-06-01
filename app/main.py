import os

from agent import check_ollama_available, consultar_agente
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

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


@app.get("/api/ollama_check")
def ollama_check():
    ok, msg = check_ollama_available()
    return JSONResponse({"ok": ok, "message": msg})
