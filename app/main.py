import os

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.agent import consultar_agente, check_ollama_available

app = FastAPI()

base_dir = os.path.dirname(__file__)
templates = Jinja2Templates(directory=os.path.join(base_dir, "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(base_dir, "static")), name="static")


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "user_name": "LexBen",
            "user_role": "Pro",
            "app_name": "Gemini",
        },
    )


@app.post("/api/chat")
async def chat_endpoint(request: Request):
    data = await request.json()
    pregunta = data.get("message", "")
    if not pregunta:
        return JSONResponse({"error": "empty message"}, status_code=400)

    resultado = consultar_agente(pregunta)
    return JSONResponse(resultado)


@app.get("/api/ollama_check")
def ollama_check():
    ok, msg = check_ollama_available()
    return JSONResponse({"ok": ok, "message": msg})
