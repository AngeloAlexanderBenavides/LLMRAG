# Chatbot FastAPI view

Minimal FastAPI app that exposes a web view and `/api/chat` endpoint which uses the existing `agent` flow (ChromaDB + Ollama + DuckDuckGo search).

Run locally on the Orange Pi (assumes `ollama` and the model are installed and accessible):

```powershell
python -m pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Visit http://<orangepi-ip>:8000 in your browser.

## Ollama setup

Before using the app, make sure Ollama is running and the model exists locally.

```bash
ollama serve
ollama pull qwen2.5:3b
```

If you want to use another model, set `OLLAMA_MODEL` before starting the app:

```powershell
$env:OLLAMA_MODEL = "qwen2.5:3b"
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Quick local test

```powershell
uv run python -c "from app.agent import consultar_agente; print(consultar_agente('¿Cuál fue el ganador de la Copa Mundial 2022?'))"
```

If Ollama is not available, the app will now return a clear error instead of failing silently.

## Dependency management with `uv`

This project includes the `uv` package to help manage dependency versions and lockfiles.

Basic workflow:

```bash
# install uv in your environment
python -m pip install uv

# create a lockfile (example; check `uv` docs for exact flags)
uv lock

# synchronize/install pinned deps from the lockfile
uv sync
```

If you prefer `pip` only, `requirements.txt` is available and works with `pip install -r requirements.txt`.

## Windows-specific steps (this repo is being developed on Windows)

1. Install Python dependencies and run the server (PowerShell):

```powershell
.\run_server.ps1
```

2. Install and run Ollama:

- Follow the official Ollama installation guide: https://docs.ollama.com/installation
- Start the Ollama daemon (example):

```powershell
ollama serve
ollama pull qwen2.5:3b
```

3. Verify Ollama from the project:

```powershell
uv run curl --silent http://localhost:8000/api/ollama_check | ConvertFrom-Json
```

If the check reports `ok: false` the returned message will help diagnose.

## Upload to GitHub

To push this project to GitHub:

```powershell
git init
git add .
git commit -m "Initial commit - chatbot with Ollama + ChromaDB UI"
git remote add origin https://github.com/<tu-usuario>/<tu-repo>.git
git push -u origin main
```

Then on a different machine (or Orange Pi), clone and run:

```bash
git clone https://github.com/<tu-usuario>/<tu-repo>.git
cd <tu-repo>
./run_server.sh   # on Linux/OrangePi
# or on Windows PowerShell
.\run_server.ps1
```
