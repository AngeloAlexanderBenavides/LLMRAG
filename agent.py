"""Compatibility shim so `from agent import ...` keeps working.

The real implementation lives in `app/agent.py`.
"""

from app.agent import buscar_en_internet, check_ollama_available, consultar_agente
