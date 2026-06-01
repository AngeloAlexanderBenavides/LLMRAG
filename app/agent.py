import os
import uuid
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Limit BLAS threads to avoid native lib issues on some systems
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# Paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CHROMA_PATH = os.path.join(PROJECT_ROOT, "memoria_agente")

# Optional imports
try:
    import chromadb
except Exception as e:
    chromadb = None
    logger.debug("chromadb not available: %s", e)

try:
    import ollama
except Exception as e:
    ollama = None
    logger.debug("ollama not available: %s", e)

try:
    from duckduckgo_search import DDGS
except Exception:
    DDGS = None


def _get_collection():
    if chromadb is None:
        return None
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        return client.get_or_create_collection(name="conocimiento_general")
    except Exception as e:
        logger.warning("Failed to initialize ChromaDB: %s", e)
        return None


def buscar_en_internet(query: str) -> str:
    if DDGS is None:
        logger.warning("duckduckgo_search not installed; internet search disabled")
        return ""
    try:
        resultados = DDGS().text(query, max_results=3)
        texto_extraido = " ".join([res.get("body", "") for res in resultados])
        return texto_extraido
    except Exception as e:
        logger.warning("Error during internet search: %s", e)
        return ""


def check_ollama_available() -> (bool, Optional[str]):
    if ollama is None:
        return False, "ollama package not installed"
    try:
        _ = ollama.chat(model=os.environ.get("OLLAMA_MODEL", "llama3:latest"), messages=[{"role": "user", "content": "PING"}])
        return True, None
    except Exception as e:
        return False, str(e)


def consultar_agente(pregunta: str) -> dict:
    coleccion = _get_collection()
    contexto = ""
    if coleccion is not None:
        try:
            resultados_db = coleccion.query(query_texts=[pregunta], n_results=1)
            docs = resultados_db.get("documents") if isinstance(resultados_db, dict) else None
            if docs and docs[0]:
                contexto = docs[0][0]
        except Exception as e:
            logger.warning("Error querying ChromaDB: %s", e)

    prompt_evaluacion = (
        f"Contexto: {contexto}\n"
        f"Pregunta: {pregunta}\n"
        "Instruccion: Si puedes responder la pregunta con exactitud usando el contexto o tu conocimiento seguro, da la respuesta. "
        "Si no estás 100% seguro o te falta información actualizada, responde EXACTAMENTE y únicamente con la palabra: NO_SE"
    )

    if ollama is None:
        return {"answer": "Ollama no está disponible en este entorno. Instala y ejecuta Ollama localmente para obtener respuestas del modelo.", "source": "error"}

    try:
        respuesta_inicial = ollama.chat(model=os.environ.get("OLLAMA_MODEL", "llama3:latest"), messages=[{"role": "user", "content": prompt_evaluacion}])
        respuesta_llm = respuesta_inicial.get("message", {}).get("content", "").strip()
    except Exception as e:
        logger.warning("Error calling Ollama: %s", e)
        return {"answer": f"Error al comunicarse con Ollama: {e}", "source": "error"}

    if "NO_SE" in respuesta_llm.upper():
        info_internet = buscar_en_internet(pregunta)
        if info_internet:
            if coleccion is not None:
                try:
                    coleccion.add(documents=[info_internet], metadatas=[{"fuente": "internet", "query": pregunta}], ids=[str(uuid.uuid4())])
                except Exception as e:
                    logger.warning("Failed to save info to ChromaDB: %s", e)

            prompt_final = f"Responde a la pregunta '{pregunta}' basándote en esta nueva información de internet: {info_internet}"
            try:
                respuesta_final = ollama.chat(model=os.environ.get("OLLAMA_MODEL", "llama3:latest"), messages=[{"role": "user", "content": prompt_final}])
                return {"answer": respuesta_final.get("message", {}).get("content", "").strip(), "source": "internet"}
            except Exception as e:
                logger.warning("Error calling Ollama for final answer: %s", e)
                return {"answer": "Se obtuvo información de internet pero falló la generación final del LLM.", "source": "internet"}
        else:
            return {"answer": "No se encontró información en internet.", "source": "internet"}
    else:
        return {"answer": respuesta_llm, "source": "local_or_model"}
