import logging
import os
import uuid
from datetime import datetime
from typing import Optional

try:
    from app.history_store import ensure_chat, save_memory_entry, save_message, set_chat_title, touch_chat
except Exception:
    from history_store import ensure_chat, save_memory_entry, save_message, set_chat_title, touch_chat

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


def _normalize_chat_id(chat_id: Optional[str]) -> str:
    value = (chat_id or "default").strip()
    return value or "default"


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


def _chat_with_ollama(messages: list[dict], temperature: float = 0.0, retries: int = 2) -> str:
    if ollama is None:
        raise RuntimeError(
            "Ollama no está disponible en este entorno. Instala y ejecuta Ollama localmente para obtener respuestas del modelo."
        )

    last_error: Optional[Exception] = None
    for _ in range(max(retries, 1)):
        try:
            respuesta = ollama.chat(
                model=os.environ.get("OLLAMA_MODEL", "llama3:latest"),
                messages=messages,
                options={"temperature": temperature},
            )
            contenido = respuesta.get("message", {}).get("content", "").strip()
            if contenido:
                return contenido
        except Exception as error:
            last_error = error

    if last_error is not None:
        raise last_error

    return ""


def clasificar_pregunta(pregunta: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "Clasifica la entrada en una sola palabra: VARIABLE, FIJA, SALUDO o INCOMPLETA. "
                "SALUDO = mensajes de saludo o apertura sin pregunta concreta. "
                "INCOMPLETA = entradas muy cortas o cortadas que no permiten responder con precisión. "
                "VARIABLE = depende de fecha, hora, noticias, clima, precios o estado actual. "
                "FIJA = hechos históricos, definiciones o conocimiento estable. "
                "Responde exactamente con una sola palabra, sin explicación."
            ),
        },
        {"role": "user", "content": pregunta},
    ]

    try:
        respuesta = _chat_with_ollama(messages, temperature=0.0)
        respuesta = respuesta.upper().strip()
        if "SALUDO" in respuesta:
            return "saludo"
        if "INCOMPLETA" in respuesta:
            return "incompleta"
        if "VARIABLE" in respuesta:
            return "variable"
        if "FIJA" in respuesta:
            return "fija"
        respuesta = _chat_with_ollama(
            [
                {
                    "role": "system",
                    "content": "Responde solo con una de estas palabras: VARIABLE, FIJA, SALUDO, INCOMPLETA.",
                },
                {"role": "user", "content": pregunta},
            ],
            temperature=0.0,
        ).upper().strip()
        if "SALUDO" in respuesta:
            return "saludo"
        if "INCOMPLETA" in respuesta:
            return "incompleta"
        if "VARIABLE" in respuesta:
            return "variable"
        if "FIJA" in respuesta:
            return "fija"
    except Exception as e:
        logger.warning("Error classifying question type with Ollama: %s", e)

    # Fallback conservador: si no podemos clasificar, tratamos la pregunta como fija.
    return "fija"


def responder_saludo(pregunta: str) -> dict:
    respuesta = _chat_with_ollama(
        [
            {
                "role": "system",
                "content": (
                    "El usuario solo saludó. Responde de forma breve, amable y natural. "
                    "No des explicaciones, no hagas listas y no agregues contexto."
                ),
            },
            {"role": "user", "content": pregunta},
        ],
        temperature=0.2,
    )
    return {"answer": respuesta or "Hola, ¿en qué te ayudo?", "source": "greeting"}


def responder_incompleta(pregunta: str) -> dict:
    respuesta = _chat_with_ollama(
        [
            {
                "role": "system",
                "content": (
                    "La entrada del usuario está incompleta o no es suficiente para responder. "
                    "Pide que complete la idea en una sola frase breve y natural. "
                    "No des ejemplos ni explicaciones."
                ),
            },
            {"role": "user", "content": pregunta},
        ],
        temperature=0.1,
    )
    return {"answer": respuesta or "¿Puedes completar tu pregunta?", "source": "incomplete"}


def responder_variable(pregunta: str) -> dict:
    respuesta_llm = _chat_with_ollama(
        [
            {
                "role": "system",
                "content": (
                    "Responde solo a lo pedido por el usuario. No des explicaciones, ejemplos ni contexto extra. "
                    "Máximo una frase corta. Si no conoces la respuesta exacta y actual, responde exactamente: NO_SE"
                ),
            },
            {"role": "user", "content": pregunta},
        ],
        temperature=0.0,
    )

    if "NO_SE" not in respuesta_llm.upper():
        return {"answer": respuesta_llm, "source": "llm_variable"}

    info_internet = buscar_en_internet(pregunta)
    if not info_internet:
        if any(palabra in pregunta.lower() for palabra in ("día", "fecha", "hoy", "date")):
            fecha_actual = datetime.now().strftime("%d-%m-%Y")
            return {"answer": f"Hoy es {fecha_actual}.", "source": "system_date"}

        return {"answer": f"No encontré información actualizada sobre: {pregunta}", "source": "internet"}

    prompt_final = (
        f"Pregunta: {pregunta}\n"
        f"Información de internet: {info_internet}\n"
        "Responde solo lo que se pidió. No agregues contexto, explicaciones ni listas."
    )
    respuesta_final = _chat_with_ollama(
        [
            {
                "role": "system",
                "content": (
                    "Responde con una sola respuesta breve y directa. "
                    "No expliques el proceso, no menciones el contexto y no agregues información extra."
                ),
            },
            {"role": "user", "content": prompt_final},
        ],
        temperature=0.2,
    )

    if not respuesta_final:
        return {"answer": info_internet, "source": "internet"}

    return {"answer": respuesta_final, "source": "internet"}


def check_ollama_available() -> (bool, Optional[str]):
    if ollama is None:
        return False, "ollama package not installed"
    try:
        _ = ollama.chat(model=os.environ.get("OLLAMA_MODEL", "llama3:latest"), messages=[
                        {"role": "user", "content": "PING"}])
        return True, None
    except Exception as e:
        return False, str(e)


def consultar_agente(pregunta: str, chat_id: Optional[str] = None) -> dict:
    chat_id_normalizado = _normalize_chat_id(chat_id)
    ensure_chat(chat_id_normalizado)
    save_message(chat_id_normalizado, "user", pregunta)
    titulo_chat = pregunta.strip().replace("\n", " ")[:60]
    if titulo_chat:
        set_chat_title(chat_id_normalizado, titulo_chat)
    tipo_pregunta = clasificar_pregunta(pregunta)

    if tipo_pregunta == "saludo":
        respuesta_saludo = responder_saludo(pregunta)
        save_message(
            chat_id_normalizado,
            "assistant",
            respuesta_saludo["answer"],
            source=respuesta_saludo["source"],
            question_type="saludo",
        )
        return {
            "answer": respuesta_saludo["answer"],
            "source": respuesta_saludo["source"],
            "question_type": "saludo",
            "chat_id": chat_id_normalizado,
        }

    if tipo_pregunta == "incompleta":
        respuesta_incompleta = responder_incompleta(pregunta)
        save_message(
            chat_id_normalizado,
            "assistant",
            respuesta_incompleta["answer"],
            source=respuesta_incompleta["source"],
            question_type="incompleta",
        )
        return {
            "answer": respuesta_incompleta["answer"],
            "source": respuesta_incompleta["source"],
            "question_type": "incompleta",
            "chat_id": chat_id_normalizado,
        }

    if tipo_pregunta == "variable":
        respuesta_variable = responder_variable(pregunta)
        save_message(
            chat_id_normalizado,
            "assistant",
            respuesta_variable["answer"],
            source=respuesta_variable["source"],
            question_type="variable",
        )
        return {
            "answer": respuesta_variable["answer"],
            "source": respuesta_variable["source"],
            "question_type": "variable",
            "chat_id": chat_id_normalizado,
        }

    coleccion = _get_collection()
    contexto = ""
    if coleccion is not None:
        try:
            resultados_db = coleccion.query(
                query_texts=[pregunta],
                n_results=1,
                where={"chat_id": chat_id_normalizado},
            )
            docs = resultados_db.get("documents") if isinstance(
                resultados_db, dict) else None
            if docs and docs[0]:
                contexto = docs[0][0]
        except Exception as e:
            logger.warning("Error querying ChromaDB: %s", e)

    prompt_evaluacion = (
        f"Contexto: {contexto}\n"
        f"Pregunta: {pregunta}\n"
        "Instruccion: Responde solo lo que se pidió, en forma breve y directa. "
        "Si no puedes responder con exactitud usando el contexto o tu conocimiento seguro, responde EXACTAMENTE y únicamente con la palabra: NO_SE"
    )

    try:
        respuesta_llm = _chat_with_ollama(
            [
                {
                    "role": "system",
                    "content": (
                        "Responde solo a la pregunta del usuario. "
                        "No des explicaciones, no agregues contexto extra, no hagas listas y sé breve."
                    ),
                },
                {"role": "user", "content": prompt_evaluacion},
            ],
            temperature=0.0,
        )
    except Exception as e:
        logger.warning("Error calling Ollama: %s", e)
        save_message(chat_id_normalizado, "assistant",
                     f"Error al comunicarse con Ollama: {e}", source="error", question_type="fija")
        return {"answer": f"Error al comunicarse con Ollama: {e}", "source": "error", "question_type": "fija", "chat_id": chat_id_normalizado}

    if "NO_SE" in respuesta_llm.upper():
        info_internet = buscar_en_internet(pregunta)
        if info_internet:
            if coleccion is not None:
                try:
                    coleccion.add(
                        documents=[info_internet],
                        metadatas=[{"fuente": "internet", "query": pregunta,
                                    "chat_id": chat_id_normalizado}],
                        ids=[str(uuid.uuid4())],
                    )
                except Exception as e:
                    logger.warning("Failed to save info to ChromaDB: %s", e)
            save_memory_entry(chat_id_normalizado, pregunta, info_internet, "internet")

            prompt_final = f"Responde a la pregunta '{pregunta}' basándote en esta nueva información de internet: {info_internet}"
            try:
                respuesta_final = _chat_with_ollama(
                    [
                        {
                            "role": "system",
                            "content": (
                                "Responde con una sola respuesta breve y directa. "
                                "No expliques el proceso, no menciones el contexto y no agregues información extra."
                            ),
                        },
                        {"role": "user", "content": prompt_final},
                    ],
                    temperature=0.2,
                )
                save_message(chat_id_normalizado, "assistant",
                             respuesta_final or info_internet, source="internet", question_type="fija")
                return {"answer": respuesta_final or info_internet, "source": "internet", "question_type": "fija", "chat_id": chat_id_normalizado}
            except Exception as e:
                logger.warning("Error calling Ollama for final answer: %s", e)
                save_message(chat_id_normalizado, "assistant",
                             "Se obtuvo información de internet pero falló la generación final del LLM.", source="internet", question_type="fija")
                return {"answer": "Se obtuvo información de internet pero falló la generación final del LLM.", "source": "internet", "question_type": "fija", "chat_id": chat_id_normalizado}
        else:
            save_message(chat_id_normalizado, "assistant",
                         "No se encontró información en internet.", source="internet", question_type="fija")
            return {"answer": "No se encontró información en internet.", "source": "internet", "question_type": "fija", "chat_id": chat_id_normalizado}

    save_message(chat_id_normalizado, "assistant", respuesta_llm,
                 source="local_or_model", question_type="fija")
    return {"answer": respuesta_llm, "source": "local_or_model", "question_type": "fija", "chat_id": chat_id_normalizado}
