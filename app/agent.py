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


from typing import Generator, Union

def _chat_with_ollama(
    messages: list[dict], 
    temperature: float = 0.0, 
    retries: int = 2,
    stream: bool = False
) -> Union[str, Generator[dict, None, None]]:
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
                stream=stream
            )
            if stream:
                return respuesta
            
            contenido = respuesta.get("message", {}).get("content", "").strip()
            if contenido:
                return contenido
        except Exception as error:
            last_error = error

    if last_error is not None:
        raise last_error

    return ""


def clasificar_pregunta(pregunta: str) -> str:
    # Todo es clasificado mediante el LLM (Inteligencia Artificial) de Ollama, sin filtros de código (sin cosas quemadas).
    prompt_sistema = (
        "Tu única tarea es clasificar la entrada del usuario en una sola palabra en mayúsculas, eligiendo estrictamente de esta lista:\n"
        "1. SALUDO\n"
        "2. INCOMPLETA\n"
        "3. VARIABLE\n"
        "4. FIJA\n\n"
        "Guía de clasificación con ejemplos:\n\n"
        "- SALUDO:\n"
        "  * Mensajes de saludo, despedida o cortesía breves y sencillos sin una consulta o pregunta sustancial.\n"
        "  * Ejemplos: 'Hola', 'Buenos días', 'Buenas tardes', 'Hola, cómo estás?', 'Saludos', 'hey', 'hi', 'hello', 'que tal', 'como va'.\n"
        "  * Si la entrada es 'Hola', tu respuesta DEBE ser 'SALUDO'.\n\n"
        "- INCOMPLETA:\n"
        "  * Entradas extremadamente cortas (de una o dos palabras), fragmentos cortados, signos de puntuación sueltos o palabras sueltas sin sentido claro ni verbo que no permiten dar una respuesta informativa.\n"
        "  * Ejemplos: 'Que', 'Por', 'a', 'entonces', 'de', 'cómo', '?', '...', 'q', 'k', 'quien', 'por que', 'y'.\n"
        "  * Si la entrada es 'Que' o '¿Que?', tu respuesta DEBE ser 'INCOMPLETA'.\n\n"
        "- VARIABLE:\n"
        "  * Preguntas sobre información dinámica o en tiempo real que depende del día, la hora, el clima actual, noticias del día o cotizaciones financieras en vivo.\n"
        "  * Ejemplos: '¿Qué hora es?', '¿Cómo estará el clima hoy?', 'noticias de hoy sobre fútbol', '¿quién ganó ayer?'.\n\n"
        "- FIJA:\n"
        "  * Preguntas o consultas sobre conocimiento e información estables o históricos que no cambian con el tiempo (conceptos teóricos, hechos históricos, biografías, recetas de cocina estables, etc.).\n"
        "  * Ejemplos: '¿Qué es la fotosíntesis?', '¿Quién descubrió América?', 'capital de Francia', '¿cómo funciona el motor?'.\n\n"
        "Regla de salida obligatoria:\n"
        "- Responde únicamente con una palabra de las cuatro opciones: SALUDO, INCOMPLETA, VARIABLE o FIJA.\n"
        "- NO incluyas introducciones, explicaciones, justificaciones ni signos de puntuación."
    )

    messages = [
        {"role": "system", "content": prompt_sistema},
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
        
        # Reintento con un prompt aún más directo si el modelo falló el formato en el primer intento
        respuesta_reintento = _chat_with_ollama(
            [
                {
                    "role": "system",
                    "content": "Responde únicamente con una de estas cuatro palabras en mayúsculas: SALUDO, INCOMPLETA, VARIABLE, FIJA.",
                },
                {"role": "user", "content": pregunta},
            ],
            temperature=0.0,
        ).upper().strip()
        if "SALUDO" in respuesta_reintento:
            return "saludo"
        if "INCOMPLETA" in respuesta_reintento:
            return "incompleta"
        if "VARIABLE" in respuesta_reintento:
            return "variable"
        if "FIJA" in respuesta_reintento:
            return "fija"
    except Exception as e:
        logger.warning("Error classifying question type with Ollama: %s", e)

    # Fallback conservador
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


def _yield_string_as_tokens(text: str) -> Generator[dict, None, None]:
    import time
    words = text.split(" ")
    for i, word in enumerate(words):
        space = " " if i > 0 else ""
        yield {"type": "token", "content": space + word}
        time.sleep(0.015)


def consultar_agente_stream(pregunta: str, chat_id: Optional[str] = None) -> Generator[dict, None, None]:
    import time
    chat_id_normalizado = _normalize_chat_id(chat_id)
    
    print(f"\n[LOG DE AGENTE] ===== INICIO DE PROCESAMIENTO =====")
    print(f"[LOG DE AGENTE] Pregunta recibida: '{pregunta}'")
    print(f"[LOG DE AGENTE] ID de chat: '{chat_id_normalizado}'")
    
    ensure_chat(chat_id_normalizado)
    save_message(chat_id_normalizado, "user", pregunta)
    
    titulo_chat = pregunta.strip().replace("\n", " ")[:60]
    if titulo_chat:
        set_chat_title(chat_id_normalizado, titulo_chat)

    yield {"type": "status", "content": "Clasificando pregunta..."}
    
    print(f"[LOG DE AGENTE] Ejecutando clasificación por IA...")
    tipo_pregunta = clasificar_pregunta(pregunta)
    print(f"[LOG DE AGENTE] Clasificación obtenida: {tipo_pregunta.upper()}")

    if tipo_pregunta == "saludo":
        yield {"type": "status", "content": "Generando saludo..."}
        messages = [
            {
                "role": "system",
                "content": (
                    "El usuario solo saludó. Responde de forma breve, amable y natural. "
                    "No des explicaciones, no hagas listas y no agregues contexto."
                ),
            },
            {"role": "user", "content": pregunta},
        ]
        
        print(f"[LOG DE AGENTE] [SALUDO] Enviando prompt a Ollama...")
        try:
            response_gen = _chat_with_ollama(messages, temperature=0.2, stream=True)
            full_text = ""
            for chunk in response_gen:
                token = chunk.get("message", {}).get("content", "")
                if token:
                    full_text += token
                    yield {"type": "token", "content": token}
            
            print(f"[LOG DE AGENTE] [SALUDO] Respuesta generada: '{full_text}'")
            save_message(chat_id_normalizado, "assistant", full_text, source="greeting", question_type="saludo")
            yield {"type": "done", "chat_id": chat_id_normalizado, "source": "greeting", "answer": full_text}
            print(f"[LOG DE AGENTE] ===== FIN DE PROCESAMIENTO =====")
            return
        except Exception as e:
            print(f"[LOG DE AGENTE] [ERROR SALUDO] Fallo en Ollama: {e}")
            error_msg = "Hola, ¿en qué te ayudo?"
            for t in _yield_string_as_tokens(error_msg):
                yield t
            save_message(chat_id_normalizado, "assistant", error_msg, source="greeting", question_type="saludo")
            yield {"type": "done", "chat_id": chat_id_normalizado, "source": "greeting", "answer": error_msg}
            return

    if tipo_pregunta == "incompleta":
        yield {"type": "status", "content": "Procesando entrada..."}
        messages = [
            {
                "role": "system",
                "content": (
                    "La entrada del usuario está incompleta o no es suficiente para responder. "
                    "Pide que complete la idea en una sola frase breve y natural. "
                    "No des ejemplos ni explicaciones."
                ),
            },
            {"role": "user", "content": pregunta},
        ]
        
        print(f"[LOG DE AGENTE] [INCOMPLETA] Enviando prompt a Ollama...")
        try:
            response_gen = _chat_with_ollama(messages, temperature=0.1, stream=True)
            full_text = ""
            for chunk in response_gen:
                token = chunk.get("message", {}).get("content", "")
                if token:
                    full_text += token
                    yield {"type": "token", "content": token}
            
            print(f"[LOG DE AGENTE] [INCOMPLETA] Respuesta generada: '{full_text}'")
            save_message(chat_id_normalizado, "assistant", full_text, source="incomplete", question_type="incompleta")
            yield {"type": "done", "chat_id": chat_id_normalizado, "source": "incomplete", "answer": full_text}
            print(f"[LOG DE AGENTE] ===== FIN DE PROCESAMIENTO =====")
            return
        except Exception as e:
            print(f"[LOG DE AGENTE] [ERROR INCOMPLETA] Fallo en Ollama: {e}")
            error_msg = "¿Puedes completar tu pregunta?"
            for t in _yield_string_as_tokens(error_msg):
                yield t
            save_message(chat_id_normalizado, "assistant", error_msg, source="incomplete", question_type="incompleta")
            yield {"type": "done", "chat_id": chat_id_normalizado, "source": "incomplete", "answer": error_msg}
            return

    if tipo_pregunta == "variable":
        yield {"type": "status", "content": "Analizando información..."}
        messages = [
            {
                "role": "system",
                "content": (
                    "Responde solo a lo pedido por el usuario. No des explicaciones, ejemplos ni contexto extra. "
                    "Máximo una frase corta. Si no conoces la respuesta exacta y actual, responde exactamente: NO_SE"
                ),
            },
            {"role": "user", "content": pregunta},
        ]
        
        print(f"[LOG DE AGENTE] [VARIABLE] Evaluando si el modelo conoce la respuesta de inmediato...")
        respuesta_llm = ""
        try:
            respuesta_llm = _chat_with_ollama(messages, temperature=0.0)
            print(f"[LOG DE AGENTE] [VARIABLE] Respuesta tentativa del LLM: '{respuesta_llm}'")
        except Exception as e:
            print(f"[LOG DE AGENTE] [VARIABLE] Error al llamar Ollama: {e}")
            respuesta_llm = "NO_SE"

        if "NO_SE" not in respuesta_llm.upper():
            print(f"[LOG DE AGENTE] [VARIABLE] LLM conoce la respuesta. Transmitiendo en tiempo real...")
            for t in _yield_string_as_tokens(respuesta_llm):
                yield t
            save_message(chat_id_normalizado, "assistant", respuesta_llm, source="llm_variable", question_type="variable")
            yield {"type": "done", "chat_id": chat_id_normalizado, "source": "llm_variable", "answer": respuesta_llm}
            print(f"[LOG DE AGENTE] ===== FIN DE PROCESAMIENTO =====")
            return

        print(f"[LOG DE AGENTE] [VARIABLE] El LLM respondió NO_SE. Iniciando búsqueda web...")
        yield {"type": "status", "content": "Buscando en Internet..."}
        
        print(f"[LOG DE AGENTE] [VARIABLE] Ejecutando búsqueda en DuckDuckGo para: '{pregunta}'")
        info_internet = buscar_en_internet(pregunta)
        print(f"[LOG DE AGENTE] [VARIABLE] Información recuperada de internet: {info_internet[:200]}...")

        if not info_internet:
            if any(palabra in pregunta.lower() for palabra in ("día", "fecha", "hoy", "date")):
                fecha_actual = datetime.now().strftime("%d-%m-%Y")
                ans = f"Hoy es {fecha_actual}."
                print(f"[LOG DE AGENTE] [VARIABLE] Fallback de fecha local activado: '{ans}'")
                for t in _yield_string_as_tokens(ans):
                    yield t
                save_message(chat_id_normalizado, "assistant", ans, source="system_date", question_type="variable")
                yield {"type": "done", "chat_id": chat_id_normalizado, "source": "system_date", "answer": ans}
                print(f"[LOG DE AGENTE] ===== FIN DE PROCESAMIENTO =====")
                return

            ans = f"No encontré información actualizada sobre: {pregunta}"
            print(f"[LOG DE AGENTE] [VARIABLE] Sin resultados. Retornando fallback: '{ans}'")
            for t in _yield_string_as_tokens(ans):
                yield t
            save_message(chat_id_normalizado, "assistant", ans, source="internet", question_type="variable")
            yield {"type": "done", "chat_id": chat_id_normalizado, "source": "internet", "answer": ans}
            print(f"[LOG DE AGENTE] ===== FIN DE PROCESAMIENTO =====")
            return

        yield {"type": "status", "content": "Redactando respuesta desde Internet..."}
        prompt_final = (
            f"Pregunta: {pregunta}\n"
            f"Información de internet: {info_internet}\n"
            "Responde solo lo que se pidió. No agregues contexto, explicaciones ni listas."
        )
        final_messages = [
            {
                "role": "system",
                "content": (
                    "Responde con una sola respuesta breve y directa. "
                    "No expliques el proceso, no menciones el contexto y no agregues información extra."
                ),
            },
            {"role": "user", "content": prompt_final},
        ]
        
        print(f"[LOG DE AGENTE] [VARIABLE] Enviando información de Internet al LLM para redacción final...")
        try:
            response_gen = _chat_with_ollama(final_messages, temperature=0.2, stream=True)
            full_text = ""
            for chunk in response_gen:
                token = chunk.get("message", {}).get("content", "")
                if token:
                    full_text += token
                    yield {"type": "token", "content": token}
            
            print(f"[LOG DE AGENTE] [VARIABLE] Redacción final completada: '{full_text}'")
            save_message(chat_id_normalizado, "assistant", full_text or info_internet, source="internet", question_type="variable")
            yield {"type": "done", "chat_id": chat_id_normalizado, "source": "internet", "answer": full_text or info_internet}
            print(f"[LOG DE AGENTE] ===== FIN DE PROCESAMIENTO =====")
            return
        except Exception as e:
            print(f"[LOG DE AGENTE] [ERROR VARIABLE] Error al redactar respuesta: {e}. Enviando datos puros de internet.")
            ans = info_internet
            for t in _yield_string_as_tokens(ans):
                yield t
            save_message(chat_id_normalizado, "assistant", ans, source="internet", question_type="variable")
            yield {"type": "done", "chat_id": chat_id_normalizado, "source": "internet", "answer": ans}
            print(f"[LOG DE AGENTE] ===== FIN DE PROCESAMIENTO =====")
            return

    # Pregunta Fija (RAG)
    print(f"[LOG DE AGENTE] [FIJA] Buscando en base de conocimientos ChromaDB...")
    yield {"type": "status", "content": "Buscando en base de conocimientos..."}
    coleccion = _get_collection()
    contexto = ""
    if coleccion is not None:
        try:
            resultados_db = coleccion.query(
                query_texts=[pregunta],
                n_results=1,
                where={"chat_id": chat_id_normalizado},
            )
            print(f"[LOG DE AGENTE] [FIJA] Consulta ChromaDB completada. Resultados: {resultados_db}")
            docs = resultados_db.get("documents") if isinstance(resultados_db, dict) else None
            if docs and docs[0]:
                contexto = docs[0][0]
                print(f"[LOG DE AGENTE] [FIJA] Contexto RAG recuperado: '{contexto[:200]}...'")
            else:
                print(f"[LOG DE AGENTE] [FIJA] No se encontró contexto previo para este chat en ChromaDB.")
        except Exception as e:
            print(f"[LOG DE AGENTE] [ERROR CHROMADB] Error en consulta: {e}")

    prompt_evaluacion = (
        f"Contexto: {contexto}\n"
        f"Pregunta: {pregunta}\n"
        "Instruccion: Responde solo lo que se pidió, en forma breve y directa. "
        "Si no puedes responder con exactitud usando el contexto o tu conocimiento seguro, responde EXACTAMENTE y únicamente con la palabra: NO_SE"
    )

    eval_messages = [
        {
            "role": "system",
            "content": (
                "Responde solo a la pregunta del usuario. "
                "No des explicaciones, no agregues contexto extra, no hagas listas y sé breve."
            ),
        },
        {"role": "user", "content": prompt_evaluacion},
    ]

    print(f"[LOG DE AGENTE] [FIJA] Enviando pregunta de evaluación al LLM...")
    respuesta_llm = ""
    try:
        respuesta_llm = _chat_with_ollama(eval_messages, temperature=0.0)
        print(f"[LOG DE AGENTE] [FIJA] Respuesta del LLM a evaluación: '{respuesta_llm}'")
    except Exception as e:
        print(f"[LOG DE AGENTE] [FIJA] Error al llamar a Ollama para evaluación: {e}")
        respuesta_llm = "NO_SE"

    if "NO_SE" not in respuesta_llm.upper():
        print(f"[LOG DE AGENTE] [FIJA] El LLM conoce la respuesta segura. Transmitiendo en tiempo real...")
        for t in _yield_string_as_tokens(respuesta_llm):
            yield t
        save_message(chat_id_normalizado, "assistant", respuesta_llm, source="local_or_model", question_type="fija")
        yield {"type": "done", "chat_id": chat_id_normalizado, "source": "local_or_model", "answer": respuesta_llm}
        print(f"[LOG DE AGENTE] ===== FIN DE PROCESAMIENTO =====")
        return

    print(f"[LOG DE AGENTE] [FIJA] El LLM respondió NO_SE. Iniciando búsqueda web de fallback...")
    yield {"type": "status", "content": "Buscando información en Internet..."}
    
    print(f"[LOG DE AGENTE] [FIJA] Ejecutando búsqueda en DuckDuckGo para: '{pregunta}'")
    info_internet = buscar_en_internet(pregunta)
    print(f"[LOG DE AGENTE] [FIJA] Información de Internet recuperada: {info_internet[:200]}...")

    if info_internet:
        if coleccion is not None:
            try:
                print(f"[LOG DE AGENTE] [FIJA] Guardando nueva información en ChromaDB para el chat '{chat_id_normalizado}'...")
                coleccion.add(
                    documents=[info_internet],
                    metadatas=[{"fuente": "internet", "query": pregunta, "chat_id": chat_id_normalizado}],
                    ids=[str(uuid.uuid4())],
                )
            except Exception as e:
                print(f"[LOG DE AGENTE] [ERROR CHROMADB] No se pudo guardar la información: {e}")
        save_memory_entry(chat_id_normalizado, pregunta, info_internet, "internet")

        yield {"type": "status", "content": "Redactando respuesta desde Internet..."}
        prompt_final = f"Responde a la pregunta '{pregunta}' basándote en esta nueva información de internet: {info_internet}"
        final_messages = [
            {
                "role": "system",
                "content": (
                    "Responde con una sola respuesta breve y directa. "
                    "No expliques el proceso, no menciones el contexto y no agregues información extra."
                ),
            },
            {"role": "user", "content": prompt_final},
        ]
        
        print(f"[LOG DE AGENTE] [FIJA] Enviando información de Internet al LLM para redacción final...")
        try:
            response_gen = _chat_with_ollama(final_messages, temperature=0.2, stream=True)
            full_text = ""
            for chunk in response_gen:
                token = chunk.get("message", {}).get("content", "")
                if token:
                    full_text += token
                    yield {"type": "token", "content": token}
            
            print(f"[LOG DE AGENTE] [FIJA] Redacción final completada: '{full_text}'")
            save_message(chat_id_normalizado, "assistant", full_text or info_internet, source="internet", question_type="fija")
            yield {"type": "done", "chat_id": chat_id_normalizado, "source": "internet", "answer": full_text or info_internet}
            print(f"[LOG DE AGENTE] ===== FIN DE PROCESAMIENTO =====")
            return
        except Exception as e:
            print(f"[LOG DE AGENTE] [ERROR FIJA] Error al redactar respuesta final: {e}. Enviando datos de internet puros.")
            ans = info_internet
            for t in _yield_string_as_tokens(ans):
                yield t
            save_message(chat_id_normalizado, "assistant", ans, source="internet", question_type="fija")
            yield {"type": "done", "chat_id": chat_id_normalizado, "source": "internet", "answer": ans}
            print(f"[LOG DE AGENTE] ===== FIN DE PROCESAMIENTO =====")
            return
    else:
        ans = "No se encontró información en internet."
        print(f"[LOG DE AGENTE] [FIJA] No se obtuvieron resultados en Internet. Retornando fallback: '{ans}'")
        for t in _yield_string_as_tokens(ans):
            yield t
        save_message(chat_id_normalizado, "assistant", ans, source="internet", question_type="fija")
        yield {"type": "done", "chat_id": chat_id_normalizado, "source": "internet", "answer": ans}
        print(f"[LOG DE AGENTE] ===== FIN DE PROCESAMIENTO =====")
        return
