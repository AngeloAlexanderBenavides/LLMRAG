import logging
import os
import uuid
import warnings
from datetime import datetime
from typing import Optional

# Suppress all RuntimeWarnings in this module
warnings.simplefilter("ignore", category=RuntimeWarning)

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
    from ddgs import DDGS
except Exception:
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


def buscar_en_db_qa(pregunta: str) -> Optional[dict]:
    coleccion = _get_collection()
    if coleccion is None:
        return None
    try:
        # Buscamos globalmente pares Q&A que tengan el tipo "qa_pair"
        resultados = coleccion.query(
            query_texts=[pregunta],
            n_results=1,
            where={"type": "qa_pair"}
        )
        if not resultados:
            return None
        
        ids = resultados.get("ids")
        distances = resultados.get("distances")
        metadatas = resultados.get("metadatas")
        documents = resultados.get("documents")
        
        if ids and ids[0] and distances and distances[0] and metadatas and metadatas[0] and documents and documents[0]:
            distancia = distances[0][0]
            metadata = metadatas[0][0]
            # Umbral de similitud: distancia <= 0.4 indica coincidencia semántica muy cercana
            if distancia <= 0.4:
                print(f"[LOG DE AGENTE] Coincidencia encontrada en DB para '{pregunta}'. Pregunta original: '{documents[0][0]}'. Distancia: {distancia}")
                return {
                    "question": documents[0][0],
                    "answer": metadata.get("answer"),
                    "question_type": metadata.get("question_type", "fija"),
                    "source": "database_memory"
                }
    except Exception as e:
        logger.warning("Error buscando en base de datos de memoria: %s", e)
    return None


def obtener_contexto_db(pregunta: str) -> str:
    coleccion = _get_collection()
    if coleccion is None:
        return ""
    try:
        resultados = coleccion.query(
            query_texts=[pregunta],
            n_results=1
        )
        if not resultados:
            return ""
            
        ids = resultados.get("ids")
        documents = resultados.get("documents")
        metadatas = resultados.get("metadatas")
        
        if ids and ids[0] and documents and documents[0]:
            doc = documents[0][0]
            metadata = metadatas[0][0] if (metadatas and metadatas[0]) else {}
            
            # Si es un par Q&A, estructuramos el contexto con la pregunta y respuesta
            if metadata.get("type") == "qa_pair":
                return f"Pregunta anterior similar: {doc}\nRespuesta anterior: {metadata.get('answer')}"
            else:
                # Es un snippet de internet o documento plano
                return doc
    except Exception as e:
        logger.warning("Error al obtener contexto de ChromaDB: %s", e)
    return ""


def guardar_en_db_qa(pregunta: str, respuesta: str, question_type: str, chat_id: str) -> None:
    coleccion = _get_collection()
    if coleccion is None:
        return
    try:
        print(f"[LOG DE AGENTE] Guardando par Q&A en la base de datos: '{pregunta}' -> '{respuesta[:50]}...'")
        coleccion.add(
            documents=[pregunta],
            metadatas=[{
                "answer": respuesta,
                "chat_id": chat_id,
                "question_type": question_type,
                "type": "qa_pair"
            }],
            ids=[str(uuid.uuid4())]
        )
        # Guardar también en el histórico relacional para persistencia
        save_memory_entry(chat_id, pregunta, respuesta, "database_memory")
    except Exception as e:
        logger.warning("Error al guardar en base de datos de memoria: %s", e)



def _normalize_chat_id(chat_id: Optional[str]) -> str:
    value = (chat_id or "default").strip()
    return value or "default"


def buscar_en_internet(query: str) -> str:
    if DDGS is None:
        logger.warning("duckduckgo_search not installed; internet search disabled")
        return ""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            old_showwarning = warnings.showwarning
            warnings.showwarning = lambda *args, **kwargs: None
            try:
                resultados = DDGS().text(query, max_results=3)
            finally:
                warnings.showwarning = old_showwarning
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


def _get_conversational_history(chat_id: Optional[str], limit: int = 10) -> list[dict]:
    if not chat_id:
        return []
    try:
        from app.history_store import get_chat_messages
    except Exception:
        from history_store import get_chat_messages
        
    try:
        all_messages = get_chat_messages(chat_id)
        filtered = []
        for msg in all_messages:
            role = msg.get("role")
            content = msg.get("content")
            if role in ("user", "assistant") and content:
                # Filtrar posibles errores de la base de datos para no contaminar el contexto
                if msg.get("source") == "error":
                    continue
                filtered.append({"role": role, "content": content})
        return filtered[-limit:]
    except Exception as e:
        logger.warning("Error loading conversational history: %s", e)
        return []


def generar_query_busqueda(pregunta: str, history: list[dict]) -> str:
    if not history:
        return pregunta
        
    prompt_sistema = (
        "Tu tarea es generar un único término o frase de búsqueda en internet conciso y optimizado para motores de búsqueda (como Google o Bing), "
        "que combine de manera lógica la última entrada del usuario con el contexto de la conversación anterior.\n\n"
        "Instrucciones clave:\n"
        "- Debe ser una frase o conjunto de palabras clave directas para buscar en la web.\n"
        "- NO respondas la pregunta. Tu única salida debe ser el término de búsqueda.\n"
        "- Si la última entrada ya es clara y completa por sí sola, devuélvela exactamente igual.\n"
        "- Evita agregar explicaciones, comentarios, comillas o preámbulos.\n\n"
        "Ejemplo 1:\n"
        "Historial:\n"
        "User: ¿Cuál es el país más grande del mundo?\n"
        "Assistant: Rusia.\n"
        "User: ¿y cuál es su capital?\n"
        "Resultado: capital de Rusia\n\n"
        "Ejemplo 2:\n"
        "Historial:\n"
        "User: Me interesa saber de las universidades en Ecuador.\n"
        "Assistant: Entendido, ¿sobre qué aspecto te gustaría saber?\n"
        "User: sobre el ranking de sostenibilidad\n"
        "Resultado: ranking de sostenibilidad universidades Ecuador"
    )
    
    messages = [
        {"role": "system", "content": prompt_sistema}
    ] + history + [{"role": "user", "content": f"Genera la frase de búsqueda para: '{pregunta}'"}]
    
    try:
        query_generado = _chat_with_ollama(messages, temperature=0.0)
        query_generado = query_generado.strip().replace('"', '').replace("'", "")
        if query_generado:
            return query_generado
    except Exception as e:
        logger.warning("Error generating search query: %s", e)
    return pregunta


def clasificar_pregunta(pregunta: str, chat_id: Optional[str] = None, history: Optional[list[dict]] = None) -> str:
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

    if history is None:
        history = _get_conversational_history(chat_id, limit=6)
    messages = [{"role": "system", "content": prompt_sistema}] + history + [{"role": "user", "content": pregunta}]

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
        messages_reintento = [
            {
                "role": "system",
                "content": "Responde únicamente con una de estas cuatro palabras en mayúsculas: SALUDO, INCOMPLETA, VARIABLE, FIJA.",
            }
        ] + history + [{"role": "user", "content": pregunta}]
        
        respuesta_reintento = _chat_with_ollama(messages_reintento, temperature=0.0).upper().strip()
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
    
    # 1. Obtener historial conversacional PREVIO (antes de guardar la pregunta actual)
    history = _get_conversational_history(chat_id_normalizado, limit=8)
    
    # 2. Guardar la pregunta actual en la base de datos
    save_message(chat_id_normalizado, "user", pregunta)
    titulo_chat = pregunta.strip().replace("\n", " ")[:60]
    if titulo_chat:
        set_chat_title(chat_id_normalizado, titulo_chat)

    # 3. Revisar la base de datos primero (coincidencia de Q&A)
    coincidencia = buscar_en_db_qa(pregunta)
    if coincidencia:
        answer = coincidencia["answer"]
        source = coincidencia["source"]
        q_type = coincidencia["question_type"]
        save_message(chat_id_normalizado, "assistant", answer, source=source, question_type=q_type)
        return {
            "answer": answer,
            "source": source,
            "question_type": q_type,
            "chat_id": chat_id_normalizado,
        }

    # 4. Clasificar la pregunta pasándole el historial pre-cargado
    tipo_pregunta = clasificar_pregunta(pregunta, chat_id=chat_id_normalizado, history=history)

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
        # Preguntar al LLM directamente con temperatura 0.0
        messages = [
            {
                "role": "system",
                "content": (
                    "Responde con exactitud y de forma directa a la pregunta del usuario. "
                    "Si no tienes o no conoces la información exacta y actual en tu conocimiento interno para responder con precisión, responde exactamente con la palabra: NO_SE"
                ),
            },
        ] + history + [{"role": "user", "content": pregunta}]
        
        try:
            respuesta_llm = _chat_with_ollama(messages, temperature=0.0)
        except Exception as e:
            logger.warning("Error calling Ollama in variable mode: %s", e)
            respuesta_llm = "NO_SE"

        if "NO_SE" not in respuesta_llm.upper() and respuesta_llm.strip():
            # LLM conoce la respuesta. Guardamos en DB y retornamos.
            guardar_en_db_qa(pregunta, respuesta_llm, "variable", chat_id_normalizado)
            save_message(chat_id_normalizado, "assistant", respuesta_llm, source="llm_variable", question_type="variable")
            return {
                "answer": respuesta_llm,
                "source": "llm_variable",
                "question_type": "variable",
                "chat_id": chat_id_normalizado,
            }

        # Si el LLM no sabe, buscamos en internet
        query_busqueda = generar_query_busqueda(pregunta, history)
        info_internet = buscar_en_internet(query_busqueda)
        
        if not info_internet:
            # Fallback de fecha local si es relevante
            if any(palabra in pregunta.lower() for palabra in ("día", "fecha", "hoy", "date")):
                fecha_actual = datetime.now().strftime("%d-%m-%Y")
                ans = f"Hoy es {fecha_actual}."
                save_message(chat_id_normalizado, "assistant", ans, source="system_date", question_type="variable")
                return {"answer": ans, "source": "system_date", "question_type": "variable", "chat_id": chat_id_normalizado}
                
            ans = f"No encontré información actualizada sobre: {pregunta}"
            save_message(chat_id_normalizado, "assistant", ans, source="internet", question_type="variable")
            return {"answer": ans, "source": "internet", "question_type": "variable", "chat_id": chat_id_normalizado}

        # Formular respuesta final con info de internet
        prompt_final = (
            f"Pregunta: {pregunta}\n"
            f"Información obtenida de internet: {info_internet}\n"
            "Responde a la pregunta usando la información de internet de forma clara, natural y corta."
        )
        final_messages = [
            {
                "role": "system",
                "content": (
                    "Redacta una respuesta amigable, corta, natural y directa basada estrictamente en la información de internet proporcionada. "
                    "Evita explicaciones largas, responde con exactitud lo solicitado."
                ),
            },
        ] + history + [{"role": "user", "content": prompt_final}]
        
        try:
            respuesta_final = _chat_with_ollama(final_messages, temperature=0.2)
            answer_text = respuesta_final or info_internet
        except Exception as e:
            logger.warning("Error calling Ollama for final variable answer: %s", e)
            answer_text = info_internet

        # Guardar en base de datos
        guardar_en_db_qa(pregunta, answer_text, "variable", chat_id_normalizado)
        save_message(chat_id_normalizado, "assistant", answer_text, source="internet", question_type="variable")
        return {"answer": answer_text, "source": "internet", "question_type": "variable", "chat_id": chat_id_normalizado}

    # Si es "fija"
    # RAG: Buscar contexto
    contexto = obtener_contexto_db(pregunta)
    
    prompt_evaluacion = (
        f"Contexto: {contexto}\n"
        f"Pregunta: {pregunta}\n"
        "Instrucción: Intenta responder de manera clara y natural a la pregunta usando el contexto o tu conocimiento seguro. "
        "Si no puedes responder con precisión y seguridad usando el contexto proporcionado, responde únicamente con la palabra: NO_SE"
    )

    eval_messages = [
        {
            "role": "system",
            "content": (
                "Responde solo a la pregunta del usuario de manera clara, natural y concisa."
            ),
        },
    ] + history + [{"role": "user", "content": prompt_evaluacion}]

    try:
        respuesta_llm = _chat_with_ollama(eval_messages, temperature=0.0)
    except Exception as e:
        logger.warning("Error calling Ollama in fija mode: %s", e)
        respuesta_llm = "NO_SE"

    if "NO_SE" not in respuesta_llm.upper() and respuesta_llm.strip():
        # LLM conoce la respuesta. Guardamos en DB y retornamos.
        guardar_en_db_qa(pregunta, respuesta_llm, "fija", chat_id_normalizado)
        save_message(chat_id_normalizado, "assistant", respuesta_llm, source="local_or_model", question_type="fija")
        return {
            "answer": respuesta_llm,
            "source": "local_or_model",
            "question_type": "fija",
            "chat_id": chat_id_normalizado,
        }

    # Si no sabe (NO_SE), buscamos en internet
    query_busqueda = generar_query_busqueda(pregunta, history)
    info_internet = buscar_en_internet(query_busqueda)
    
    if not info_internet:
        ans = "No se encontró información en internet."
        save_message(chat_id_normalizado, "assistant", ans, source="internet", question_type="fija")
        return {"answer": ans, "source": "internet", "question_type": "fija", "chat_id": chat_id_normalizado}

    # Formular respuesta final amigable con info de internet
    prompt_final = f"Pregunta: {pregunta}\nInformación obtenida de internet: {info_internet}\nResponde a la pregunta usando la información de internet de forma clara, natural y detallada."
    final_messages = [
        {
            "role": "system",
            "content": (
                "Redacta una respuesta amigable, completa y bien explicada usando los datos recuperados de internet. "
                "Asegúrate de dar una respuesta detallada e informativa que responda completamente a la inquietud del usuario."
            ),
        },
    ] + history + [{"role": "user", "content": prompt_final}]
    
    try:
        respuesta_final = _chat_with_ollama(final_messages, temperature=0.2)
        answer_text = respuesta_final or info_internet
    except Exception as e:
        logger.warning("Error calling Ollama for final fija answer: %s", e)
        answer_text = info_internet

    # Guardar en base de datos
    guardar_en_db_qa(pregunta, answer_text, "fija", chat_id_normalizado)
    save_message(chat_id_normalizado, "assistant", answer_text, source="internet", question_type="fija")
    return {"answer": answer_text, "source": "internet", "question_type": "fija", "chat_id": chat_id_normalizado}


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
    
    # 1. Obtener historial conversacional PREVIO (antes de guardar la pregunta actual)
    history = _get_conversational_history(chat_id_normalizado, limit=8)
    print(f"[LOG DE AGENTE] Historial conversacional cargado ({len(history)} mensajes previos).")
    
    # 2. Guardar la pregunta actual en la base de datos
    save_message(chat_id_normalizado, "user", pregunta)
    
    titulo_chat = pregunta.strip().replace("\n", " ")[:60]
    if titulo_chat:
        set_chat_title(chat_id_normalizado, titulo_chat)

    # 3. Revisar la base de datos primero (coincidencia de Q&A)
    yield {"type": "status", "content": "Revisando base de datos..."}
    coincidencia = buscar_en_db_qa(pregunta)
    if coincidencia:
        print(f"[LOG DE AGENTE] [DB COINCIDENCIA] Retornando respuesta guardada de inmediato...")
        answer = coincidencia["answer"]
        source = coincidencia["source"]
        q_type = coincidencia["question_type"]
        
        # Transmitimos la respuesta palabra por palabra
        for t in _yield_string_as_tokens(answer):
            yield t
            
        save_message(chat_id_normalizado, "assistant", answer, source=source, question_type=q_type)
        yield {"type": "done", "chat_id": chat_id_normalizado, "source": source, "answer": answer}
        print(f"[LOG DE AGENTE] ===== FIN DE PROCESAMIENTO =====")
        return

    # 4. Clasificar la pregunta pasándole el historial pre-cargado
    yield {"type": "status", "content": "Clasificando pregunta..."}
    print(f"[LOG DE AGENTE] Ejecutando clasificación por IA con contexto...")
    tipo_pregunta = clasificar_pregunta(pregunta, chat_id=chat_id_normalizado, history=history)
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
        ] + history + [{"role": "user", "content": pregunta}]
        
        print(f"[LOG DE AGENTE] [SALUDO] Enviando prompt a Ollama con contexto...")
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
        ] + history + [{"role": "user", "content": pregunta}]
        
        print(f"[LOG DE AGENTE] [INCOMPLETA] Enviando prompt a Ollama con contexto...")
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
                    "Responde con exactitud y de forma directa a la pregunta del usuario. "
                    "Si no tienes o no conoces la información exacta y actual en tu conocimiento interno para responder con precisión, responde exactamente con la palabra: NO_SE"
                ),
            },
        ] + history + [{"role": "user", "content": pregunta}]
        
        print(f"[LOG DE AGENTE] [VARIABLE] Evaluando si el modelo conoce la respuesta de inmediato con contexto...")
        respuesta_llm = ""
        try:
            respuesta_llm = _chat_with_ollama(messages, temperature=0.0)
            print(f"[LOG DE AGENTE] [VARIABLE] Respuesta tentativa del LLM: '{respuesta_llm}'")
        except Exception as e:
            print(f"[LOG DE AGENTE] [VARIABLE] Error al llamar Ollama: {e}")
            respuesta_llm = "NO_SE"

        if "NO_SE" not in respuesta_llm.upper() and respuesta_llm.strip():
            print(f"[LOG DE AGENTE] [VARIABLE] LLM conoce la respuesta. Transmitiendo en tiempo real y guardando...")
            for t in _yield_string_as_tokens(respuesta_llm):
                yield t
            # Guardamos en la base de datos
            guardar_en_db_qa(pregunta, respuesta_llm, "variable", chat_id_normalizado)
            save_message(chat_id_normalizado, "assistant", respuesta_llm, source="llm_variable", question_type="variable")
            yield {"type": "done", "chat_id": chat_id_normalizado, "source": "llm_variable", "answer": respuesta_llm}
            print(f"[LOG DE AGENTE] ===== FIN DE PROCESAMIENTO =====")
            return

        print(f"[LOG DE AGENTE] [VARIABLE] El LLM respondió NO_SE o requiere datos. Iniciando búsqueda web con contexto...")
        yield {"type": "status", "content": "Buscando en Internet..."}
        
        # Generar query de búsqueda optimizado con el contexto del historial
        query_busqueda = generar_query_busqueda(pregunta, history)
        print(f"[LOG DE AGENTE] [VARIABLE] Query de búsqueda generado con contexto: '{query_busqueda}'")
        
        print(f"[LOG DE AGENTE] [VARIABLE] Ejecutando búsqueda en DuckDuckGo para: '{query_busqueda}'")
        info_internet = buscar_en_internet(query_busqueda)
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
            f"Información obtenida de internet: {info_internet}\n"
            "Responde a la pregunta usando la información de internet de forma clara, natural y corta."
        )
        final_messages = [
            {
                "role": "system",
                "content": (
                    "Redacta una respuesta amigable, corta, natural y directa basada estrictamente en la información de internet proporcionada. "
                    "Evita explicaciones largas, responde con exactitud lo solicitado."
                ),
            },
        ] + history + [{"role": "user", "content": prompt_final}]
        
        print(f"[LOG DE AGENTE] [VARIABLE] Enviando información de Internet al LLM para redacción final con contexto...")
        try:
            response_gen = _chat_with_ollama(final_messages, temperature=0.2, stream=True)
            full_text = ""
            for chunk in response_gen:
                token = chunk.get("message", {}).get("content", "")
                if token:
                    full_text += token
                    yield {"type": "token", "content": token}
            
            print(f"[LOG DE AGENTE] [VARIABLE] Redacción final completada: '{full_text}'")
            # Guardamos en la base de datos
            guardar_en_db_qa(pregunta, full_text or info_internet, "variable", chat_id_normalizado)
            save_message(chat_id_normalizado, "assistant", full_text or info_internet, source="internet", question_type="variable")
            yield {"type": "done", "chat_id": chat_id_normalizado, "source": "internet", "answer": full_text or info_internet}
            print(f"[LOG DE AGENTE] ===== FIN DE PROCESAMIENTO =====")
            return
        except Exception as e:
            print(f"[LOG DE AGENTE] [ERROR VARIABLE] Error al redactar respuesta: {e}. Enviando datos puros de internet.")
            ans = info_internet
            for t in _yield_string_as_tokens(ans):
                yield t
            # Guardamos en la base de datos
            guardar_en_db_qa(pregunta, ans, "variable", chat_id_normalizado)
            save_message(chat_id_normalizado, "assistant", ans, source="internet", question_type="variable")
            yield {"type": "done", "chat_id": chat_id_normalizado, "source": "internet", "answer": ans}
            print(f"[LOG DE AGENTE] ===== FIN DE PROCESAMIENTO =====")
            return

    # Pregunta Fija (RAG)
    print(f"[LOG DE AGENTE] [FIJA] Buscando en base de conocimientos ChromaDB...")
    yield {"type": "status", "content": "Buscando en base de conocimientos..."}
    contexto = obtener_contexto_db(pregunta)

    prompt_evaluacion = (
        f"Contexto: {contexto}\n"
        f"Pregunta: {pregunta}\n"
        "Instrucción: Intenta responder de manera clara y natural a la pregunta usando el contexto o tu conocimiento seguro. "
        "Si no puedes responder con precisión y seguridad usando el contexto proporcionado, responde únicamente con la palabra: NO_SE"
    )

    eval_messages = [
        {
            "role": "system",
            "content": (
                "Responde solo a la pregunta del usuario de manera clara, natural y concisa."
            ),
        },
    ] + history + [{"role": "user", "content": prompt_evaluacion}]

    print(f"[LOG DE AGENTE] [FIJA] Enviando pregunta de evaluación al LLM...")
    respuesta_llm = ""
    try:
        respuesta_llm = _chat_with_ollama(eval_messages, temperature=0.0)
        print(f"[LOG DE AGENTE] [FIJA] Respuesta del LLM a evaluación: '{respuesta_llm}'")
    except Exception as e:
        print(f"[LOG DE AGENTE] [FIJA] Error al llamar a Ollama para evaluación: {e}")
        respuesta_llm = "NO_SE"

    if "NO_SE" not in respuesta_llm.upper() and respuesta_llm.strip():
        print(f"[LOG DE AGENTE] [FIJA] El LLM conoce la respuesta segura. Transmitiendo en tiempo real y guardando...")
        for t in _yield_string_as_tokens(respuesta_llm):
            yield t
        # Guardamos en la base de datos
        guardar_en_db_qa(pregunta, respuesta_llm, "fija", chat_id_normalizado)
        save_message(chat_id_normalizado, "assistant", respuesta_llm, source="local_or_model", question_type="fija")
        yield {"type": "done", "chat_id": chat_id_normalizado, "source": "local_or_model", "answer": respuesta_llm}
        print(f"[LOG DE AGENTE] ===== FIN DE PROCESAMIENTO =====")
        return

    print(f"[LOG DE AGENTE] [FIJA] El LLM respondió NO_SE. Iniciando búsqueda web de fallback con contexto...")
    yield {"type": "status", "content": "Buscando información en Internet..."}
    
    # Generar query de búsqueda optimizado con el contexto del historial
    query_busqueda = generar_query_busqueda(pregunta, history)
    print(f"[LOG DE AGENTE] [FIJA] Query de búsqueda generado con contexto: '{query_busqueda}'")
    
    print(f"[LOG DE AGENTE] [FIJA] Ejecutando búsqueda en DuckDuckGo para: '{query_busqueda}'")
    info_internet = buscar_en_internet(query_busqueda)
    print(f"[LOG DE AGENTE] [FIJA] Información de Internet recuperada: {info_internet[:200]}...")

    if info_internet:
        yield {"type": "status", "content": "Redactando respuesta desde Internet..."}
        prompt_final = f"Pregunta: {pregunta}\nInformación obtenida de internet: {info_internet}\nResponde a la pregunta usando la información de internet de forma clara, natural y detallada."
        final_messages = [
            {
                "role": "system",
                "content": (
                    "Redacta una respuesta amigable, completa y bien explicada usando los datos recuperados de internet. "
                    "Asegúrate de dar una respuesta detallada e informativa que responda completamente a la inquietud del usuario."
                ),
            },
        ] + history + [{"role": "user", "content": prompt_final}]
        
        print(f"[LOG DE AGENTE] [FIJA] Enviando información de Internet al LLM para redacción final con contexto...")
        try:
            response_gen = _chat_with_ollama(final_messages, temperature=0.2, stream=True)
            full_text = ""
            for chunk in response_gen:
                token = chunk.get("message", {}).get("content", "")
                if token:
                    full_text += token
                    yield {"type": "token", "content": token}
            
            print(f"[LOG DE AGENTE] [FIJA] Redacción final completada: '{full_text}'")
            # Guardamos en la base de datos
            guardar_en_db_qa(pregunta, full_text or info_internet, "fija", chat_id_normalizado)
            save_message(chat_id_normalizado, "assistant", full_text or info_internet, source="internet", question_type="fija")
            yield {"type": "done", "chat_id": chat_id_normalizado, "source": "internet", "answer": full_text or info_internet}
            print(f"[LOG DE AGENTE] ===== FIN DE PROCESAMIENTO =====")
            return
        except Exception as e:
            print(f"[LOG DE AGENTE] [ERROR FIJA] Error al redactar respuesta final: {e}. Enviando datos de internet puros.")
            ans = info_internet
            for t in _yield_string_as_tokens(ans):
                yield t
            # Guardamos en la base de datos
            guardar_en_db_qa(pregunta, ans, "fija", chat_id_normalizado)
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
