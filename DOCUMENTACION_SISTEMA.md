# Documentación Técnica del Sistema - Chatbot con Memoria RAG Global

Este documento proporciona una explicación detallada de la arquitectura, componentes, bases de datos y el flujo de toma de decisiones del chatbot RAG inteligente.

---

## 1. Arquitectura General y Stack Tecnológico

El sistema está construido como una aplicación web de IA de extremo a extremo (End-to-End) y funciona en local. Se compone de las siguientes tecnologías:

```mermaid
graph TD
    A[Frontend: HTML5 / CSS / Vanilla JS] <-->|HTTP / SSE Stream| B[Backend: FastAPI / Uvicorn]
    B <-->|Historial y Memoria Relacional| C[(SQLite: chat_history.db)]
    B <-->|Búsqueda Semántica Global| D[(ChromaDB: memoria_agente)]
    B <-->|Inferencia de Modelos| E[Ollama: Modelo Llama3]
    B -->|Búsquedas en Internet de Fallback| F[API DuckDuckGo: ddgs]
```

### Detalle del Stack
1. **Servidor Backend (Python 3.10+)**: 
   - **FastAPI**: Framwork web moderno y rápido para construir la API.
   - **Uvicorn**: Servidor ASGI de alto rendimiento para correr la aplicación.
2. **Frontend (Web UI)**:
   - **HTML5 & CSS**: Diseño moderno y responsivo con estética premium basada en el modo oscuro de Gemini (gradientes fluidos, efectos de hover, paneles colapsables).
   - **TailwindCSS (CDN)**: Utilidades de diseño rápido.
   - **JavaScript (Vanilla JS)**: Controla las animaciones de la UI, la comunicación por Server-Sent Events (SSE) para respuestas en streaming, el colapso del menú lateral y las ventanas de renombrado.
   - **Marked.js**: Biblioteca integrada para renderizar Markdown (listas, bloques de código, negritas, etc.) directamente en las burbujas del chat.
3. **Modelos de Lenguaje (LLM)**:
   - **Ollama**: Orquestador local para ejecutar modelos como `llama3:latest` o `llama3.1:latest` con latencia baja y sin depender de APIs de pago externas.
4. **Base de Datos Vectorial (RAG)**:
   - **ChromaDB**: Base de datos vectorial embebida que almacena los conocimientos adquiridos. Se utiliza para identificar si el usuario está realizando preguntas equivalentes semánticamente a algo que el agente ya aprendió o contestó antes.
5. **Base de Datos Relacional (Historial)**:
   - **SQLite**: Almacena de forma persistente los chats creados, el historial de mensajes de cada conversación y un registro histórico de entradas de memoria.
6. **Búsqueda Web**:
   - **ddgs (DuckDuckGo Search)**: Realiza búsquedas de fondo en internet si el LLM no conoce la respuesta a una pregunta.

---

## 2. Flujo de Decisión RAG (Toma de Decisiones)

El agente de IA implementa un flujo inteligente que prioriza el conocimiento existente sobre las llamadas costosas al LLM.

```mermaid
graph TD
    Start([Recibir Pregunta del Usuario]) --> DB_Check{1. ¿Existe en ChromaDB?}
    
    %% Caso Cache Hit
    DB_Check -->|SÍ: Distancia <= 0.4| Return_DB[Retornar Respuesta desde ChromaDB de inmediato] --> End
    
    %% Caso Cache Miss
    DB_Check -->|NO: Distancia > 0.4| Classify[2. Clasificar Pregunta mediante LLM]
    
    Classify --> Intention{¿Qué tipo de pregunta es?}
    
    Intention -->|Saludo / Incompleta| Direct_LLM[3. LLM responde directamente] --> End
    
    Intention -->|Fija o Variable| Query_LLM_RAG[3. Preguntar al LLM con contexto de la Base de Datos]
    
    Query_LLM_RAG --> LLM_Response{4. ¿El LLM conoce la respuesta?}
    
    %% LLM Sabe
    LLM_Response -->|SÍ: No responde 'NO_SE'| Save_QA[5. Guardar par Q&A en ChromaDB y SQLite] --> Render[Responder al Usuario] --> End
    
    %% LLM No Sabe (Fallback)
    LLM_Response -->|NO: Responde 'NO_SE'| Formulate_Search[5. LLM formula frase de búsqueda óptima]
    Formulate_Search --> Web_Search[6. Realizar búsqueda en internet con DDGS]
    Web_Search --> Synthesize[7. Enviar snippets web al LLM para formular respuesta amigable]
    Synthesize --> Save_Web_QA[8. Guardar par Q&A en ChromaDB y SQLite]
    Save_Web_QA --> Render --> End
```

---

## 3. Esquemas y Gestión de Bases de Datos

El sistema maneja dos motores de bases de datos en paralelo para fines distintos:

### A. Base de Datos Relacional (SQLite)
Se almacena localmente en el archivo [chat_history.db](file:///f:/Angelo%20Archivos/IA/chat_history.db). Contiene tres tablas principales gestionadas en [history_store.py](file:///f:/Angelo%20Archivos/IA/app/history_store.py):

#### 1. Tabla `chats`
Representa cada hilo de conversación creado por el usuario en el panel izquierdo.
* `chat_id` (TEXT, Primary Key): Identificador único UUID del chat.
* `title` (TEXT): Título amigable del chat (renombrable).
* `created_at` (TEXT): Timestamp ISO del momento de creación.
* `updated_at` (TEXT): Timestamp ISO de la última interacción, utilizado para ordenar los chats por orden de actualización descendente.

#### 2. Tabla `messages`
Almacena todos los mensajes de ida y vuelta de la conversación.
* `id` (INTEGER, Primary Key AUTOINCREMENT)
* `chat_id` (TEXT, Foreign Key): Referencia al chat correspondiente.
* `role` (TEXT): El rol del emisor (`user` o `assistant`).
* `content` (TEXT): El contenido en texto plano o markdown del mensaje.
* `source` (TEXT, Nullable): Fuente del mensaje (ej: `internet`, `database_memory`, `ollama`).
* `question_type` (TEXT, Nullable): Categoría de la pregunta (ej: `fija`, `variable`, `saludo`).
* `created_at` (TEXT): Timestamp ISO de creación del mensaje.

#### 3. Tabla `memory_entries`
Historial de datos recuperados y guardados en memoria asociativa.
* `id` (INTEGER, Primary Key AUTOINCREMENT)
* `chat_id` (TEXT, Foreign Key)
* `query` (TEXT): Pregunta del usuario.
* `content` (TEXT): Respuesta que se asoció a esa pregunta.
* `source` (TEXT): Origen de la información.
* `created_at` (TEXT)

---

### B. Base de Datos Vectorial (ChromaDB)
Se almacena de forma persistente en la carpeta [memoria_agente/](file:///f:/Angelo%20Archivos/IA/memoria_agente). Se utiliza para el almacenamiento y recuperación de conocimiento asociativo basado en similitud semántica.

* **Colección**: `conocimiento_general`.
* **Embeddings**: ChromaDB utiliza por defecto el modelo `all-MiniLM-L6-v2` para vectorizar textos a vectores de 384 dimensiones.
* **Documento Vectorizado**: El texto de la **Pregunta original** realizada por el usuario (ej: *"¿Cuál es la fórmula del agua?"*).
* **Metadatos asociados**:
  ```json
  {
    "answer": "La fórmula química del agua es H2O...",
    "chat_id": "9efa95fd-b783-460a-9f0c-60cdcb9bd5d8",
    "question_type": "fija",
    "type": "qa_pair"
  }
  ```

#### Búsqueda Global y Umbral de Similitud Semántica
* **Búsqueda Global**: Al buscar conocimiento en ChromaDB, no se aplica ningún filtro de `chat_id`. Esto significa que si el chatbot aprende algo en la *"Conversación A"*, el conocimiento se comparte de inmediato y estará disponible de forma global para responder preguntas similares en la *"Conversación B"*.
* **Umbrales de Similitud (Distance Thresholds)**: ChromaDB calcula la distancia semántica (por defecto, la distancia L2 al cuadrado) entre el vector de la pregunta nueva y los vectores guardados:
  * **Caché Directa (`distancia <= 0.4`)**: Si la distancia es menor o igual a `0.4`, representa preguntas semánticamente equivalentes pero con redacción ligeramente diferente. El agente recupera el valor de `metadata["answer"]` y responde de inmediato, saltándose la inferencia del LLM o búsquedas en internet.
  * **Recuperación de Contexto (`distancia <= 0.8`)**: Cuando el agente busca contexto para preguntas de tipo `fija`, exige que la distancia sea menor o igual a `0.8`. Si la distancia es mayor, se descarta el contexto por considerarlo ruidoso e irrelevante (por ejemplo, evitaría inyectar información de "sal" en una consulta sobre "CPU"), permitiendo que el LLM resuelva correctamente la pregunta basándose en el historial de la conversación.

---


## 4. Clasificación de Preguntas y Búsqueda en Internet

Cuando la pregunta no está registrada en la memoria caché global de ChromaDB, el backend entra al flujo normal del agente:

1. **Clasificación**: Se llama a Ollama pidiéndole que clasifique la pregunta del usuario en una de estas categorías:
   - `saludo`: Saludos corteses que no requieren base de datos o búsquedas.
   - `incompleta`: Preguntas sin sentido o fragmentadas.
   - `fija`: Preguntas sobre hechos generales, históricos o científicos estables (ej: *"¿Quién descubrió América?"*).
   - `variable`: Preguntas que dependen del tiempo o son de actualidad (ej: *"¿Dónde será el próximo mundial?"*).
2. **Manejo del Fallback (`"NO_SE"`)**:
   - En preguntas de tipo `fija` o `variable`, el agente le pregunta al LLM local. 
   - Diseñamos el prompt del sistema del LLM para que devuelva explícitamente la palabra clave **`NO_SE`** si no cuenta con información suficiente o actualizada sobre el tema.
   - Al detectar la respuesta `NO_SE`, el sistema invoca al LLM para que formule una palabra/frase de búsqueda optimizada para motores de búsqueda (ej. de *"¿Dónde será el mundial de fútbol 2026?"* extrae *"Mundial 2026 sedes ciudades"*).
   - Se ejecuta una búsqueda en DuckDuckGo usando la biblioteca `ddgs` de Python para traer los 3 resultados más relevantes.
   - Se le entrega esa información de internet estructurada al LLM para que redacte una respuesta final coherente al usuario.
   - **Aprendizaje Activo**: Esta respuesta generada a partir de internet se guarda de inmediato en ChromaDB y SQLite como un par Q&A para que en la próxima pregunta similar el agente ya la "sepa" directamente.

---

## 5. Estructura de Archivos del Proyecto

El repositorio está organizado de la siguiente manera:

```text
IA/
├── chat_history.db               # Base de datos SQLite (Chats y Mensajes)
├── memoria_agente/               # Directorio persistente de vectores de ChromaDB
├── requirements.txt              # Definición de paquetes pip (FastAPI, ddgs, ollama, etc.)
├── pyproject.toml                # Configuración de dependencias moderna para gestores como UV
├── README.md                     # Manual básico de uso y arranque
└── app/
    ├── main.py                   # Rutas y Endpoints HTTP / Event Streams (FastAPI)
    ├── agent.py                  # Lógica del Agente RAG, clasificación, búsquedas y ChromaDB
    ├── history_store.py          # Consultas relacionales para SQLite (Historial de Chats)
    ├── templates/
    │   ├── base.html             # Estructura HTML base con importaciones CSS/JS y CDN de Tailwind
    │   └── index.html            # UI principal, composición de Sidebar y área de chat
    └── static/
        ├── css/
        │   └── chat.css          # Estilos detallados del tema oscuro, transiciones y animaciones
        └── js/
            └── chat.js           # Lógica frontend (Renombrar, colapsar sidebar y llamadas a API)
```

---

## 6. Comandos Útiles de Operación

* **Instalar Dependencias**:
  ```bash
  pip install -r requirements.txt
  ```
  *(o utilizando `uv` para descargas ultra rápidas: `uv pip install -r requirements.txt`)*

* **Iniciar el Servidor Web (Local)**:
  ```bash
  uvicorn app.main:app --host 0.0.0.0 --port 8000
  ```

* **Descargar el modelo en Ollama**:
  ```bash
  ollama pull llama3:latest
  ```
