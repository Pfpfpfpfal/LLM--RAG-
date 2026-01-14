# Notes RAG — локальный LLM-ассистент по конспектам  
(Hybrid RAG: BM25 + Qdrant + Local LLM)

Локальный RAG-ассистент для поиска по собственным Markdown-конспектам и
генерации ответа **с помощью локальной LLM** с обязательными цитатами.

Проект работает полностью локально:
- FastAPI — backend API
- Streamlit — UI
- Qdrant (Docker) — vector search
- BM25 — keyword retrieval
- Ollama — локальный inference-сервер для LLM

## Возможности

- Индексация Markdown-заметок с учётом заголовков (`#`, `##`, …) и разбиение на чанки
- Hybrid retrieval:
  - **BM25** (ключевые слова) — устойчив к точным терминам/аббревиатурам
  - **Vector search** в **Qdrant** (семантика) через **fastembed** (без PyTorch)
- Пост-фильтрация результатов:
  - “score gap” фильтр (отсекает мусорные фрагменты)
  - ограничение “1 чанк на файл” (diversity)
- Ответ с цитатами: показывает источники (`file`, `header_path`, `chunk_id`)
- UI на Streamlit + API на FastAPI
- Генеративные ответы поверх retrieval (LLM-backed RAG):
  - локальная LLM через **Ollama**
  - модель: `qwen2.5-coder:7b-instruct-q4_K_S`
  - строгий режим: ответ **только по найденному контексту**
- Два режима ответа:
  - `extractive` — без LLM (только найденные фрагменты)
  - `llm` — генерация ответа поверх retrieved-контекста

---

## Архитектура

1) `rag/ingest.py`
- читает `data/notes/**/*.md`
- режет на чанки (с метаданными)
- сохраняет `data/index/chunks.jsonl`
- считает эмбеддинги (fastembed) и загружает в Qdrant (`collection`)
- записывает `data/index/embed_model.txt` и `data/index/qdrant_collection.txt`

2) `rag/retrieval.py`
- загружает `chunks.jsonl`
- строит BM25 по чанкам
- делает vector search в Qdrant + BM25 search локально
- сливает результаты (weighted merge)
- формирует extractive-answer и цитаты

2.1) `rag/llm.py`
- формирует контекст из top-K чанков
- отправляет запрос в Ollama по HTTP
- получает сгенерированный ответ от LLM
- обеспечивает генерацию только на основе retrieval-контекста


3) `app/api.py`
- endpoint `POST /ask`

4) `app/ui.py`
- UI для вопросов/ответов + показ найденных фрагментов

---

## Требования

- Python 3.10+ 

---

## Локальная LLM (Ollama)

Для генеративных ответов используется локальная LLM, запускаемая через **Ollama**.

### Выбранная модель

- **Модель:** `qwen2.5-coder:7b-instruct-q4_K_S`
- Тип: instruction-tuned coder model
- Квантование: `q4_K_S`
- Причины выбора:
  - хорошо работает с техническими текстами и документацией
  - устойчиво следует инструкциям
  - меньше склонна к галлюцинациям
  - стабильно работает на CPU

Модель используется **только после retrieval** и получает на вход строго отфильтрованный контекст

---

### Установка Ollama и модели

1. Установить Ollama:
   https://ollama.com

2. Загрузить модель:

```bash
ollama pull qwen2.5-coder:7b-instruct-q4_K_S
```

## Быстрый старт

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

```powershell
pip install -r requirements.txt
```

```powershell
docker compose up -d
```

### Индексация (ingest)

**Перед первым запуском или после изменения модели/чанкинга:**
```powershell
python -u -m rag.ingest
```

Что создаётся:

- data/index/chunks.jsonl — все чанки + мета
- data/index/manifests.json — хэши файлов
- data/index/embed_model.txt — имя embedding-модели
- data/index/qdrant_collection.txt — имя Qdrant-коллекции
- коллекция в Qdrant с векторами

### Запуск API
```powershell
uvicorn app.api:app --reload
```

### Запуск UI (Streamlit)

**Во втором терминале:**

```powershell
streamlit run app/ui.py
```