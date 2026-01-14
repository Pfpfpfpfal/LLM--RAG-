# Notes RAG — локальный ассистент по конспектам (Hybrid RAG: BM25 + Qdrant)

Локальный RAG-ассистент для поиска по собственным Markdown-конспектам и выдачи ответа с цитатами.
Работает полностью локально: FastAPI для API, Streamlit для UI, Qdrant в Docker для векторного поиска, BM25 локально для keyword retrieval.

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

3) `app/api.py`
- endpoint `POST /ask`

4) `app/ui.py`
- UI для вопросов/ответов + показ найденных фрагментов

---

## Требования

- Python 3.10+ 

---

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