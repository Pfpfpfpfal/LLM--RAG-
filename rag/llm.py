from __future__ import annotations
import requests
import time
from typing import List, Dict, Any


OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
DEFAULT_MODEL = "qwen2.5-coder:7b-instruct-q4_K_S"

def format_context(retrieved: List[Dict[str, Any]], max_chars: int = 6000) -> str:
    blocks = []
    total = 0
    for r in retrieved:
        meta = r["meta"]
        src = meta.get("source_file", "unknown")
        hp_list = meta.get("header_path", [])
        hp = " > ".join(hp_list) if isinstance(hp_list, list) else (hp_list or "")
        cid = meta.get("chunk_id", "")
        header = f"[{src} | {hp} | {cid}]".strip()

        txt = r["text"].strip()
        block = f"{header}\n{txt}\n"
        if total + len(block) > max_chars:
            break
        blocks.append(block)
        total += len(block)
    return "\n---\n".join(blocks)


def generate_with_ollama(
    query: str,
    context: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
) -> str:
    system = (
    "Ты — технический ассистент, который объясняет понятия простым языком. "
    "Отвечай по-русски. Используй ТОЛЬКО предоставленный контекст. "
    "Не добавляй информацию, которой нет в контексте. "
    "Если вопрос задан 'простыми словами' — объясняй на интуитивных примерах. "
    "В конце ответа обязательно добавь раздел 'Источники:' "
    "и перечисли chunk_id, которые использовал."
)


    prompt = f"""Вопрос:
    {query}

    Контекст (фрагменты заметок):
    {context}

    Задание:
    1) Объясни ответ простыми словами (2–4 предложения).
    2) Если уместно — добавь краткое техническое уточнение.
    3) Не уходи в темы, которых нет в контексте.
    4) В конце напиши 'Источники:' и укажи chunk_id.
    """

    payload = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "options": {"temperature": temperature},
        "stream": False,
    }

    print(f"LLM: calling ollama model={model}, prompt_chars={len(prompt)}, ts={time.time()}", flush=True)
    resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    print(f"LLM: got response chars={len(data.get('response',''))}", flush=True)
    return (data.get("response") or "").strip()
