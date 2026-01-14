from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import re

@dataclass
class Chunk:
    text: str
    meta: Dict

header_re = re.compile(r"^(#{1,6})\s+(.*)\s*$")

def split_markdown_into_chunks(markdown_text: str, source_file: str, max_chars: int = 1800, overlap: int = 100) -> List[Chunk]:
    lines = markdown_text.splitlines()
    sections: List[Tuple[List[str], List[str]]] = []
    header_path: List[str] = []
    current_section_lines: List[str] = []

    def push_section():
        nonlocal current_section_lines
        if current_section_lines:
            sections.append((header_path.copy(), current_section_lines))
            current_section_lines = []

    i = 0
    while i < len(lines):
        line = lines[i]

        hm = header_re.match(line)
        if hm:
            push_section()
            level = len(hm.group(1))
            title = hm.group(2).strip()
            header_path[:] = header_path[: level - 1]
            header_path.append(title)
            i += 1
            continue

        if line.strip().startswith("```"):
            current_section_lines.append(line)
            i += 1
            while i < len(lines):
                current_section_lines.append(lines[i])
                if lines[i].strip().startswith("```"):
                    i += 1
                    break
                i += 1
            continue

        current_section_lines.append(line)
        i += 1

    push_section()

    chunks: List[Chunk] = []
    for sec_idx, (hp, content_lines) in enumerate(sections):
        content = "\n".join(content_lines).strip()
        if not content:
            continue

        parts = re.split(r"\n\s*\n", content)
        buf = ""
        part_id = 0

        def flush():
            nonlocal buf, part_id
            text = buf.strip()
            if text:
                chunks.append(Chunk(
                    text=text,
                    meta={
                        "source_file": source_file,
                        "header_path": hp,
                        "section_index": sec_idx,
                        "chunk_in_section": part_id,
                    }
                ))
            buf = ""
            part_id += 1

        for part in parts:
            part = part.strip()
            if not part:
                continue

            candidate = (buf + "\n\n" + part).strip() if buf else part
            if len(candidate) <= max_chars:
                buf = candidate
            else:
                if buf:
                    flush()

                if len(part) <= max_chars:
                    buf = part
                else:
                    MAX_ABS_PART = 200000
                    part = part[:MAX_ABS_PART]

                    start = 0
                    step = max_chars - overlap
                    if step <= 0:
                        step = max_chars

                    while start < len(part):
                        end = min(len(part), start + max_chars)
                        chunks.append(Chunk(
                            text=part[start:end],
                            meta={
                                "source_file": source_file,
                                "header_path": hp,
                                "section_index": sec_idx,
                                "chunk_in_section": part_id,
                            }
                        ))
                        part_id += 1
                        start += step

        if buf:
            flush()

    return chunks
