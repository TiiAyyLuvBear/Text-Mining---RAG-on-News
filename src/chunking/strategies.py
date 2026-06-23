from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator


WORD_RE = re.compile(r"\S+", flags=re.UNICODE)
SENTENCE_RE = re.compile(r"[^.!?…]+(?:[.!?…]+|$)", flags=re.UNICODE)
PARAGRAPH_RE = re.compile(r"\n+")

DEFAULT_STRATEGIES = ("token", "langchain_recursive", "llamaindex", "structured")


@dataclass(frozen=True)
class ChunkingConfig:
    strategy: str = "token"
    chunk_size: int = 450
    overlap: int = 80
    min_chunk_tokens: int = 80
    small_article_chars: int = 1_000
    max_chunks_per_article: int | None = None
    inject_title_description: bool = True


@dataclass(frozen=True)
class TextUnit:
    text: str
    start: int
    end: int
    token_count: int
    implementation: str
    structure: str = "content"


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    article_id: str
    strategy: str
    text: str
    chunk_text: str
    metadata: dict[str, object]


@dataclass
class ChunkingStats:
    articles_read: int = 0
    chunks_written: int = 0
    short_articles: int = 0
    truncated_articles: int = 0
    empty_articles: int = 0
    total_chunk_tokens: int = 0
    total_chunk_chars: int = 0
    min_chunk_tokens: int | None = None
    max_chunk_tokens: int = 0
    min_chunk_chars: int | None = None
    max_chunk_chars: int = 0
    elapsed_seconds: float = 0.0

    def record_chunk(self, chunk: Chunk) -> None:
        token_count = int(chunk.metadata["token_count"])
        char_count = int(chunk.metadata["chunk_chars"])
        self.chunks_written += 1
        self.total_chunk_tokens += token_count
        self.total_chunk_chars += char_count
        self.min_chunk_tokens = token_count if self.min_chunk_tokens is None else min(self.min_chunk_tokens, token_count)
        self.max_chunk_tokens = max(self.max_chunk_tokens, token_count)
        self.min_chunk_chars = char_count if self.min_chunk_chars is None else min(self.min_chunk_chars, char_count)
        self.max_chunk_chars = max(self.max_chunk_chars, char_count)

    def to_dict(self) -> dict[str, int | float | None]:
        payload = asdict(self)
        payload["avg_chunks_per_article"] = (
            round(self.chunks_written / self.articles_read, 4) if self.articles_read else 0
        )
        payload["avg_chunk_tokens"] = (
            round(self.total_chunk_tokens / self.chunks_written, 4) if self.chunks_written else 0
        )
        payload["avg_chunk_chars"] = (
            round(self.total_chunk_chars / self.chunks_written, 4) if self.chunks_written else 0
        )
        payload["chunks_per_second"] = (
            round(self.chunks_written / self.elapsed_seconds, 4) if self.elapsed_seconds > 0 else None
        )
        return payload


def estimate_tokens(text: str) -> int:
    return len(WORD_RE.findall(text))


def split_token_baseline(text: str, *, chunk_size: int, overlap: int) -> list[TextUnit]:
    matches = list(WORD_RE.finditer(text))
    if not matches:
        return []
    step = max(1, chunk_size - overlap)
    units: list[TextUnit] = []
    for start_idx in range(0, len(matches), step):
        end_idx = min(len(matches), start_idx + chunk_size)
        start = matches[start_idx].start()
        end = matches[end_idx - 1].end()
        units.append(
            TextUnit(
                text=text[start:end],
                start=start,
                end=end,
                token_count=end_idx - start_idx,
                implementation="internal_token_window",
            )
        )
        if end_idx == len(matches):
            break
    return units


def split_langchain_recursive(text: str, *, chunk_size: int, overlap: int) -> list[TextUnit]:
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
        )
        parts = splitter.split_text(text)
        implementation = "langchain_text_splitters.RecursiveCharacterTextSplitter"
    except ImportError:
        parts = _recursive_character_split(text, chunk_size=chunk_size, overlap=overlap)
        implementation = "internal_recursive_character_fallback"
    return _parts_to_units(text, parts, implementation=implementation)


def split_llamaindex_sentence(text: str, *, chunk_size: int, overlap: int) -> list[TextUnit]:
    try:
        from llama_index.core.node_parser import SentenceSplitter

        splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
        parts = splitter.split_text(text)
        implementation = "llama_index.core.node_parser.SentenceSplitter"
    except ImportError:
        parts = _sentence_window_split(text, chunk_size=chunk_size, overlap=overlap)
        implementation = "internal_sentence_splitter_fallback"
    return _parts_to_units(text, parts, implementation=implementation)


def split_structured(record: dict[str, object], config: ChunkingConfig) -> list[TextUnit]:
    content = str(record.get("content") or "")
    sections = _content_sections(content)
    units: list[TextUnit] = []
    for section_name, section_text, offset in sections:
        section_units = _sentence_window_split(section_text, chunk_size=config.chunk_size, overlap=config.overlap)
        for unit in _parts_to_units(section_text, section_units, implementation="internal_structured_sentence_window"):
            units.append(
                TextUnit(
                    text=unit.text,
                    start=offset + unit.start,
                    end=offset + unit.end,
                    token_count=unit.token_count,
                    implementation=unit.implementation,
                    structure=section_name,
                )
            )
    return units


def chunk_article(record: dict[str, object], config: ChunkingConfig) -> list[Chunk]:
    article_id = str(record["article_id"])
    content = str(record.get("content") or "")
    metadata = dict(record.get("metadata") or {})
    title = str(record.get("title") or metadata.get("title") or "")
    description = str(record.get("description") or metadata.get("description") or "")
    category = str(record.get("category") or metadata.get("category") or "")

    if not content.strip():
        chunk_units: list[TextUnit] = []
    elif len(content) <= config.small_article_chars:
        chunk_units = [
            TextUnit(
                content,
                0,
                len(content),
                estimate_tokens(content),
                implementation="single_small_article",
                structure="content",
            )
        ]
    elif config.strategy == "token":
        chunk_units = split_token_baseline(content, chunk_size=config.chunk_size, overlap=config.overlap)
    elif config.strategy == "langchain_recursive":
        chunk_units = split_langchain_recursive(content, chunk_size=config.chunk_size, overlap=config.overlap)
    elif config.strategy == "llamaindex":
        chunk_units = split_llamaindex_sentence(content, chunk_size=config.chunk_size, overlap=config.overlap)
    elif config.strategy == "structured":
        chunk_units = split_structured(record, config)
    else:
        raise ValueError(f"Unsupported chunking strategy: {config.strategy}")

    chunk_units = _merge_small_tail(chunk_units, min_chunk_tokens=config.min_chunk_tokens)

    truncated = False
    if config.max_chunks_per_article is not None and len(chunk_units) > config.max_chunks_per_article:
        chunk_units = chunk_units[: config.max_chunks_per_article]
        truncated = True

    chunks: list[Chunk] = []
    num_chunks = len(chunk_units)
    for index, unit in enumerate(chunk_units):
        chunk_text = unit.text.strip()
        embedding_text = build_embedding_text(
            title=title,
            description=description,
            category=category,
            chunk_text=chunk_text,
            inject_title_description=config.inject_title_description,
        )
        chunk_metadata = {
            **metadata,
            "article_id": article_id,
            "title": title,
            "description": description,
            "category": category,
            "strategy": config.strategy,
            "implementation": unit.implementation,
            "structure": unit.structure,
            "chunk_index": index,
            "num_chunks": num_chunks,
            "char_start": unit.start,
            "char_end": unit.end,
            "token_count": unit.token_count,
            "chunk_chars": len(chunk_text),
            "truncated_article": truncated,
        }
        chunks.append(
            Chunk(
                chunk_id=f"{article_id}_{config.strategy}_{index:04d}",
                article_id=article_id,
                strategy=config.strategy,
                text=embedding_text,
                chunk_text=chunk_text,
                metadata=chunk_metadata,
            )
        )
    return chunks


def _recursive_character_split(text: str, *, chunk_size: int, overlap: int) -> list[str]:
    parts = _split_by_separators(text, ["\n\n", "\n", ". ", "? ", "! ", " "], chunk_size)
    chunks: list[str] = []
    current = ""
    for part in parts:
        candidate = f"{current}{part}" if current else part
        if current and len(candidate) > chunk_size:
            chunks.append(current.strip())
            current = _tail_overlap(current, overlap) + part
        else:
            current = candidate
    if current.strip():
        chunks.append(current.strip())
    return chunks


def _split_by_separators(text: str, separators: list[str], chunk_size: int) -> list[str]:
    if len(text) <= chunk_size or not separators:
        return [text]
    separator = separators[0]
    if separator and separator in text:
        parts: list[str] = []
        raw_parts = text.split(separator)
        for index, part in enumerate(raw_parts):
            suffix = separator if index < len(raw_parts) - 1 else ""
            value = part + suffix
            if len(value) > chunk_size:
                parts.extend(_split_by_separators(value, separators[1:], chunk_size))
            elif value:
                parts.append(value)
        return parts
    return _split_by_separators(text, separators[1:], chunk_size)


def _tail_overlap(text: str, overlap: int) -> str:
    if overlap <= 0:
        return ""
    return text[-overlap:]


def _sentence_window_split(text: str, *, chunk_size: int, overlap: int) -> list[str]:
    sentences = [match.group(0).strip() for match in SENTENCE_RE.finditer(text) if match.group(0).strip()]
    if not sentences:
        return [text] if text.strip() else []
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for sentence in sentences:
        sentence_tokens = estimate_tokens(sentence)
        if current and current_tokens + sentence_tokens > chunk_size:
            chunks.append(" ".join(current).strip())
            current = _overlap_sentences(current, overlap)
            current_tokens = estimate_tokens(" ".join(current))
        if sentence_tokens > chunk_size:
            chunks.extend(unit.text for unit in split_token_baseline(sentence, chunk_size=chunk_size, overlap=overlap))
            current = []
            current_tokens = 0
            continue
        current.append(sentence)
        current_tokens += sentence_tokens
    if current:
        chunks.append(" ".join(current).strip())
    return chunks


def _overlap_sentences(sentences: list[str], overlap_tokens: int) -> list[str]:
    selected: list[str] = []
    token_count = 0
    for sentence in reversed(sentences):
        selected.append(sentence)
        token_count += estimate_tokens(sentence)
        if token_count >= overlap_tokens:
            break
    return list(reversed(selected))


def _parts_to_units(text: str, parts: Iterable[str | TextUnit], *, implementation: str) -> list[TextUnit]:
    units: list[TextUnit] = []
    cursor = 0
    for part in parts:
        if isinstance(part, TextUnit):
            part_text = part.text.strip()
        else:
            part_text = str(part).strip()
        if not part_text:
            continue
        start = text.find(part_text, cursor)
        if start < 0:
            start = text.find(part_text)
        if start < 0:
            start = cursor
        end = start + len(part_text)
        cursor = max(cursor, end)
        units.append(
            TextUnit(
                text=part_text,
                start=start,
                end=end,
                token_count=estimate_tokens(part_text),
                implementation=implementation,
            )
        )
    return units


def _content_sections(content: str) -> list[tuple[str, str, int]]:
    paragraphs = []
    cursor = 0
    for raw in PARAGRAPH_RE.split(content):
        start = content.find(raw, cursor)
        if start < 0:
            start = cursor
        end = start + len(raw)
        cursor = end
        text = raw.strip()
        if text:
            paragraphs.append((text, start))
    if not paragraphs:
        return []
    if len(paragraphs) == 1:
        return [("content", paragraphs[0][0], paragraphs[0][1])]
    return [(f"paragraph_{index:03d}", text, start) for index, (text, start) in enumerate(paragraphs)]


def _merge_small_tail(units: list[TextUnit], *, min_chunk_tokens: int) -> list[TextUnit]:
    if len(units) < 2 or min_chunk_tokens <= 0 or units[-1].token_count >= min_chunk_tokens:
        return units
    previous = units[-2]
    tail = units[-1]
    merged_text = _merge_overlapping_text(previous.text, tail.text)
    merged = TextUnit(
        text=merged_text,
        start=previous.start,
        end=tail.end,
        token_count=estimate_tokens(merged_text),
        implementation=previous.implementation,
        structure=previous.structure if previous.structure == tail.structure else f"{previous.structure}+{tail.structure}",
    )
    return [*units[:-2], merged]


def _merge_overlapping_text(left: str, right: str) -> str:
    left_words = left.strip().split()
    right_words = right.strip().split()
    max_overlap = min(len(left_words), len(right_words))
    overlap = 0
    for size in range(max_overlap, 0, -1):
        if left_words[-size:] == right_words[:size]:
            overlap = size
            break
    return " ".join([*left_words, *right_words[overlap:]]).strip()


def build_embedding_text(
    *,
    title: str,
    description: str,
    category: str,
    chunk_text: str,
    inject_title_description: bool,
) -> str:
    if not inject_title_description:
        return chunk_text
    return "\n".join(
        part
        for part in [
            f"Tiêu đề: {title}" if title else "",
            f"Mô tả: {description}" if description else "",
            f"Chuyên mục: {category}" if category else "",
            "Đoạn nội dung:",
            chunk_text,
        ]
        if part
    )


def chunk_to_json(chunk: Chunk) -> dict[str, object]:
    return {
        "chunk_id": chunk.chunk_id,
        "article_id": chunk.article_id,
        "strategy": chunk.strategy,
        "text": chunk.text,
        "chunk_text": chunk.chunk_text,
        "metadata": chunk.metadata,
    }


def read_jsonl(path: str | Path, *, limit: int | None = None) -> Iterator[dict[str, object]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                break
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl_row(handle, payload: dict[str, object]) -> None:
    handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    handle.write("\n")


def chunk_records(
    records: Iterable[dict[str, object]],
    output_path: str | Path,
    config: ChunkingConfig,
) -> ChunkingStats:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats = ChunkingStats()
    started = time.perf_counter()
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        for record in records:
            stats.articles_read += 1
            chunks = chunk_article(record, config)
            metadata = dict(record.get("metadata") or {})
            if metadata.get("is_short"):
                stats.short_articles += 1
            if not str(record.get("content") or "").strip():
                stats.empty_articles += 1
            if chunks and chunks[0].metadata.get("truncated_article"):
                stats.truncated_articles += 1
            for chunk in chunks:
                write_jsonl_row(handle, chunk_to_json(chunk))
                stats.record_chunk(chunk)
    stats.elapsed_seconds = round(time.perf_counter() - started, 6)
    return stats
