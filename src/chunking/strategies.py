from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator


WORD_RE = re.compile(r"\S+", flags=re.UNICODE)
SENTENCE_RE = re.compile(r"[^.!?…]+(?:[.!?…]+|$)", flags=re.UNICODE)
PARAGRAPH_RE = re.compile(r"\n+")


@dataclass(frozen=True)
class ChunkingConfig:
    strategy: str = "sentence"
    chunk_size: int = 450
    overlap: int = 80
    sentence_overlap: int = 2
    min_chunk_tokens: int = 80
    small_article_chars: int = 1_000
    long_article_chars: int = 20_000
    max_chunks_per_article: int | None = None
    inject_title_description: bool = True


@dataclass(frozen=True)
class TextUnit:
    text: str
    start: int
    end: int
    token_count: int


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
    long_articles: int = 0
    truncated_long_articles: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def estimate_tokens(text: str) -> int:
    return len(WORD_RE.findall(text))


def split_sentences(text: str) -> list[TextUnit]:
    units: list[TextUnit] = []
    for match in SENTENCE_RE.finditer(text):
        unit_text = match.group(0).strip()
        if not unit_text:
            continue
        units.append(
            TextUnit(
                text=unit_text,
                start=match.start(),
                end=match.end(),
                token_count=estimate_tokens(unit_text),
            )
        )
    return units or [TextUnit(text=text, start=0, end=len(text), token_count=estimate_tokens(text))]


def split_paragraphs(text: str) -> list[TextUnit]:
    units: list[TextUnit] = []
    cursor = 0
    for part in PARAGRAPH_RE.split(text):
        start = text.find(part, cursor)
        if start < 0:
            start = cursor
        end = start + len(part)
        cursor = end
        part = part.strip()
        if part:
            units.append(TextUnit(text=part, start=start, end=end, token_count=estimate_tokens(part)))
    return units or split_sentences(text)


def split_fixed_tokens(text: str, *, chunk_size: int, overlap: int) -> list[TextUnit]:
    matches = list(WORD_RE.finditer(text))
    if not matches:
        return []
    step = max(1, chunk_size - overlap)
    units: list[TextUnit] = []
    for start_idx in range(0, len(matches), step):
        end_idx = min(len(matches), start_idx + chunk_size)
        start = matches[start_idx].start()
        end = matches[end_idx - 1].end()
        units.append(TextUnit(text=text[start:end], start=start, end=end, token_count=end_idx - start_idx))
        if end_idx == len(matches):
            break
    return units


def merge_units(
    units: list[TextUnit],
    *,
    chunk_size: int,
    overlap_tokens: int,
    sentence_overlap: int,
    min_chunk_tokens: int,
) -> list[TextUnit]:
    if not units:
        return []
    chunks: list[TextUnit] = []
    current: list[TextUnit] = []
    current_tokens = 0
    index = 0
    while index < len(units):
        unit = units[index]
        if not current and unit.token_count > chunk_size:
            chunks.extend(
                split_fixed_tokens(unit.text, chunk_size=chunk_size, overlap=min(overlap_tokens, chunk_size - 1))
            )
            index += 1
            continue
        would_exceed = current and current_tokens + unit.token_count > chunk_size
        if would_exceed:
            chunks.append(_combine_units(current))
            current = _overlap_units(current, overlap_tokens=overlap_tokens, sentence_overlap=sentence_overlap)
            current_tokens = sum(item.token_count for item in current)
            if current_tokens >= chunk_size or current_tokens + unit.token_count > chunk_size:
                current = []
                current_tokens = 0
            continue
        current.append(unit)
        current_tokens += unit.token_count
        index += 1

    if current:
        if chunks and current_tokens < min_chunk_tokens:
            previous = chunks.pop()
            chunks.append(_combine_units([previous, *current]))
        else:
            chunks.append(_combine_units(current))
    return chunks


def _combine_units(units: list[TextUnit]) -> TextUnit:
    text = " ".join(unit.text.strip() for unit in units if unit.text.strip()).strip()
    return TextUnit(
        text=text,
        start=units[0].start,
        end=units[-1].end,
        token_count=sum(unit.token_count for unit in units),
    )


def _overlap_units(units: list[TextUnit], *, overlap_tokens: int, sentence_overlap: int) -> list[TextUnit]:
    if not units:
        return []
    if sentence_overlap > 0:
        return units[-sentence_overlap:]
    if overlap_tokens <= 0:
        return []
    selected: list[TextUnit] = []
    tokens = 0
    for unit in reversed(units):
        selected.append(unit)
        tokens += unit.token_count
        if tokens >= overlap_tokens:
            break
    return list(reversed(selected))


def chunk_article(record: dict[str, object], config: ChunkingConfig) -> list[Chunk]:
    article_id = str(record["article_id"])
    content = str(record.get("content") or "")
    metadata = dict(record.get("metadata") or {})
    title = str(record.get("title") or metadata.get("title") or "")
    description = str(record.get("description") or metadata.get("description") or "")
    category = str(record.get("category") or metadata.get("category") or "")

    if len(content) <= config.small_article_chars:
        chunk_units = [TextUnit(content, 0, len(content), estimate_tokens(content))]
    elif config.strategy == "fixed":
        chunk_units = split_fixed_tokens(content, chunk_size=config.chunk_size, overlap=config.overlap)
    elif config.strategy == "paragraph":
        paragraphs = split_paragraphs(content)
        # Many crawled articles have no paragraph breaks; fall back to sentence-aware.
        units = paragraphs if len(paragraphs) > 1 else split_sentences(content)
        chunk_units = merge_units(
            units,
            chunk_size=config.chunk_size,
            overlap_tokens=config.overlap,
            sentence_overlap=config.sentence_overlap,
            min_chunk_tokens=config.min_chunk_tokens,
        )
    elif config.strategy == "hybrid":
        units = split_paragraphs(content)
        if len(units) <= 1:
            units = split_sentences(content)
        chunk_units = merge_units(
            units,
            chunk_size=config.chunk_size if len(content) <= config.long_article_chars else max(config.chunk_size, 700),
            overlap_tokens=config.overlap,
            sentence_overlap=config.sentence_overlap,
            min_chunk_tokens=config.min_chunk_tokens,
        )
    elif config.strategy == "llamaindex":
        chunk_units = split_with_llamaindex(
            content,
            chunk_size=config.chunk_size,
            chunk_overlap=config.overlap,
        )
    elif config.strategy == "sentence":
        chunk_units = merge_units(
            split_sentences(content),
            chunk_size=config.chunk_size,
            overlap_tokens=config.overlap,
            sentence_overlap=config.sentence_overlap,
            min_chunk_tokens=config.min_chunk_tokens,
        )
    else:
        raise ValueError(f"Unsupported chunking strategy: {config.strategy}")

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


def split_with_llamaindex(text: str, *, chunk_size: int, chunk_overlap: int) -> list[TextUnit]:
    try:
        from llama_index.core.node_parser import SentenceSplitter
    except ImportError as exc:
        raise RuntimeError(
            "Strategy 'llamaindex' requires llama-index. Install dependencies from requirements.txt."
        ) from exc

    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    parts = splitter.split_text(text)
    units: list[TextUnit] = []
    cursor = 0
    for part in parts:
        part = part.strip()
        if not part:
            continue
        start = text.find(part, cursor)
        if start < 0:
            start = text.find(part)
        if start < 0:
            start = cursor
        end = start + len(part)
        cursor = max(end, cursor)
        units.append(TextUnit(text=part, start=start, end=end, token_count=estimate_tokens(part)))
    return units or [TextUnit(text=text, start=0, end=len(text), token_count=estimate_tokens(text))]


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
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        for record in records:
            stats.articles_read += 1
            chunks = chunk_article(record, config)
            metadata = dict(record.get("metadata") or {})
            if metadata.get("is_short"):
                stats.short_articles += 1
            if metadata.get("is_long"):
                stats.long_articles += 1
            if chunks and chunks[0].metadata.get("truncated_article"):
                stats.truncated_long_articles += 1
            for chunk in chunks:
                write_jsonl_row(handle, chunk_to_json(chunk))
                stats.chunks_written += 1
    return stats
