from __future__ import annotations

import csv
import html
import json
import re
import sys
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


REQUIRED_COLUMNS = ("id", "title", "description", "content", "category")
PARAGRAPH_BREAK_RE = re.compile(r"\s*(?:\r\n|\r|\n)+\s*")
SPACE_RE = re.compile(r"[ \t\f\v]+")
HTML_TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)

def _raise_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit = limit // 10


_raise_csv_field_limit()

# Conservative fixes for crawl artifacts caused by stripped hyperlinks.
# These target common Vietnamese function words and news-source prefixes without
# splitting arbitrary mixed-case names.
JOINED_PREFIX_FIXES = (
    (re.compile(r"\bTheo(?=[A-ZÀ-ỸĐ])"), "Theo "),
    (re.compile(r"\bẢnh(?=[A-ZÀ-ỸĐ])"), "Ảnh "),
    (re.compile(r"\bVideo(?=[A-ZÀ-ỸĐ])"), "Video "),
    (re.compile(r"\bTin sao Việt(?=\d)"), "Tin sao Việt "),
)
JOINED_WORD_RE = re.compile(
    r"(?<=[a-zàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệ"
    r"ìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữự"
    r"ỳýỷỹỵđ0-9])"
    r"(?=("
    r"được|đã|đang|đến|để|đây|đó|đồng|đấu|"
    r"là|làm|lại|lên|lúc|"
    r"và|với|về|vào|vẫn|"
    r"của|các|cho|khi|như|này|năm|ngày|người|"
    r"trong|trên|trúng|tại|từ|sau|theo|"
    r"thanh toán|giao dịch|làng sao|dữ liệu|"
    r"gout|opera"
    r")\b)",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class PreprocessConfig:
    min_content_chars: int = 300
    long_content_chars: int = 20_000
    strip_urls: bool = False
    include_embedding_text: bool = True


@dataclass(frozen=True)
class ArticleRecord:
    article_id: str
    title: str
    description: str
    content: str
    category: str
    text: str
    metadata: dict[str, object]
    quality_flags: list[str]


@dataclass
class ProcessStats:
    rows_read: int = 0
    rows_written: int = 0
    missing_required: int = 0
    short_content: int = 0
    long_content: int = 0
    html_like: int = 0
    url_like: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def normalize_unicode(value: str) -> str:
    return unicodedata.normalize("NFC", value)


def normalize_whitespace(value: str) -> str:
    value = PARAGRAPH_BREAK_RE.sub("\n", value)
    value = SPACE_RE.sub(" ", value)
    return "\n".join(line.strip() for line in value.splitlines() if line.strip()).strip()


def fix_joined_words(value: str) -> str:
    for pattern, replacement in JOINED_PREFIX_FIXES:
        value = pattern.sub(replacement, value)
    previous = None
    while previous != value:
        previous = value
        value = JOINED_WORD_RE.sub(" ", value)
    return value


def clean_text(value: object, *, strip_urls: bool = False) -> str:
    if value is None:
        return ""
    text = str(value)
    if text.lower() == "nan":
        return ""
    text = normalize_unicode(text)
    text = html.unescape(text)
    text = HTML_TAG_RE.sub(" ", text)
    if strip_urls:
        text = URL_RE.sub(" ", text)
    text = fix_joined_words(text)
    text = normalize_whitespace(text)
    return text


def build_document_text(
    title: str,
    description: str,
    content: str,
    *,
    category: str,
    include_embedding_text: bool,
) -> str:
    parts = [
        f"Tiêu đề: {title}",
        f"Mô tả: {description}",
        f"Chuyên mục: {category}",
        "Nội dung:",
        content,
    ]
    text = "\n".join(part for part in parts if part)
    if include_embedding_text:
        return text
    return content


def quality_flags_for(
    *,
    raw_content: str,
    content: str,
    config: PreprocessConfig,
) -> list[str]:
    flags: list[str] = []
    if len(content) < config.min_content_chars:
        flags.append("short_content")
    if len(content) > config.long_content_chars:
        flags.append("long_content")
    if HTML_TAG_RE.search(raw_content) or "&" in raw_content and re.search(r"&[a-z]+;", raw_content):
        flags.append("html_like")
    if URL_RE.search(raw_content):
        flags.append("url_like")
    return flags


def preprocess_article(row: dict[str, object], config: PreprocessConfig | None = None) -> ArticleRecord:
    config = config or PreprocessConfig()
    raw_content = "" if row.get("content") is None else str(row.get("content"))
    title = clean_text(row.get("title"), strip_urls=config.strip_urls)
    description = clean_text(row.get("description"), strip_urls=config.strip_urls)
    content = clean_text(row.get("content"), strip_urls=config.strip_urls)
    category = clean_text(row.get("category"), strip_urls=config.strip_urls)
    article_id = clean_text(row.get("id"), strip_urls=True)
    text = build_document_text(
        title,
        description,
        content,
        category=category,
        include_embedding_text=config.include_embedding_text,
    )
    flags = quality_flags_for(raw_content=raw_content, content=content, config=config)
    metadata = {
        "article_id": article_id,
        "title": title,
        "description": description,
        "category": category,
        "content_chars": len(content),
        "title_chars": len(title),
        "description_chars": len(description),
        "is_short": "short_content" in flags,
        "is_long": "long_content" in flags,
    }
    return ArticleRecord(
        article_id=article_id,
        title=title,
        description=description,
        content=content,
        category=category,
        text=text,
        metadata=metadata,
        quality_flags=flags,
    )


def validate_columns(fieldnames: Iterable[str] | None) -> None:
    columns = set(fieldnames or [])
    missing = [column for column in REQUIRED_COLUMNS if column not in columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def write_jsonl_row(handle, payload: dict[str, object]) -> None:
    handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    handle.write("\n")


def record_to_json(record: ArticleRecord) -> dict[str, object]:
    return {
        "article_id": record.article_id,
        "title": record.title,
        "description": record.description,
        "content": record.content,
        "category": record.category,
        "text": record.text,
        "metadata": record.metadata,
        "quality_flags": record.quality_flags,
    }


def process_csv(
    input_path: str | Path,
    output_path: str | Path,
    *,
    review_output_path: str | Path | None = None,
    review_limit: int = 100,
    config: PreprocessConfig | None = None,
) -> ProcessStats:
    config = config or PreprocessConfig()
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats = ProcessStats()
    review_written = 0

    review_handle = None
    if review_output_path:
        review_path = Path(review_output_path)
        review_path.parent.mkdir(parents=True, exist_ok=True)
        review_handle = review_path.open("w", encoding="utf-8", newline="")

    try:
        with input_path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            validate_columns(reader.fieldnames)
            with output_path.open("w", encoding="utf-8", newline="") as target:
                for row in reader:
                    stats.rows_read += 1
                    if any(row.get(column) in (None, "") for column in REQUIRED_COLUMNS):
                        stats.missing_required += 1
                        continue

                    record = preprocess_article(row, config)
                    payload = record_to_json(record)
                    write_jsonl_row(target, payload)
                    stats.rows_written += 1

                    if "short_content" in record.quality_flags:
                        stats.short_content += 1
                    if "long_content" in record.quality_flags:
                        stats.long_content += 1
                    if "html_like" in record.quality_flags:
                        stats.html_like += 1
                    if "url_like" in record.quality_flags:
                        stats.url_like += 1

                    should_review = (
                        review_handle is not None
                        and review_written < review_limit
                        and (record.quality_flags or row.get("content") != record.content)
                    )
                    if should_review:
                        write_jsonl_row(
                            review_handle,
                            {
                                "article_id": record.article_id,
                                "title": record.title,
                                "category": record.category,
                                "quality_flags": record.quality_flags,
                                "raw_content_preview": str(row.get("content", ""))[:800],
                                "clean_content_preview": record.content[:800],
                            },
                        )
                        review_written += 1
    finally:
        if review_handle is not None:
            review_handle.close()

    return stats
