function embeddedField(text, prefix) {
  if (typeof text !== "string") return "";
  return text.split("\n").find((line) => line.startsWith(prefix))?.slice(prefix.length).trim() || "";
}

function firstValue(items, getter) {
  for (const item of items) {
    const value = getter(item);
    if (value !== undefined && value !== null && String(value).trim()) return value;
  }
  return "";
}

export function groupSources(contexts = []) {
  const groups = new Map();

  contexts.forEach((source, index) => {
    const stableId = source.article_id ?? source.document_id ?? source.source_id ?? source.url ?? source.source_url;
    const key = stableId !== undefined && stableId !== null && String(stableId).trim()
      ? `article:${stableId}`
      : `context:${source.chunk_id ?? index}`;

    if (!groups.has(key)) groups.set(key, { key, contexts: [] });
    groups.get(key).contexts.push(source);
  });

  return Array.from(groups.values()).map((group) => {
    const items = group.contexts;
    const articleId = firstValue(items, (item) => item.article_id ?? item.document_id ?? item.source_id);
    const title = firstValue(items, (item) => item.title || embeddedField(item.text, "Tiêu đề:"));
    const category = firstValue(items, (item) => item.category || item.outlet || item.publisher || embeddedField(item.text, "Chuyên mục:"));
    const date = firstValue(items, (item) => item.published_at || item.publish_date || item.date);
    const url = firstValue(items, (item) => item.url || item.source_url);
    const snippet = firstValue(items, (item) => item.snippet || item.description || embeddedField(item.text, "Mô tả:"));

    return { ...group, articleId, title, category, date, url, snippet };
  });
}

