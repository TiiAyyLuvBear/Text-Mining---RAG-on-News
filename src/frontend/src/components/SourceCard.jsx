function field(text, prefix) { return text.split("\n").find((line) => line.startsWith(prefix))?.replace(prefix, "").trim(); }

export function SourceCard({ source, index }) {
  const text = source.text || "";
  const title = source.title || field(text, "Tiêu đề:") || `Bài viết ${source.article_id || index}`;
  const outlet = source.outlet || source.publisher || field(text, "Chuyên mục:") || "Vietnamese News";
  const date = source.published_at || source.date || "Chưa có ngày xuất bản";
  const url = source.url || source.source_url;
  const snippet = source.snippet || field(text, "Mô tả:") || text.slice(0, 220);
  return <article className="source-card" id={`source-${index}`}><p className="source-number">Bài viết {source.article_id || "N/A"}</p><h3>{title}</h3><p className="source-meta">{outlet} · {date}</p><p className="source-snippet">{snippet}</p>
    {url ? <a className="source-link" href={url} target="_blank" rel="noreferrer">Mở bài gốc ↗</a> : <span className="source-link unavailable">Liên kết bài gốc chưa có</span>}
    <details><summary>Xem context đã dùng</summary><p className="context-copy">{text}</p></details>
  </article>;
}
