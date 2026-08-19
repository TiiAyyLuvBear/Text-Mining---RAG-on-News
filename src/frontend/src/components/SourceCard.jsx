export function SourceCard({ source, index }) {
  const contexts = source.contexts || [source];
  const evidenceLabel = contexts.length === 1 ? "Xem đoạn bằng chứng" : `Xem ${contexts.length} đoạn bằng chứng`;
  return <article className="source-card" id={`source-${index}`}>
    <p className="source-number"><span>[{index}]</span> Nguồn tham khảo</p>
    {source.articleId && <p className="article-id">Bài {source.articleId}</p>}
    {source.title && <h3>{source.title}</h3>}
    {(source.category || source.date) && <p className="source-meta">{[source.category, source.date].filter(Boolean).join(" · ")}</p>}
    {source.snippet && <p className="source-snippet">{source.snippet}</p>}
    <p className="evidence-count">{contexts.length} đoạn được sử dụng</p>
    {source.url && <a className="source-link" href={source.url} target="_blank" rel="noreferrer">Xem bài gốc ↗</a>}
    <details><summary>{evidenceLabel}</summary><div className="evidence-list">{contexts.map((context, contextIndex) => <section key={context.chunk_id || contextIndex}><p className="evidence-label">Đoạn {contextIndex + 1}</p><p className="context-copy">{context.text}</p></section>)}</div></details>
  </article>;
}
