export function EmptyState() {
  return <section className="state-card"><p className="kicker">News Desk</p><h2>Bắt đầu từ một câu hỏi.</h2><p>Chọn gợi ý hoặc viết câu hỏi tự nhiên. Nguồn dùng để trả lời sẽ xuất hiện tại đây.</p></section>;
}

export function LoadingSkeleton({ phase }) {
  const labels = { retrieving: "Đang tìm nguồn liên quan…", reranking: "Đang chọn bằng chứng phù hợp nhất…", generating: "Đang tổng hợp câu trả lời…" };
  return <section className="state-card loading" aria-live="polite"><p className="kicker">{labels[phase]}</p><div className="skeleton title" /><div className="skeleton" /><div className="skeleton short" /><div className="skeleton" /></section>;
}

export function NoResultState() {
  return <section className="state-card"><p className="kicker">Chưa đủ bằng chứng</p><h2>Không tìm thấy nguồn phù hợp.</h2><p>Thử diễn đạt cụ thể hơn, dùng tên sự kiện, nhân vật hoặc mốc thời gian.</p></section>;
}

export function ErrorState({ error, onRetry }) {
  return <section className="state-card error-card" role="alert"><p className="kicker">Không thể hoàn tất truy vấn</p><h2>News Desk đang bận.</h2><p>Kiểm tra kết nối rồi thử lại câu hỏi này.</p><button className="retry-button" onClick={onRetry}>Thử lại →</button>{import.meta.env.DEV && <details><summary>Chi tiết kỹ thuật</summary><code>{error}</code></details>}</section>;
}
