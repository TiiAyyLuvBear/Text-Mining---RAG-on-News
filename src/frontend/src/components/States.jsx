export function EmptyState() {
  return <section className="state-card"><p className="kicker">News Desk</p><h2>Bắt đầu từ một câu hỏi.</h2><p>Chọn gợi ý hoặc viết câu hỏi tự nhiên. Nguồn dùng để trả lời sẽ xuất hiện tại đây.</p></section>;
}

export function LoadingSkeleton() {
  return <section className="state-card loading" role="status"><p className="kicker">Đang tra cứu</p><h2>Đang truy xuất và đối chiếu nguồn…</h2><p>News Desk đang tìm các đoạn tin liên quan trước khi tổng hợp câu trả lời.</p><div className="loading-rule" aria-hidden="true"><span /></div></section>;
}

export function NoResultState() {
  return <section className="state-card"><p className="kicker">Chưa đủ bằng chứng</p><h2>Chưa thể trả lời đáng tin cậy.</h2><p>Các nguồn được truy xuất hiện chưa đủ để đưa ra câu trả lời đáng tin cậy. Thử dùng tên sự kiện, nhân vật hoặc mốc thời gian cụ thể hơn.</p></section>;
}

export function ErrorState({ error, onRetry }) {
  const llmError = error?.code === "LLM_UNAVAILABLE";
  return <section className="state-card error-card" role="alert"><p className="kicker">{llmError ? "Không thể tạo câu trả lời" : "Không thể kết nối"}</p><h2>{llmError ? "Chưa thể tổng hợp câu trả lời." : "Không thể kết nối tới hệ thống hỏi đáp."}</h2><p>{llmError ? "Nguồn đã được truy xuất nhưng hệ thống chưa thể tổng hợp câu trả lời lúc này." : "Vui lòng kiểm tra dịch vụ và thử lại."}</p><button className="retry-button" onClick={onRetry}>Thử lại</button></section>;
}
