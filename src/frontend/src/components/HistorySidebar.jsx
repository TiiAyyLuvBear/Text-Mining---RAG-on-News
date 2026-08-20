export function HistorySidebar({ history, onSelect, onClear }) {
  return <aside className="history-sidebar" aria-label="Lịch sử hỏi đáp">
    <div className="history-header">
      <div><p className="kicker">Lịch sử</p><h2>Câu hỏi cũ</h2></div>
      {history.length > 0 && <button type="button" className="history-clear" onClick={onClear}>Xóa</button>}
    </div>
    {history.length === 0 ? <p className="history-empty">Các câu hỏi và câu trả lời sẽ được lưu tại đây.</p> : <ul className="history-list">
      {history.map((item) => <li key={item.id}>
        <button type="button" className="history-item" onClick={() => onSelect(item)}>
          <strong>{item.question}</strong><span>{item.answer}</span>
          <time dateTime={item.createdAt}>{new Date(item.createdAt).toLocaleString("vi-VN")}</time>
        </button>
      </li>)}
    </ul>}
  </aside>;
}
