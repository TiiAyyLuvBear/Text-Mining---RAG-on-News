import { AnswerSection } from "./AnswerSection";
import { SourceCard } from "./SourceCard";

export function ResultTabs({ activeTab, onTabChange, answer, contexts, topK, result }) {
  const shownContexts = contexts.slice(0, topK);
  return <>
    <div className="result-tabs" role="tablist" aria-label="Kết quả truy vấn">
      <button role="tab" aria-selected={activeTab === "answer"} className={activeTab === "answer" ? "active" : ""} onClick={() => onTabChange("answer")}>Câu trả lời</button>
      <button role="tab" aria-selected={activeTab === "evidence"} className={activeTab === "evidence" ? "active" : ""} onClick={() => onTabChange("evidence")}>Bằng chứng ({shownContexts.length})</button>
    </div>
    {activeTab === "answer" ? <div className="result-layout"><AnswerSection answer={answer || "Chưa có nội dung trả lời."} contexts={contexts} /><aside className="result-facts"><span>Độ tin cậy</span><b>{Number(result.confidence_percent ?? (result.confidence || 0) * 100).toFixed(0)}%</b><span>Thời gian phản hồi</span><b>{result.response_time_ms ? `${(result.response_time_ms / 1000).toFixed(1)}s` : "—"}</b><span>Nguồn dùng</span><b>{contexts.length}</b></aside></div> : <section className="sources evidence-tab"><div><p className="kicker">Top {shownContexts.length} context</p><h2>Bằng chứng đã chọn.</h2><p className="evidence-intro">Mỗi thẻ hiển thị bài viết theo <code>article_id</code>. Mở thẻ để xem context tương ứng.</p></div><div className="source-grid">{shownContexts.map((source, index) => <SourceCard key={source.chunk_id || index} source={source} index={index + 1} />)}</div></section>}
  </>;
}
