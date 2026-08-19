import { AnswerSection } from "./AnswerSection";
import { SourceCard } from "./SourceCard";
import { groupSources } from "../sourcePresentation";

export function ResultTabs({ activeTab, onTabChange, answer, contexts, topK, result }) {
  const shownContexts = contexts.slice(0, topK);
  const groupedSources = groupSources(shownContexts);
  return <>
    <div className="result-tabs" role="tablist" aria-label="Kết quả truy vấn">
      <button role="tab" aria-selected={activeTab === "answer"} className={activeTab === "answer" ? "active" : ""} onClick={() => onTabChange("answer")}>Câu trả lời</button>
      <button role="tab" aria-selected={activeTab === "evidence"} className={activeTab === "evidence" ? "active" : ""} onClick={() => onTabChange("evidence")}>Bằng chứng ({groupedSources.length} bài)</button>
    </div>
    {activeTab === "answer" ? <div className="result-layout"><aside className="result-facts"><div><span>Trạng thái bằng chứng</span><b>{result.evidence_sufficient ? "Đủ bằng chứng" : "Chưa đủ bằng chứng"}</b></div><div><span>Thời gian phản hồi</span><b>{result.response_time_ms ? `${(result.response_time_ms / 1000).toFixed(1)}s` : "—"}</b></div><div><span>Nguồn dùng</span><b>{groupedSources.length} bài báo</b><small>{shownContexts.length} đoạn bằng chứng</small></div></aside><AnswerSection answer={answer || "Chưa có nội dung trả lời."} /></div> : <section className="sources evidence-tab"><div><p className="kicker">Nguồn tham khảo</p><h2>Bằng chứng đã chọn.</h2><p className="evidence-intro">{groupedSources.length} bài báo · {shownContexts.length} đoạn bằng chứng.</p></div><div className="source-grid">{groupedSources.map((source, index) => <SourceCard key={source.key} source={source} index={index + 1} />)}</div></section>}
  </>;
}
