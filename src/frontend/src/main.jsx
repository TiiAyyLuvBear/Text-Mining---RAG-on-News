import { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { askNewsDesk } from "./api";
import { QuestionForm } from "./components/QuestionForm";
import { AnswerSection } from "./components/AnswerSection";
import { SourceCard } from "./components/SourceCard";
import { EmptyState, ErrorState, LoadingSkeleton, NoResultState } from "./components/States";
import { groupSources } from "./sourcePresentation";
import { HistorySidebar } from "./components/HistorySidebar";

const HISTORY_STORAGE_KEY = "news-desk-history";

function readHistory() {
  try {
    const stored = JSON.parse(localStorage.getItem(HISTORY_STORAGE_KEY) || "[]");
    return Array.isArray(stored) ? stored : [];
  } catch { return []; }
}
import "./styles.css";

function App() {
  const [question, setQuestion] = useState("");
  const [topK, setTopK] = useState(5);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState(readHistory);

  useEffect(() => {
    localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(history));
  }, [history]);

  async function submit(event) {
    event?.preventDefault();
    if (!question.trim() || loading) return;
    setResult(null); setError(null); setLoading(true);
    try {
      const askedQuestion = question.trim();
      const nextResult = await askNewsDesk(askedQuestion, topK);
      setResult(nextResult);
      setHistory((current) => [{
        id: `1787239300822-${Math.random().toString(36).slice(2)}`,
        question: askedQuestion,
        answer: nextResult.answer || "Chưa có câu trả lời.",
        result: nextResult,
        createdAt: new Date().toISOString()
      }, ...current]);
    }
    catch (caught) { setError(caught instanceof Error ? caught : new Error("Không thể kết nối dịch vụ.")); }
    finally { setLoading(false); }
  }

  const contexts = result?.contexts || result?.retrieval || [];
  const groupedSources = groupSources(contexts);
  const noResult = result && (!contexts.length || result.answer_status === "abstained");
  function selectHistory(item) {
    setQuestion(item.question);
    setResult(item.result);
    setError("");
  }
  return <div className="page-shell"><HistorySidebar history={history} onSelect={selectHistory} onClear={() => setHistory([])} /><div className="page">
    <header className="masthead"><a className="brand" href="#top"><i />News Desk</a><p>Hỏi đáp tin tức Việt Nam</p></header>
    <main id="top">
      <section className="hero"><div className="hero-main"><p className="kicker">Tin tức, có nguồn</p><h1>HỎI GÌ,<br /><span>CŨNG RÕ.</span></h1><p className="hero-copy">Truy xuất bài viết liên quan, đối chiếu bằng chứng, rồi trả lời để bạn kiểm tra.</p></div><aside className="hero-note"><strong>Đọc nhanh.</strong><span>Kiểm tra nguồn.</span><span>Hiểu đúng bối cảnh.</span></aside></section>
      <QuestionForm {...{ question, setQuestion, topK, setTopK, loading }} onSubmit={submit} />
      <section className="results" aria-live="polite">
        {loading && <LoadingSkeleton />}
        {error && <ErrorState error={error} onRetry={submit} />}
        {!loading && !error && !result && <EmptyState />}
        {noResult && <NoResultState />}
        {result && !noResult && <><div className="result-layout"><aside className="result-facts"><div><span>Trạng thái bằng chứng</span><b>{result.evidence_sufficient ? "Đủ bằng chứng" : "Chưa đủ bằng chứng"}</b></div><div><span>Thời gian phản hồi</span><b>{result.response_time_ms ? `${(result.response_time_ms / 1000).toFixed(1)}s` : "—"}</b></div><div><span>Nguồn đã dùng</span><b>{groupedSources.length} bài báo</b><small>{contexts.length} đoạn bằng chứng</small></div></aside><AnswerSection answer={result.answer || "Chưa có nội dung trả lời."} /></div><section className="sources"><div><p className="kicker">Nguồn tham khảo</p><h2>Các bài báo hỗ trợ câu trả lời.</h2><p className="source-summary">{groupedSources.length} bài báo · {contexts.length} đoạn bằng chứng</p></div><div className="source-grid">{groupedSources.map((source, index) => <SourceCard key={source.key} source={source} index={index + 1} />)}</div></section></>}
      </section>
      <section className="how-it-works"><p className="kicker">Cách hoạt động</p><div><article><b>01</b><h3>Dense retrieval</h3><p>E5 và Qdrant tìm các đoạn tin có khả năng liên quan.</p></article><article><b>02</b><h3>BGE reranker</h3><p>Đối chiếu câu hỏi với từng context, chọn bằng chứng tốt nhất.</p></article><article><b>03</b><h3>Grounded answer</h3><p>Câu trả lời được tổng hợp từ context đã chọn và hiển thị cùng nguồn hỗ trợ.</p></article></div></section>
    </main>
    <footer>Vietnamese News QA · Evidence-first answers</footer>
  </div></div>;
}

createRoot(document.getElementById("root")).render(<App />);
