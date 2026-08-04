import { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { askNewsDesk } from "./api";
import { QuestionForm } from "./components/QuestionForm";
import { AnswerSection } from "./components/AnswerSection";
import { SourceCard } from "./components/SourceCard";
import { EmptyState, ErrorState, LoadingSkeleton, NoResultState } from "./components/States";
import "./styles.css";

function App() {
  const [question, setQuestion] = useState("");
  const [topK, setTopK] = useState(5);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [phase, setPhase] = useState(null);
  const loading = Boolean(phase);

  useEffect(() => {
    if (!loading) return undefined;
    const rerank = setTimeout(() => setPhase("reranking"), 550);
    const generate = setTimeout(() => setPhase("generating"), 1250);
    return () => { clearTimeout(rerank); clearTimeout(generate); };
  }, [loading]);

  async function submit(event) {
    event?.preventDefault();
    if (!question.trim() || loading) return;
    setResult(null); setError(""); setPhase("retrieving");
    try { setResult(await askNewsDesk(question.trim(), topK)); }
    catch (caught) { setError(caught instanceof Error ? caught.message : "Không thể kết nối dịch vụ."); }
    finally { setPhase(null); }
  }

  const contexts = result?.contexts || result?.retrieval || [];
  const noResult = result && (!contexts.length || result.answer_status === "abstained");
  return <div className="page">
    <header className="masthead"><a className="brand" href="#top"><i />News Desk</a><p>Hỏi đáp tin tức Việt Nam</p></header>
    <main id="top">
      <section className="hero"><div><p className="kicker">Tin tức, có nguồn</p><h1>HỎI GÌ,<br /><span> CŨNG RÕ.</span></h1><p className="hero-copy">Truy xuất bài viết liên quan, đối chiếu bằng chứng, rồi trả lời để bạn kiểm tra.</p></div><div className="hero-note"><strong>Đọc nhanh.</strong><span>Kiểm tra nguồn.</span><span>Hiểu đúng bối cảnh.</span></div></section>
      <QuestionForm {...{ question, setQuestion, topK, setTopK, loading, phase }} onSubmit={submit} />
      <section className="results" aria-live="polite">
        {loading && <LoadingSkeleton phase={phase} />}
        {error && <ErrorState error={error} onRetry={submit} />}
        {!loading && !error && !result && <EmptyState />}
        {noResult && <NoResultState />}
        {result && !noResult && <><div className="result-layout"><AnswerSection answer={result.answer || "Chưa có nội dung trả lời."} contexts={contexts} /><aside className="result-facts"><span>Độ tin cậy</span><b>{Number(result.confidence_percent ?? (result.confidence || 0) * 100).toFixed(0)}%</b><span>Thời gian phản hồi</span><b>{result.response_time_ms ? `${(result.response_time_ms / 1000).toFixed(1)}s` : "—"}</b><span>Nguồn dùng</span><b>{contexts.length}</b></aside></div><section className="sources"><div><p className="kicker">Nguồn tham khảo</p><h2>Đọc lại bằng chứng.</h2></div><div className="source-grid">{contexts.map((source, index) => <SourceCard key={source.chunk_id || index} source={source} index={index + 1} />)}</div></section></>}
      </section>
      <section className="how-it-works"><p className="kicker">Cách hoạt động</p><div><article><b>01</b><h3>Hybrid retrieval</h3><p>BM25 và dense retrieval cùng tìm nguồn có khả năng liên quan.</p></article><article><b>02</b><h3>BGE reranker</h3><p>Đối chiếu câu hỏi với từng context, chọn bằng chứng tốt nhất.</p></article><article><b>03</b><h3>Grounded answer</h3><p>Câu trả lời được tổng hợp từ context đã chọn, kèm citation.</p></article></div></section>
    </main>
    <footer>Vietnamese News QA · Evidence-first answers</footer>
  </div>;
}

createRoot(document.getElementById("root")).render(<App />);
