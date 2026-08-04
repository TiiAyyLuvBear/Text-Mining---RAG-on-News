import { StatusSteps } from "./StatusSteps";

const examples = [
  "Tại sao nước dùng hầm xương có thể gây hại cho thận?",
  "Loại nội tạng nào nên hạn chế để tránh tăng axit uric?",
  "Nhà tù Hỏa Lò có ý nghĩa gì đối với du lịch Việt Nam?"
];

export function QuestionForm({ question, setQuestion, topK, setTopK, onSubmit, loading, phase }) {
  return <section className="question-panel" aria-labelledby="ask-title"><div><p className="kicker">Đặt câu hỏi</p><h2 id="ask-title">Bạn muốn biết tìm hiểu điều gì hôm nay?</h2></div>
    <form onSubmit={onSubmit}><label className="sr-only" htmlFor="question">Câu hỏi</label><textarea id="question" value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="Đặt câu hỏi bằng tiếng Việt…" rows="4" disabled={loading} />
      <div className="form-actions"><label>Số nguồn <select value={topK} onChange={(event) => setTopK(Number(event.target.value))} disabled={loading}><option value="3">3</option><option value="5">5</option><option value="8">8</option></select></label><button className="ask-button" disabled={loading}>{loading ? "Đang xử lý…" : "Tìm câu trả lời"}</button></div>
    </form><div className="examples">{examples.map((item) => <button type="button" key={item} onClick={() => setQuestion(item)} disabled={loading}>{item}</button>)}</div>{loading && <StatusSteps phase={phase} />}
  </section>;
}
