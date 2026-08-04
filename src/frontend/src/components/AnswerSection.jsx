import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

function Citation({ number }) { return <a className="citation" href={`#source-${number}`} aria-label={`Xem nguồn ${number}`}>[{number}]</a>; }

export function AnswerSection({ answer, contexts }) {
  let paragraphIndex = 0;
  return <article className="answer-section"><p className="kicker">Câu trả lời</p><ReactMarkdown remarkPlugins={[remarkGfm]} components={{
    p: ({ children }) => { const index = paragraphIndex++; return <p>{children} {contexts.length > 0 && <Citation number={(index % contexts.length) + 1} />}</p>; },
    a: ({ href, children }) => <a href={href} target="_blank" rel="noreferrer">{children}</a>
  }}>{answer}</ReactMarkdown></article>;
}
