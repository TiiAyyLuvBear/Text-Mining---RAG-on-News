import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

export function AnswerSection({ answer }) {
  return <article className="answer-section"><p className="kicker">Câu trả lời</p><ReactMarkdown remarkPlugins={[remarkGfm]} components={{
    a: ({ href, children }) => <a href={href} target="_blank" rel="noreferrer">{children}</a>
  }}>{answer}</ReactMarkdown></article>;
}
