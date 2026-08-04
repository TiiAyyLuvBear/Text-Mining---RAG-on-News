const steps = [
  ["retrieving", "Đang tìm nguồn"],
  ["reranking", "Đang đối chiếu"],
  ["generating", "Đang viết câu trả lời"]
];

export function StatusSteps({ phase }) {
  const activeIndex = steps.findIndex(([key]) => key === phase);
  return <ol className="status-steps" aria-live="polite">
    {steps.map(([key, label], index) => <li key={key} className={index <= activeIndex ? "active" : ""}>
      <span>{index + 1}</span>{label}
    </li>)}
  </ol>;
}
