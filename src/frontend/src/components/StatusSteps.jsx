const steps = ["Truy xuất nguồn", "Xếp hạng bằng chứng", "Tổng hợp câu trả lời"];

export function StatusSteps() {
  return <div className="loading-status" role="status">
    <p><span aria-hidden="true" />Đang truy xuất và đối chiếu nguồn…</p>
    <ol className="status-steps" aria-label="Các bước xử lý">
    {steps.map((label, index) => <li key={label}>
      <span>{String(index + 1).padStart(2, "0")}</span>{label}
    </li>)}
    </ol>
  </div>;
}
