export function getRefusalMessage(result) {
  if (result?.decision !== "REFUSE") return "";
  const category = String(result?.failure_category || "").toUpperCase();
  const code = String(result?.refusal_reason_code || "").toUpperCase();
  if (category === "GENERATOR" || code === "GENERATOR_ERROR" || code === "GENERATOR_EMPTY") {
    return "Không thể tạo câu trả lời do lỗi bộ sinh.";
  }
  if (category === "VERIFICATION" || code === "VERIFICATION_FAILED") {
    return "Câu trả lời không vượt qua bước kiểm chứng bằng chứng.";
  }
  if (category === "EVIDENCE" || code === "INSUFFICIENT_EVIDENCE" || code === "RETRY_EXHAUSTED") {
    return "Chưa đủ bằng chứng để trả lời.";
  }
  return "Không thể tạo câu trả lời cho yêu cầu này.";
}

export function getEvidenceStatusLabel(result) {
  const category = String(result?.failure_category || "").toUpperCase();
  if (category === "GENERATOR") return "Lỗi bộ sinh";
  if (category === "VERIFICATION") return "Không đạt kiểm chứng";
  return result?.evidence_sufficient ? "Đủ bằng chứng" : "Chưa đủ bằng chứng";
}

export function getResultPresentation(result) {
  const contexts = Array.isArray(result?.contexts) && result.contexts.length > 0
    ? result.contexts
    : Array.isArray(result?.retrieval)
      ? result.retrieval
      : [];
  const rawAnswer = typeof result?.answer === "string" ? result.answer.trim() : "";
  const answer = rawAnswer || getRefusalMessage(result);
  const isRefused = result?.decision === "REFUSE";

  return {
    answer,
    contexts,
    evidenceStatusLabel: getEvidenceStatusLabel(result),
    isAbstained: isRefused || result?.answer_status === "abstained",
    showEmptyState: Boolean(result) && !answer && contexts.length === 0,
  };
}
