export function stripCitationMarkers(answer) {
  if (typeof answer !== "string") return "";
  return answer
    .replace(/\s*\[(?:Nguồn\s*)?\d+(?:\s*,\s*(?:Nguồn\s*)?\d+)*\]/gi, "")
    .replace(/[ \t]+([,.;:!?])/g, "$1")
    .replace(/[ \t]{2,}/g, " ")
    .replace(/\n[ \t]+/g, "\n")
    .trim();
}

function missingEvidenceDescriptions(result) {
  const missing = Array.isArray(result?.missing_evidence) ? result.missing_evidence : [];
  const subQuestions = Array.isArray(result?.evidence_plan?.sub_questions)
    ? result.evidence_plan.sub_questions
    : [];
  const descriptions = new Map(
    subQuestions.map((item) => [String(item?.id || ""), String(item?.text || "").trim()]),
  );
  return missing
    .map((item) => descriptions.get(String(item)) || String(item || "").trim())
    .filter((item) => item && !/^sq\d+$/i.test(item));
}

export function getRefusalMessage(result) {
  if (result?.decision !== "REFUSE") return "";
  const category = String(result?.failure_category || "").toUpperCase();
  const code = String(result?.refusal_reason_code || "").toUpperCase();
  if (category === "GENERATOR" || code === "GENERATOR_ERROR" || code === "GENERATOR_EMPTY") {
    return "Hệ thống đã tìm thấy nguồn liên quan, nhưng mô hình tạo câu trả lời hiện không phản hồi ổn định. Đây không có nghĩa là dữ liệu bị thiếu; vui lòng thử lại sau ít phút.";
  }
  if (category === "VERIFICATION" || code === "VERIFICATION_FAILED") {
    return "Hệ thống đã tìm thấy nguồn liên quan, nhưng bản trả lời tạo ra có chi tiết hoặc trích dẫn chưa được bằng chứng xác nhận đầy đủ. Nội dung đó không được hiển thị để tránh gây hiểu nhầm; bạn có thể thử lại.";
  }
  if (category === "EVIDENCE" || code === "INSUFFICIENT_EVIDENCE" || code === "RETRY_EXHAUSTED") {
    const descriptions = missingEvidenceDescriptions(result);
    const retryNote = Number(result?.retry_count || 0) > 0
      ? " Sau một lần tìm kiếm bổ sung, các nguồn vẫn chưa bao phủ đủ thông tin cần thiết."
      : " Các nguồn truy xuất chưa bao phủ đủ thông tin cần thiết.";
    const missingNote = descriptions.length > 0
      ? ` Phần chưa thể xác minh: ${descriptions.map((item) => `“${item}”`).join("; ")}.`
      : "";
    return `Hiện chưa đủ bằng chứng để trả lời câu hỏi này một cách đáng tin cậy.${retryNote}${missingNote} Hệ thống dừng lại thay vì suy đoán; hãy thử bổ sung tên sự kiện, nhân vật hoặc mốc thời gian cụ thể hơn.`;
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
  const rawAnswer = stripCitationMarkers(result?.answer);
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
