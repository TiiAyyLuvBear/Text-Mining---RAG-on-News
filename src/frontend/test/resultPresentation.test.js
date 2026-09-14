import test from "node:test";
import assert from "node:assert/strict";

import {
  getEvidenceStatusLabel,
  getResultPresentation,
  stripCitationMarkers,
} from "../src/resultPresentation.js";


test("generator errors are not displayed as insufficient evidence", () => {
  const result = {
    decision: "REFUSE",
    answer: "",
    evidence_sufficient: true,
    failure_category: "GENERATOR",
    refusal_reason_code: "GENERATOR_ERROR",
  };

  assert.match(getResultPresentation(result).answer, /mô hình tạo câu trả lời hiện không phản hồi ổn định/);
  assert.equal(getEvidenceStatusLabel(result), "Lỗi bộ sinh");
});


test("evidence and verification refusals have distinct messages", () => {
  const evidence = getResultPresentation({
    decision: "REFUSE", failure_category: "EVIDENCE", refusal_reason_code: "INSUFFICIENT_EVIDENCE",
  });
  const verification = getResultPresentation({
    decision: "REFUSE", failure_category: "VERIFICATION", refusal_reason_code: "VERIFICATION_FAILED",
  });

  assert.match(evidence.answer, /chưa đủ bằng chứng/i);
  assert.match(evidence.answer, /dừng lại thay vì suy đoán/i);
  assert.match(verification.answer, /chưa được bằng chứng xác nhận đầy đủ/i);
  assert.equal(getEvidenceStatusLabel({ failure_category: "VERIFICATION" }), "Không đạt kiểm chứng");
});


test("answer citations are hidden only in frontend presentation", () => {
  const raw = "Dư luận bất ngờ vì doanh nghiệp mới thành lập [Nguồn 1] [Nguồn 2].";
  const grouped = "Mức thuê gần 60 tỷ đồng [Nguồn 1, Nguồn 2].";

  assert.equal(
    stripCitationMarkers(`${raw}\n\n${grouped}`),
    "Dư luận bất ngờ vì doanh nghiệp mới thành lập.\n\nMức thuê gần 60 tỷ đồng.",
  );
  assert.equal(getResultPresentation({ decision: "ANSWER", answer: raw }).answer.includes("[Nguồn"), false);
});


test("insufficient evidence explains retry and missing subquestion", () => {
  const presentation = getResultPresentation({
    decision: "REFUSE",
    failure_category: "EVIDENCE",
    refusal_reason_code: "RETRY_EXHAUSTED",
    retry_count: 1,
    missing_evidence: ["sq2"],
    evidence_plan: {
      sub_questions: [{ id: "sq2", text: "Danh tính doanh nghiệp trúng đấu giá" }],
    },
  });

  assert.match(presentation.answer, /Sau một lần tìm kiếm bổ sung/);
  assert.match(presentation.answer, /Danh tính doanh nghiệp trúng đấu giá/);
});


test("insufficient evidence falls back to the retrieved source pool", () => {
  const presentation = getResultPresentation({
    decision: "REFUSE",
    failure_category: "EVIDENCE",
    contexts: [],
    retrieval: [{ article_id: "a", text: "candidate" }],
  });
  assert.equal(presentation.contexts.length, 1);
  assert.equal(presentation.contexts[0].article_id, "a");
});
