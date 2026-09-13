import test from "node:test";
import assert from "node:assert/strict";

import {
  getEvidenceStatusLabel,
  getResultPresentation,
} from "../src/resultPresentation.js";


test("generator errors are not displayed as insufficient evidence", () => {
  const result = {
    decision: "REFUSE",
    answer: "",
    evidence_sufficient: true,
    failure_category: "GENERATOR",
    refusal_reason_code: "GENERATOR_ERROR",
  };

  assert.equal(getResultPresentation(result).answer, "Không thể tạo câu trả lời do lỗi bộ sinh.");
  assert.equal(getEvidenceStatusLabel(result), "Lỗi bộ sinh");
});


test("evidence and verification refusals have distinct messages", () => {
  const evidence = getResultPresentation({
    decision: "REFUSE", failure_category: "EVIDENCE", refusal_reason_code: "INSUFFICIENT_EVIDENCE",
  });
  const verification = getResultPresentation({
    decision: "REFUSE", failure_category: "VERIFICATION", refusal_reason_code: "VERIFICATION_FAILED",
  });

  assert.equal(evidence.answer, "Chưa đủ bằng chứng để trả lời.");
  assert.equal(verification.answer, "Câu trả lời không vượt qua bước kiểm chứng bằng chứng.");
  assert.equal(getEvidenceStatusLabel({ failure_category: "VERIFICATION" }), "Không đạt kiểm chứng");
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
