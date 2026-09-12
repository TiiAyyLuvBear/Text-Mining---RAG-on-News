export const HISTORY_LIMIT = 20;

function stringValue(value, fallback = "") {
  return typeof value === "string" && value.trim() ? value : fallback;
}

export function compactResult(result = {}) {
  return {
    answer: stringValue(result.answer, "Chưa có câu trả lời."),
    answer_status: stringValue(result.answer_status),
    evidence_sufficient: Boolean(result.evidence_sufficient),
    evidence_status: stringValue(result.evidence_status),
    response_time_ms: Number.isFinite(result.response_time_ms) ? result.response_time_ms : null,
    route_decision: result.route_decision && typeof result.route_decision === "object"
      ? {
          route: stringValue(result.route_decision.route),
          reason: stringValue(result.route_decision.reason),
        }
      : {},
  };
}

export function compactHistory(history) {
  if (!Array.isArray(history)) return [];
  return history.slice(0, HISTORY_LIMIT).flatMap((item) => {
    if (!item || typeof item !== "object" || !stringValue(item.question)) return [];
    const result = compactResult(item.result || { answer: item.answer });
    return [{
      id: stringValue(item.id, `${item.question}-${item.createdAt || ""}`),
      question: item.question,
      answer: result.answer,
      result,
      createdAt: stringValue(item.createdAt, new Date(0).toISOString()),
    }];
  });
}

export function saveHistory(storage, key, history) {
  try {
    storage.setItem(key, JSON.stringify(compactHistory(history)));
    return true;
  } catch {
    return false;
  }
}
