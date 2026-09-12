export function getResultPresentation(result) {
  const contexts = Array.isArray(result?.contexts)
    ? result.contexts
    : Array.isArray(result?.retrieval)
      ? result.retrieval
      : [];
  const answer = typeof result?.answer === "string" ? result.answer.trim() : "";

  return {
    answer,
    contexts,
    isAbstained: result?.answer_status === "abstained",
    showEmptyState: Boolean(result) && !answer && contexts.length === 0,
  };
}
