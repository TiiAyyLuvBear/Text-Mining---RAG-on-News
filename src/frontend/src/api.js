const baseUrl = (import.meta.env.VITE_API_BASE_URL || "").replace(/\/$/, "");

class NewsDeskError extends Error {
  constructor(message, code = "CONNECTION_ERROR") {
    super(message);
    this.name = "NewsDeskError";
    this.code = code;
  }
}

export async function askNewsDesk(question, topK) {
  let response;
  try {
    response = await fetch(`${baseUrl}/api/qa/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, top_k: topK })
    });
  } catch {
    throw new NewsDeskError("Không thể kết nối đến backend. Hãy kiểm tra dịch vụ FastAPI.");
  }
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const message = data.error?.message || data.detail || "Dịch vụ tạm thời không phản hồi.";
    throw new NewsDeskError(message, data.error?.code || "API_ERROR");
  }
  return data;
}
