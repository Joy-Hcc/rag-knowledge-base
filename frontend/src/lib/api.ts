const API_BASE = "/api";
const QUERY_TIMEOUT = 60_000; // LLM 推理较慢，给 60s
const DEFAULT_TIMEOUT = 10_000;

export interface QueryResponse {
  answer: string;
  sources: string[];
}

export interface StatsResponse {
  document_count: number;
  total_chars: number;
  documents: string[];
  chunk_count: number;
}

export interface UploadResponse {
  message: string;
  filename: string;
  chars: number;
  chunks: number;
}

async function request<T>(
  url: string,
  options: RequestInit = {},
  timeout = DEFAULT_TIMEOUT
): Promise<T> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeout);

  try {
    const res = await fetch(url, {
      ...options,
      signal: controller.signal,
    });

    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || `请求失败 (${res.status})`);
    }

    return res.json();
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") {
      throw new Error("请求超时，请稍后重试");
    }
    throw err;
  } finally {
    clearTimeout(timer);
  }
}

export async function uploadDocument(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append("file", file);

  return request<UploadResponse>(
    `${API_BASE}/upload`,
    { method: "POST", body: formData },
    30_000 // 上传可能较慢
  );
}

export async function query(question: string): Promise<QueryResponse> {
  return request<QueryResponse>(
    `${API_BASE}/query`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    },
    QUERY_TIMEOUT
  );
}

export async function getStats(): Promise<StatsResponse> {
  return request<StatsResponse>(`${API_BASE}/stats`);
}

export async function deleteDocument(filename: string): Promise<void> {
  await request<Record<string, never>>(
    `${API_BASE}/documents/${encodeURIComponent(filename)}`,
    { method: "DELETE" }
  );
}
