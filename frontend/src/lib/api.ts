const API_BASE = "/api";

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

export async function uploadDocument(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API_BASE}/upload`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "上传失败" }));
    throw new Error(err.detail || "上传失败");
  }

  return res.json();
}

export async function query(question: string): Promise<QueryResponse> {
  const res = await fetch(`${API_BASE}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "请求失败" }));
    throw new Error(err.detail || "请求失败");
  }

  return res.json();
}

export async function getStats(): Promise<StatsResponse> {
  const res = await fetch(`${API_BASE}/stats`);

  if (!res.ok) {
    throw new Error("获取统计信息失败");
  }

  return res.json();
}

export async function deleteDocument(filename: string): Promise<void> {
  const res = await fetch(`${API_BASE}/documents/${encodeURIComponent(filename)}`, {
    method: "DELETE",
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "删除失败" }));
    throw new Error(err.detail || "删除失败");
  }
}
