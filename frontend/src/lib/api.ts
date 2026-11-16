const API_BASE = "/api";
const QUERY_TIMEOUT = 120_000; // 流式输出可能较长，给 120s
const DEFAULT_TIMEOUT = 10_000;

export interface QueryResponse {
  answer: string;
  sources: string[];
  conversation_id: string;
  citations?: Citation[];
}

/** 一条可展开的引用片段 */
export interface Citation {
  filename: string;
  text: string;
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

export interface StreamCallbacks {
  onToken: (token: string) => void;
  onDone: (conversationId: string, sources: string[], citations?: Citation[]) => void;
  onError: (error: string) => void;
  /** 阶段提示：retrieving（检索中）/ generating（生成中） */
  onStage?: (stage: string) => void;
  /** 用户主动取消（区别于超时/网络错误） */
  onAbort?: () => void;
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
    180_000 // 解析 + embedding 可能较慢
  );
}

/** 非流式查询（兼容旧接口） */
export async function query(
  question: string,
  conversationId?: string
): Promise<QueryResponse> {
  return request<QueryResponse>(
    `${API_BASE}/query`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        conversation_id: conversationId,
      }),
    },
    QUERY_TIMEOUT
  );
}

/** SSE 流式查询。传入 signal 可让用户主动停止生成。 */
export async function queryStream(
  question: string,
  conversationId: string | undefined,
  callbacks: StreamCallbacks,
  signal?: AbortSignal
): Promise<void> {
  const controller = new AbortController();
  let abortedByUser = false;

  // 用户点"停止"时，中断内部请求并标记为主动取消（超时用另一条路径）
  const onUserAbort = () => {
    abortedByUser = true;
    controller.abort();
  };
  if (signal?.aborted) {
    onUserAbort();
  } else {
    signal?.addEventListener("abort", onUserAbort, { once: true });
  }

  // 初始连接超时 30s
  let timer: ReturnType<typeof setTimeout> | null = setTimeout(
    () => controller.abort(),
    30_000
  );

  try {
    const res = await fetch(`${API_BASE}/query/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        conversation_id: conversationId,
      }),
      signal: controller.signal,
    });

    // 连接成功，切换到流式超时（每收到一个 chunk 重置）
    if (timer) clearTimeout(timer);
    timer = setTimeout(() => controller.abort(), 60_000);

    if (!res.ok) {
      if (timer) clearTimeout(timer);
      const body = await res.json().catch(() => ({}));
      callbacks.onError(body.detail || `请求失败 (${res.status})`);
      return;
    }

    const reader = res.body?.getReader();
    if (!reader) {
      if (timer) clearTimeout(timer);
      callbacks.onError("无法读取响应流");
      return;
    }

    const decoder = new TextDecoder();
    let buffer = "";

    let receivedDone = false;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      // 每收到数据就重置超时计时器
      if (timer) clearTimeout(timer);
      timer = setTimeout(() => controller.abort(), 60_000);

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || ""; // 保留未完成的行

      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        const jsonStr = line.slice(6).trim();
        if (!jsonStr) continue;

        try {
          const event = JSON.parse(jsonStr);

          if (event.type === "token") {
            callbacks.onToken(event.content);
          } else if (event.type === "answer") {
            // 后端在"没有文档 / 没有命中"时直接返回整段答案，同样按内容追加
            callbacks.onToken(event.content);
          } else if (event.type === "stage") {
            callbacks.onStage?.(event.content);
          } else if (event.type === "done") {
            receivedDone = true;
            callbacks.onDone(event.conversation_id, event.sources || [], event.citations || []);
          } else if (event.type === "error") {
            receivedDone = true;
            callbacks.onError(event.content);
          }
        } catch {
          // 忽略解析失败的行
        }
      }
    }

    // 流结束但没有收到 done/error 事件（网络中断、服务端崩溃等）
    if (!receivedDone && !abortedByUser) {
      callbacks.onError("连接中断，请重试");
    }
  } catch (err) {
    if (abortedByUser) {
      callbacks.onAbort?.();
    } else if (err instanceof DOMException && err.name === "AbortError") {
      callbacks.onError("请求超时，请稍后重试");
    } else {
      callbacks.onError(err instanceof Error ? err.message : "未知错误");
    }
  } finally {
    if (timer) clearTimeout(timer);
    signal?.removeEventListener("abort", onUserAbort);
  }
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

export async function createConversation(): Promise<string> {
  const res = await request<{ conversation_id: string }>(
    `${API_BASE}/conversations`,
    { method: "POST" }
  );
  return res.conversation_id;
}

export async function deleteConversation(conversationId: string): Promise<void> {
  await request<Record<string, never>>(
    `${API_BASE}/conversations/${encodeURIComponent(conversationId)}`,
    { method: "DELETE" }
  );
}
