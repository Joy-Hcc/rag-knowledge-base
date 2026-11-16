"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import Header from "@/components/Header";
import Sidebar from "@/components/Sidebar";
import ChatPanel from "@/components/ChatPanel";
import ChatInput from "@/components/ChatInput";
import type { Message } from "@/components/ChatMessage";
import type { StatsResponse } from "@/lib/api";
import * as api from "@/lib/api";

const STORAGE_KEY = "rag-kb:session";

// 用时间戳做起点，避免恢复历史消息后与新消息 id 撞车
let msgId = Date.now();
const nextId = () => String(++msgId);

interface StoredSession {
  messages: Message[];
  conversationId?: string;
}

function loadSession(): StoredSession {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return { messages: [] };
    const parsed = JSON.parse(raw) as StoredSession;
    const messages = Array.isArray(parsed.messages)
      ? parsed.messages.map((m) => ({ ...m, pending: false }))
      : [];
    return { messages, conversationId: parsed.conversationId };
  } catch {
    return { messages: [] };
  }
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [statsLoading, setStatsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [stage, setStage] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [hydrated, setHydrated] = useState(false);

  // 用 ref 存 conversationId，避免 state 异步更新导致竞态
  const conversationRef = useRef<string | undefined>(undefined);
  const abortRef = useRef<AbortController | null>(null);

  // 加载统计信息
  const refreshStats = useCallback(async () => {
    try {
      setStatsLoading(true);
      const data = await api.getStats();
      setStats(data);
    } catch (err) {
      if (process.env.NODE_ENV === "development") {
        console.warn("Stats fetch failed:", err);
      }
    } finally {
      setStatsLoading(false);
    }
  }, []);

  // 恢复上次会话（刷新不丢对话）
  useEffect(() => {
    const session = loadSession();
    if (session.messages.length > 0) {
      setMessages(session.messages);
      conversationRef.current = session.conversationId;
    }
    setHydrated(true);
    refreshStats();
  }, [refreshStats]);

  // 持久化会话
  useEffect(() => {
    if (!hydrated) return;
    try {
      if (messages.length === 0) {
        window.localStorage.removeItem(STORAGE_KEY);
      } else {
        window.localStorage.setItem(
          STORAGE_KEY,
          JSON.stringify({ messages, conversationId: conversationRef.current })
        );
      }
    } catch {
      // 隐私模式/配额满时忽略
    }
  }, [messages, hydrated]);

  // 新建对话
  const handleNewConversation = useCallback(() => {
    abortRef.current?.abort();
    conversationRef.current = undefined;
    setMessages([]);
    setStage(null);
    setSidebarOpen(false);
  }, []);

  // 上传文档
  const handleUpload = useCallback(
    async (file: File) => {
      try {
        setError(null);
        await api.uploadDocument(file);
        await refreshStats();
      } catch (err) {
        setError(err instanceof Error ? err.message : "上传失败");
      }
    },
    [refreshStats]
  );

  // 删除文档
  const handleDelete = useCallback(
    async (filename: string) => {
      try {
        setError(null);
        await api.deleteDocument(filename);
        await refreshStats();
      } catch (err) {
        setError(err instanceof Error ? err.message : "删除失败");
      }
    },
    [refreshStats]
  );

  // 停止生成
  const handleStop = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  // 发送问题（SSE 流式）
  const handleSend = useCallback(async (question: string) => {
    // 添加用户消息
    setMessages((prev) => [
      ...prev,
      { id: nextId(), role: "user", content: question },
    ]);

    // 添加一个空的 assistant 消息占位
    const assistantId = nextId();
    setMessages((prev) => [
      ...prev,
      { id: assistantId, role: "assistant", content: "", pending: true },
    ]);

    setIsStreaming(true);
    setStage("retrieving");
    setSidebarOpen(false);

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      await api.queryStream(
        question,
        conversationRef.current,
        {
          onStage: (s: string) => setStage(s),
          onToken: (token: string) => {
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === assistantId
                  ? { ...msg, content: msg.content + token }
                  : msg
              )
            );
          },
          onDone: (convId: string, sources: string[], citations?: api.Citation[]) => {
            // 立即更新 ref（同步，无竞态）
            conversationRef.current = convId;
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === assistantId
                  ? { ...msg, sources, citations, pending: false }
                  : msg
              )
            );
            setIsStreaming(false);
            setStage(null);
          },
          onAbort: () => {
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === assistantId
                  ? { ...msg, pending: false, stopped: true }
                  : msg
              )
            );
            setIsStreaming(false);
            setStage(null);
          },
          onError: (err: string) => {
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === assistantId
                  ? { ...msg, content: `请求失败: ${err}`, pending: false }
                  : msg
              )
            );
            setIsStreaming(false);
            setStage(null);
          },
        },
        controller.signal
      );
    } catch (err) {
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === assistantId
            ? { ...msg, content: `请求失败: ${err instanceof Error ? err.message : "未知错误"}`, pending: false }
            : msg
        )
      );
      setIsStreaming(false);
      setStage(null);
    } finally {
      abortRef.current = null;
    }
  }, []);

  return (
    <div className="min-h-screen flex flex-col bg-background text-foreground">
      <Header
        onNewConversation={handleNewConversation}
        hasConversation={messages.length > 0}
        onToggleSidebar={() => setSidebarOpen((v) => !v)}
      />
      {error && (
        <div className="bg-red-50 border-b border-red-200 px-4 py-2 text-sm text-red-600 flex items-center justify-between">
          <span>{error}</span>
          <button onClick={() => setError(null)} className="text-red-400 hover:text-red-600" aria-label="关闭错误提示">
            ✕
          </button>
        </div>
      )}
      <div className="flex-1 flex flex-col md:flex-row overflow-hidden">
        {sidebarOpen && (
          <button
            aria-label="关闭侧栏"
            onClick={() => setSidebarOpen(false)}
            className="fixed inset-0 z-30 bg-black/30 md:hidden"
          />
        )}
        <Sidebar
          stats={stats}
          statsLoading={statsLoading}
          onUpload={handleUpload}
          onDelete={handleDelete}
          onRefresh={refreshStats}
          open={sidebarOpen}
        />
        <main className="flex-1 min-h-0 flex flex-col bg-background">
          <ChatPanel messages={messages} stage={stage} />
          <ChatInput onSend={handleSend} onStop={handleStop} disabled={isStreaming} />
        </main>
      </div>
    </div>
  );
}
