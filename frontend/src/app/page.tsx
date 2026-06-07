"use client";

import { useState, useEffect, useCallback } from "react";
import Header from "@/components/Header";
import Sidebar from "@/components/Sidebar";
import ChatPanel from "@/components/ChatPanel";
import ChatInput from "@/components/ChatInput";
import type { Message } from "@/components/ChatMessage";
import type { StatsResponse } from "@/lib/api";
import * as api from "@/lib/api";

let msgId = 0;
const nextId = () => String(++msgId);

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [statsLoading, setStatsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

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

  useEffect(() => {
    refreshStats();
  }, [refreshStats]);

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

  // 发送问题
  const handleSend = useCallback(async (question: string) => {
    setMessages((prev) => [
      ...prev,
      { id: nextId(), role: "user", content: question },
    ]);

    try {
      const res = await api.query(question);
      setMessages((prev) => [
        ...prev,
        {
          id: nextId(),
          role: "assistant",
          content: res.answer,
          sources: res.sources,
        },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          id: nextId(),
          role: "assistant",
          content: `请求失败: ${err instanceof Error ? err.message : "未知错误"}`,
        },
      ]);
    }
  }, []);

  return (
    <div className="h-screen flex flex-col">
      <Header />
      {error && (
        <div className="bg-red-50 border-b border-red-200 px-4 py-2 text-sm text-red-600 flex items-center justify-between">
          <span>{error}</span>
          <button onClick={() => setError(null)} className="text-red-400 hover:text-red-600">
            ✕
          </button>
        </div>
      )}
      <div className="flex-1 flex overflow-hidden">
        <Sidebar
          stats={stats}
          statsLoading={statsLoading}
          onUpload={handleUpload}
          onDelete={handleDelete}
          onRefresh={refreshStats}
        />
        <main className="flex-1 flex flex-col bg-background">
          <ChatPanel messages={messages} />
          <ChatInput onSend={handleSend} />
        </main>
      </div>
    </div>
  );
}
