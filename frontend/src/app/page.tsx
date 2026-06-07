"use client";

import { useState, useEffect, useCallback } from "react";
import Header from "@/components/Header";
import Sidebar from "@/components/Sidebar";
import ChatPanel from "@/components/ChatPanel";
import ChatInput from "@/components/ChatInput";
import type { Message } from "@/components/ChatMessage";
import type { StatsResponse } from "@/lib/api";
import * as api from "@/lib/api";

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [stats, setStats] = useState<StatsResponse | null>(null);

  // 加载统计信息
  const refreshStats = useCallback(async () => {
    try {
      const data = await api.getStats();
      setStats(data);
    } catch {
      // 后端未连接时静默处理
    }
  }, []);

  useEffect(() => {
    refreshStats();
  }, [refreshStats]);

  // 上传文档
  const handleUpload = useCallback(
    async (file: File) => {
      await api.uploadDocument(file);
      await refreshStats();
    },
    [refreshStats]
  );

  // 删除文档
  const handleDelete = useCallback(
    async (filename: string) => {
      await api.deleteDocument(filename);
      await refreshStats();
    },
    [refreshStats]
  );

  // 发送问题
  const handleSend = useCallback(async (question: string) => {
    // 添加用户消息
    setMessages((prev) => [...prev, { role: "user", content: question }]);

    try {
      const res = await api.query(question);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: res.answer,
          sources: res.sources,
        },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: `请求失败: ${err instanceof Error ? err.message : "未知错误"}`,
        },
      ]);
    }
  }, []);

  return (
    <div className="h-screen flex flex-col">
      <Header />
      <div className="flex-1 flex overflow-hidden">
        <Sidebar
          stats={stats}
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
