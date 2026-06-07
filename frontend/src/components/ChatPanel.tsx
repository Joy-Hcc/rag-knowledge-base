"use client";

import { useEffect, useRef } from "react";
import ChatMessage, { type Message } from "./ChatMessage";

interface ChatPanelProps {
  messages: Message[];
}

export default function ChatPanel({ messages }: ChatPanelProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  // 自动滚动到底部
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  if (messages.length === 0) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="text-center">
          <p className="text-4xl mb-3">💬</p>
          <p className="text-muted text-sm">
            上传文档后，在这里向 AI 提问
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-auto p-4 space-y-4">
      {messages.map((msg) => (
        <ChatMessage key={msg.id} message={msg} />
      ))}
      <div ref={bottomRef} />
    </div>
  );
}
