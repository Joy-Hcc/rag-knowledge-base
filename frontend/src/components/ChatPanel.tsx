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
      <div className="flex-1 flex items-center justify-center p-6">
        <div className="max-w-xl text-center">
          <div className="mx-auto mb-5 grid h-14 w-14 place-items-center rounded-lg bg-accent-light text-accent font-semibold">
            Q
          </div>
          <h2 className="text-xl font-semibold text-foreground mb-2">开始一次基于文档的问答</h2>
          <p className="text-muted text-sm leading-6">
            上传文档后，在这里向 AI 提问。系统会检索相关片段，并在回答下方展示来源。
          </p>
          <div className="mt-5 grid gap-2 text-left sm:grid-cols-3">
            {["总结这份文档", "提取关键结论", "根据来源回答"].map((tip) => (
              <div key={tip} className="rounded-md border border-border bg-surface px-3 py-2 text-xs text-muted">
                {tip}
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-auto p-4 sm:p-6 space-y-4">
      {messages.map((msg) => (
        <ChatMessage key={msg.id} message={msg} />
      ))}
      <div ref={bottomRef} />
    </div>
  );
}
