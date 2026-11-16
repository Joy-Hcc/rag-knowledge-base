"use client";

import { useEffect, useRef, useState } from "react";
import ChatMessage, { type Message } from "./ChatMessage";

interface ChatPanelProps {
  messages: Message[];
  /** 当前阶段：retrieving / generating，用于提示"检索中/生成中" */
  stage?: string | null;
}

const STAGE_LABEL: Record<string, string> = {
  retrieving: "正在检索文档…",
  generating: "正在生成回答…",
};

export default function ChatPanel({ messages, stage }: ChatPanelProps) {
  const bottomRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  // 只有用户本来就贴在底部时才自动跟随，往上翻历史的时候不再把人拽回来
  const [stickToBottom, setStickToBottom] = useState(true);

  useEffect(() => {
    if (stickToBottom) {
      bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, stage, stickToBottom]);

  const handleScroll = () => {
    const el = containerRef.current;
    if (!el) return;
    const distance = el.scrollHeight - el.scrollTop - el.clientHeight;
    setStickToBottom(distance < 80);
  };

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
    <div className="relative flex-1 min-h-0">
      <div ref={containerRef} onScroll={handleScroll} className="h-full overflow-auto p-4 sm:p-6 space-y-4">
        {messages.map((msg) => (
          <ChatMessage key={msg.id} message={msg} />
        ))}
        {stage && STAGE_LABEL[stage] && (
          <div className="flex justify-start">
            <span className="rounded-full border border-border bg-surface px-2.5 py-1 text-[11px] text-muted">
              {STAGE_LABEL[stage]}
            </span>
          </div>
        )}
        <div ref={bottomRef} />
      </div>
      {!stickToBottom && (
        <button
          onClick={() => {
            setStickToBottom(true);
            bottomRef.current?.scrollIntoView({ behavior: "smooth" });
          }}
          className="absolute bottom-4 left-1/2 -translate-x-1/2 rounded-full border border-border bg-surface px-3 py-1 text-xs text-muted shadow-sm hover:text-accent hover:border-accent/40 transition-colors"
        >
          回到最新 ↓
        </button>
      )}
    </div>
  );
}
