import { useState } from "react";
import ReactMarkdown from "react-markdown";
import type { Citation } from "@/lib/api";

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: string[];
  citations?: Citation[];
  /** 是否仍在生成中：undefined 视为历史消息（非生成态） */
  pending?: boolean;
  /** 用户主动停止生成 */
  stopped?: boolean;
}

export default function ChatMessage({ message }: { message: Message }) {
  const [copied, setCopied] = useState(false);

  const isUser = message.role === "user";
  const isError = message.role === "assistant" && message.content.startsWith("请求失败");
  const isWaiting = message.role === "assistant" && message.content === "" && message.pending !== false;
  const isStreaming = message.role === "assistant" && message.pending === true && message.content !== "";
  const citations = message.citations ?? [];

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(message.content);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // 剪贴板不可用（非 https / 无权限）时静默失败，不影响阅读
    }
  };

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={`max-w-[88%] sm:max-w-[76%] rounded-lg px-4 py-3 shadow-sm ${
          isUser
            ? "bg-accent text-white"
            : isError
              ? "bg-red-50 border border-red-200"
              : "bg-surface border border-border"
        }`}
      >
        {isWaiting ? (
          <div className="flex items-center gap-1 py-1">
            <span className="w-1.5 h-1.5 bg-muted rounded-full animate-bounce" style={{ animationDelay: "0ms" }} />
            <span className="w-1.5 h-1.5 bg-muted rounded-full animate-bounce" style={{ animationDelay: "150ms" }} />
            <span className="w-1.5 h-1.5 bg-muted rounded-full animate-bounce" style={{ animationDelay: "300ms" }} />
          </div>
        ) : isUser || isError ? (
          <p className={`text-sm whitespace-pre-wrap ${isError ? "text-red-600" : ""}`}>
            {message.content}
          </p>
        ) : (
          <div className="text-sm markdown-body">
            <ReactMarkdown>{message.content}</ReactMarkdown>
            {isStreaming && (
              <span className="inline-block w-0.5 h-4 bg-accent ml-0.5 animate-pulse" />
            )}
          </div>
        )}

        {message.stopped && (
          <p className="mt-1 text-[11px] text-muted">已停止生成</p>
        )}

        {message.role === "assistant" && message.content !== "" && !isError && (
          <div className="mt-2 pt-2 border-t border-border/50 space-y-1.5">
            {citations.length > 0 ? (
              <details className="group">
                <summary className="cursor-pointer select-none text-xs text-muted hover:text-accent transition-colors">
                  引用片段 · {citations.length} 段（{message.sources?.join("、")}）
                </summary>
                <ol className="mt-2 space-y-2">
                  {citations.map((c, i) => (
                    <li key={`${c.filename}-${i}`} className="rounded border border-border/70 bg-surface-soft px-2.5 py-2">
                      <p className="text-[11px] font-medium text-accent/90 mb-1">
                        [{i + 1}] {c.filename}
                      </p>
                      <p className="text-[11px] leading-5 text-muted whitespace-pre-wrap">{c.text}</p>
                    </li>
                  ))}
                </ol>
              </details>
            ) : message.sources && message.sources.length > 0 ? (
              <p className="text-xs text-muted leading-5">
                来源: {message.sources.join(", ")}
              </p>
            ) : null}
            <button
              onClick={handleCopy}
              className="text-[11px] text-muted hover:text-accent transition-colors"
              aria-label="复制回答"
            >
              {copied ? "已复制" : "复制回答"}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
