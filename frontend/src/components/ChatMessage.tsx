import ReactMarkdown from "react-markdown";

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: string[];
  /** 是否仍在生成中：undefined 视为历史消息（非生成态） */
  pending?: boolean;
}

export default function ChatMessage({ message }: { message: Message }) {
  const isUser = message.role === "user";
  const isError = message.role === "assistant" && message.content.startsWith("请求失败");
  const isWaiting = message.role === "assistant" && message.content === "" && message.pending !== false;
  const isStreaming = message.role === "assistant" && message.pending === true && message.content !== "";

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
        {message.sources && message.sources.length > 0 && (
          <div className="mt-2 pt-2 border-t border-border/50">
            <p className="text-xs text-muted leading-5">
              来源: {message.sources.join(", ")}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
