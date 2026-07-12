export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: string[];
}

export default function ChatMessage({ message }: { message: Message }) {
  const isUser = message.role === "user";
  const isError = message.role === "assistant" && message.content.startsWith("请求失败");
  const isWaiting = message.role === "assistant" && message.content === "" && !message.sources?.length;
  const isStreaming = message.role === "assistant" && message.content !== "" && !message.sources?.length && !isError;

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={`max-w-[80%] rounded-lg px-4 py-2.5 ${
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
        ) : (
          <p className={`text-sm whitespace-pre-wrap ${isError ? "text-red-600" : ""}`}>
            {message.content}
            {isStreaming && (
              <span className="inline-block w-0.5 h-4 bg-accent ml-0.5 animate-pulse" />
            )}
          </p>
        )}
        {message.sources && message.sources.length > 0 && (
          <div className="mt-2 pt-2 border-t border-border/50">
            <p className="text-xs text-muted">
              来源: {message.sources.join(", ")}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
