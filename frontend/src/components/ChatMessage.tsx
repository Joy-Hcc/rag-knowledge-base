export interface Message {
  role: "user" | "assistant";
  content: string;
  sources?: string[];
}

export default function ChatMessage({ message }: { message: Message }) {
  const isUser = message.role === "user";

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={`max-w-[80%] rounded-lg px-4 py-2.5 ${
          isUser
            ? "bg-accent text-white"
            : "bg-surface border border-border"
        }`}
      >
        <p className="text-sm whitespace-pre-wrap">{message.content}</p>
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
