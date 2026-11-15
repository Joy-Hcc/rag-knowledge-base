interface HeaderProps {
  onNewConversation?: () => void;
  hasConversation?: boolean;
}

export default function Header({ onNewConversation, hasConversation }: HeaderProps) {
  return (
    <header className="min-h-16 border-b border-border bg-surface/95 flex items-center gap-4 px-4 sm:px-6 shrink-0 shadow-sm shadow-zinc-200/40">
      <div className="flex items-center gap-3 min-w-0">
        <span className="grid h-9 w-9 place-items-center rounded-md bg-accent text-sm font-semibold text-white">
          AI
        </span>
        <div className="min-w-0">
          <h1 className="text-base font-semibold text-foreground leading-tight">AI 知识库</h1>
          <p className="text-xs text-muted sm:hidden">上传文档，向 AI 提问</p>
        </div>
      </div>
      <p className="ml-4 text-sm text-muted hidden sm:block">
        上传文档，向 AI 提问
      </p>
      {hasConversation && onNewConversation && (
        <button
          onClick={onNewConversation}
          className="ml-auto shrink-0 text-sm font-medium text-accent hover:text-white transition-colors px-3 py-2 rounded-md border border-accent/30 hover:bg-accent"
        >
          + 新建对话
        </button>
      )}
    </header>
  );
}
