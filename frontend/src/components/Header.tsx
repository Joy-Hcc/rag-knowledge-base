interface HeaderProps {
  onNewConversation?: () => void;
  hasConversation?: boolean;
}

export default function Header({ onNewConversation, hasConversation }: HeaderProps) {
  return (
    <header className="h-14 border-b border-border bg-surface flex items-center px-6 shrink-0">
      <div className="flex items-center gap-2">
        <span className="text-lg">📚</span>
        <h1 className="text-base font-semibold text-foreground">AI 知识库</h1>
      </div>
      <p className="ml-4 text-sm text-muted hidden sm:block">
        上传文档，向 AI 提问
      </p>
      {hasConversation && onNewConversation && (
        <button
          onClick={onNewConversation}
          className="ml-auto text-sm text-muted hover:text-foreground transition-colors px-3 py-1 rounded border border-border hover:border-accent/50"
        >
          + 新建对话
        </button>
      )}
    </header>
  );
}
