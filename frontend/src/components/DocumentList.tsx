"use client";

interface DocumentListProps {
  documents: string[];
  onDelete: (filename: string) => Promise<void>;
}

export default function DocumentList({ documents, onDelete }: DocumentListProps) {
  if (documents.length === 0) {
    return (
      <div className="text-sm text-muted py-6 text-center rounded-md border border-dashed border-border bg-white/60">
        暂无文档
      </div>
    );
  }

  return (
    <div className="space-y-1.5">
      {documents.map((doc) => (
        <div
          key={doc}
          className="flex items-center gap-2 px-3 py-2 rounded-md border border-transparent bg-white/50 hover:bg-white hover:border-border group transition-colors"
        >
          <span className="text-sm truncate flex-1" title={doc}>
            📄 {doc}
          </span>
          <button
            onClick={() => {
              if (window.confirm(`确认删除文档「${doc}」？此操作不可撤销。`)) {
                onDelete(doc);
              }
            }}
            aria-label={`删除文档 ${doc}`}
            className="text-muted hover:text-red-500 opacity-100 sm:opacity-0 group-hover:opacity-100 transition-opacity text-xs"
          >
            ✕
          </button>
        </div>
      ))}
    </div>
  );
}
