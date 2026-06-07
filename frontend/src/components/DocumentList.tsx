"use client";

interface DocumentListProps {
  documents: string[];
  onDelete: (filename: string) => Promise<void>;
}

export default function DocumentList({ documents, onDelete }: DocumentListProps) {
  if (documents.length === 0) {
    return (
      <div className="text-sm text-muted py-4 text-center">
        暂无文档
      </div>
    );
  }

  return (
    <div className="space-y-1">
      {documents.map((doc) => (
        <div
          key={doc}
          className="flex items-center gap-2 px-3 py-2 rounded-md hover:bg-border/50 group"
        >
          <span className="text-sm truncate flex-1" title={doc}>
            📄 {doc}
          </span>
          <button
            onClick={() => onDelete(doc)}
            className="text-muted hover:text-red-500 opacity-0 group-hover:opacity-100 transition-opacity text-xs"
            title="删除文档"
          >
            ✕
          </button>
        </div>
      ))}
    </div>
  );
}
