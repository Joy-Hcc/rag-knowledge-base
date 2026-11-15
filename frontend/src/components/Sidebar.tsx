"use client";

import UploadZone from "./UploadZone";
import DocumentList from "./DocumentList";
import StatsCard from "./StatsCard";
import type { StatsResponse } from "@/lib/api";

interface SidebarProps {
  stats: StatsResponse | null;
  statsLoading: boolean;
  onUpload: (file: File) => Promise<void>;
  onDelete: (filename: string) => Promise<void>;
  onRefresh: () => Promise<void>;
}

export default function Sidebar({ stats, statsLoading, onUpload, onDelete, onRefresh }: SidebarProps) {
  return (
    <aside className="w-full md:w-80 border-b md:border-b-0 md:border-r border-border bg-sidebar flex flex-col shrink-0 md:max-h-[calc(100vh-4rem)]">
      <div className="p-4 space-y-5 flex-1 overflow-auto">
        {/* 上传区 */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <h2 className="text-sm font-semibold text-foreground">上传文档</h2>
            <span className="text-[11px] font-medium text-success bg-white/70 border border-border rounded px-2 py-0.5">
              RAG Ready
            </span>
          </div>
          <UploadZone onUpload={onUpload} />
        </div>

        {/* 文档列表 */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <h2 className="text-sm font-semibold text-foreground">已上传</h2>
            <button
              onClick={onRefresh}
              className="text-xs font-medium text-muted hover:text-accent transition-colors"
            >
              刷新
            </button>
          </div>
          {statsLoading ? (
            <div className="text-sm text-muted py-6 text-center rounded-md border border-dashed border-border bg-white/60">加载中...</div>
          ) : (
            <DocumentList documents={stats?.documents ?? []} onDelete={onDelete} />
          )}
        </div>
      </div>

      {/* 统计信息 */}
      <div className="p-4 border-t border-border bg-surface-soft">
        <h2 className="text-sm font-semibold text-foreground mb-2">统计</h2>
        <StatsCard
          documentCount={stats?.document_count ?? 0}
          totalChars={stats?.total_chars ?? 0}
          chunkCount={stats?.chunk_count ?? 0}
        />
      </div>
    </aside>
  );
}
