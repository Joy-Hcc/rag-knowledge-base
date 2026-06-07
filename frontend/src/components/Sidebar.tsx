"use client";

import UploadZone from "./UploadZone";
import DocumentList from "./DocumentList";
import StatsCard from "./StatsCard";
import type { StatsResponse } from "@/lib/api";

interface SidebarProps {
  stats: StatsResponse | null;
  onUpload: (file: File) => Promise<void>;
  onDelete: (filename: string) => Promise<void>;
  onRefresh: () => Promise<void>;
}

export default function Sidebar({ stats, onUpload, onDelete, onRefresh }: SidebarProps) {
  return (
    <aside className="w-72 border-r border-border bg-sidebar flex flex-col shrink-0">
      <div className="p-4 space-y-4 flex-1 overflow-auto">
        {/* 上传区 */}
        <div>
          <h2 className="text-sm font-medium text-foreground mb-2">上传文档</h2>
          <UploadZone onUpload={onUpload} />
        </div>

        {/* 文档列表 */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <h2 className="text-sm font-medium text-foreground">已上传</h2>
            <button
              onClick={onRefresh}
              className="text-xs text-muted hover:text-accent transition-colors"
            >
              刷新
            </button>
          </div>
          <DocumentList documents={stats?.documents ?? []} onDelete={onDelete} />
        </div>
      </div>

      {/* 统计信息 */}
      <div className="p-4 border-t border-border">
        <h2 className="text-sm font-medium text-foreground mb-2">统计</h2>
        <StatsCard
          documentCount={stats?.document_count ?? 0}
          totalChars={stats?.total_chars ?? 0}
          chunkCount={stats?.chunk_count ?? 0}
        />
      </div>
    </aside>
  );
}
