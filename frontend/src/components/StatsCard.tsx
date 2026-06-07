interface StatsCardProps {
  documentCount: number;
  totalChars: number;
  chunkCount: number;
}

export default function StatsCard({ documentCount, totalChars, chunkCount }: StatsCardProps) {
  return (
    <div className="bg-accent-light rounded-lg p-3 space-y-1">
      <div className="flex justify-between text-sm">
        <span className="text-muted">文档数</span>
        <span className="font-medium">{documentCount}</span>
      </div>
      <div className="flex justify-between text-sm">
        <span className="text-muted">总字符</span>
        <span className="font-medium">{totalChars.toLocaleString()}</span>
      </div>
      <div className="flex justify-between text-sm">
        <span className="text-muted">向量块</span>
        <span className="font-medium">{chunkCount.toLocaleString()}</span>
      </div>
    </div>
  );
}
