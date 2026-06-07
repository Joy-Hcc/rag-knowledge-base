"use client";

import { useRef, useState, useCallback } from "react";

const MAX_SIZE = 10 * 1024 * 1024; // 10MB
const ALLOWED_EXTENSIONS = [".pdf", ".docx", ".txt"];

interface UploadZoneProps {
  onUpload: (file: File) => Promise<void>;
}

function validateFile(file: File): string | null {
  const ext = "." + file.name.split(".").pop()?.toLowerCase();
  if (!ALLOWED_EXTENSIONS.includes(ext)) {
    return `不支持的文件格式: ${ext}，请上传 PDF/DOCX/TXT`;
  }
  if (file.size > MAX_SIZE) {
    return `文件过大 (${(file.size / 1024 / 1024).toFixed(1)}MB)，最大允许 10MB`;
  }
  return null;
}

export default function UploadZone({ onUpload }: UploadZoneProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFile = useCallback(
    async (file: File) => {
      const err = validateFile(file);
      if (err) {
        setError(err);
        return;
      }
      setError(null);
      setIsUploading(true);
      try {
        await onUpload(file);
      } catch (e) {
        setError(e instanceof Error ? e.message : "上传失败");
      } finally {
        setIsUploading(false);
      }
    },
    [onUpload]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  return (
    <div
      className={`border-2 border-dashed rounded-lg p-4 text-center cursor-pointer transition-colors ${
        isDragging
          ? "border-accent bg-accent-light"
          : "border-border hover:border-accent/50"
      }`}
      onClick={() => inputRef.current?.click()}
      onDragOver={(e) => {
        e.preventDefault();
        setIsDragging(true);
      }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={handleDrop}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".pdf,.docx,.txt"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) handleFile(file);
          e.target.value = "";
        }}
      />
      {error && (
        <p className="text-xs text-red-500 mb-2">{error}</p>
      )}
      {isUploading ? (
        <div className="flex items-center justify-center gap-2 text-accent">
          <div className="w-4 h-4 border-2 border-accent border-t-transparent rounded-full animate-spin" />
          <span className="text-sm">上传中...</span>
        </div>
      ) : (
        <div>
          <p className="text-sm text-muted">
            拖拽文件到这里，或点击选择
          </p>
          <p className="text-xs text-muted mt-1">
            支持 PDF / DOCX / TXT，最大 10MB
          </p>
        </div>
      )}
    </div>
  );
}
