import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AI 知识库问答系统",
  description: "上传文档，AI 检索相关内容后回答你的问题",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="zh-CN" className="h-full antialiased">
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}
