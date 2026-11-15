import { NextRequest, NextResponse } from "next/server";

// 服务端代理：浏览器只访问同源 /api，由此转发到后端并注入鉴权头。
// API_KEY 不带 NEXT_PUBLIC_ 前缀，只在服务端可见，不会进入客户端 bundle。
const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8000";
const API_KEY = process.env.API_KEY || "";

export function middleware(request: NextRequest) {
  const path = request.nextUrl.pathname.replace(/^\/api/, "");
  const url = new URL(path + request.nextUrl.search, BACKEND_URL);

  const headers = new Headers(request.headers);
  if (API_KEY) {
    headers.set("X-API-Key", API_KEY);
  }
  return NextResponse.rewrite(url, { request: { headers } });
}

export const config = {
  matcher: "/api/:path*",
};
