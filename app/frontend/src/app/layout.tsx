import "./globals.css";
import type { Metadata, Viewport } from "next";
import type { ReactNode } from "react";
import AppShell from "@/components/AppShell";

export const metadata: Metadata = {
  title: "Particle Lab — SEM 입자 분석",
  description: "SEM 입자 분석: Watershed / AI Hybrid 분할 · 파라미터 튜너 · 배치 분석 (Excel/PPT) · AI 진단 콘솔",
};

export const viewport: Viewport = {
  themeColor: "#ffffff",
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="ko">
      <body className="font-sans antialiased min-h-screen">
        <AppShell />
        {children}
      </body>
    </html>
  );
}
