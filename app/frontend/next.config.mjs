/** @type {import('next').NextConfig} */
// 출력 모드:
//   BUILD_TARGET=export   → out/ 정적 번들 (FastAPI가 직접 서빙, Node 런타임 불필요 — 로컬 데스크탑 런처용)
//   BUILD_TARGET=standalone (또는 미지정) → 기존 standalone 흐름 (build_portable_bundles.sh 호환)
import path from "path";
import { fileURLToPath } from "url";
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const BUILD_TARGET = process.env.BUILD_TARGET || "standalone";
const nextConfig = {
  reactStrictMode: true,
  output: BUILD_TARGET === "export" ? "export" : "standalone",
  // export 모드는 같은 origin (FastAPI가 정적+API 동시 서빙) → 상대 경로
  // standalone 모드는 별도 포트 → 절대 URL
  env: {
    NEXT_PUBLIC_API_BASE:
      process.env.NEXT_PUBLIC_API_BASE ??
      (BUILD_TARGET === "export" ? "" : "http://localhost:8765"),
  },
  webpack: (config, { isServer, webpack }) => {
    // tsconfig `paths` (@/* → ./src/*) 가 일부 빌드 환경(Windows)에서 자동 적용 안 돼
    // 명시적 alias 로 강제.
    config.resolve.alias = {
      ...(config.resolve.alias || {}),
      "@": path.resolve(__dirname, "src"),
    };
    // pptxgenjs references node:https / node:fs / node:zlib for its node entry path;
    // the browser bundle (which we trigger via dynamic import in a "use client" file)
    // does not need them. Map all `node:*` builtins to empty modules on the client.
    if (!isServer) {
      config.plugins = config.plugins || [];
      config.plugins.push(
        new webpack.NormalModuleReplacementPlugin(/^node:/, (resource) => {
          resource.request = resource.request.replace(/^node:/, "");
        })
      );
      config.resolve.fallback = {
        ...(config.resolve.fallback || {}),
        https: false,
        http: false,
        fs: false,
        path: false,
        zlib: false,
        stream: false,
        crypto: false,
      };
    }
    return config;
  },
};
export default nextConfig;
