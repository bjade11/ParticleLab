"use client";

// 줌(휠) / 팬(드래그) / 맞춤·100% / 다운로드 / 라이트박스 이미지 뷰어.
// hybrid-lab 의 ImageBox 동작을 참고한 독립 컴포넌트.

import { useCallback, useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { Download, Maximize2, Minus, Plus, Scan, X } from "lucide-react";
import clsx from "clsx";

const MIN = 0.1;
const MAX = 16;
const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));

export interface ImageViewerProps {
  src: string | null | undefined;
  label?: string;
  /** 파일명 (다운로드용). 없으면 label 에서 생성 */
  downloadName?: string;
  /** 컨테이너 높이 (px 또는 CSS). 기본 "100%" → 부모가 높이를 정해야 함 */
  height?: number | string;
  /** 라이트박스 버튼 표시 */
  lightbox?: boolean;
  /** 비어 있을 때 메시지 */
  emptyText?: string;
  /** 외부에서 열어 둔 라이트박스 모드로 렌더 (툴바 어두운 테마) */
  dark?: boolean;
  className?: string;
  /** 이미지 로드 완료 콜백 */
  onLoaded?: (w: number, h: number) => void;
}

interface Transform {
  scale: number;
  tx: number;
  ty: number;
}

function useViewerTransform(src: string | null | undefined) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [natural, setNatural] = useState<{ w: number; h: number } | null>(null);
  const [t, setT] = useState<Transform>({ scale: 1, tx: 0, ty: 0 });
  const [fitScale, setFitScale] = useState(1);
  const [dragging, setDragging] = useState(false);
  const dragStart = useRef<{ x: number; y: number; tx: number; ty: number } | null>(null);
  const fitPending = useRef(true);

  const computeFit = useCallback(() => {
    const el = containerRef.current;
    if (!el || !natural) return 1;
    const r = el.getBoundingClientRect();
    if (r.width <= 0 || r.height <= 0) return 1;
    return Math.min(r.width / natural.w, r.height / natural.h);
  }, [natural]);

  const fit = useCallback(() => {
    const s = computeFit();
    setFitScale(s);
    setT({ scale: s, tx: 0, ty: 0 });
  }, [computeFit]);

  const actual = useCallback(() => setT({ scale: 1, tx: 0, ty: 0 }), []);

  // src 바뀌면 리셋
  useEffect(() => {
    setNatural(null);
    fitPending.current = true;
  }, [src]);

  // 이미지 크기 알게 되면 fit
  useEffect(() => {
    if (natural && fitPending.current) {
      fitPending.current = false;
      fit();
    }
  }, [natural, fit]);

  // 컨테이너 리사이즈 시 fit 상태였다면 재 fit
  useEffect(() => {
    const el = containerRef.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(() => {
      const s = computeFit();
      setFitScale(s);
      setT((prev) => (Math.abs(prev.scale - fitScale) < 1e-6 && prev.tx === 0 && prev.ty === 0 ? { scale: s, tx: 0, ty: 0 } : prev));
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [computeFit, fitScale]);

  const zoomAt = useCallback((factor: number, cx: number, cy: number) => {
    setT((prev) => {
      const next = clamp(prev.scale * factor, MIN, MAX);
      if (next === prev.scale) return prev;
      const ratio = next / prev.scale;
      return { scale: next, tx: cx - (cx - prev.tx) * ratio, ty: cy - (cy - prev.ty) * ratio };
    });
  }, []);

  const onWheel = useCallback(
    (e: React.WheelEvent<HTMLDivElement>) => {
      const rect = containerRef.current?.getBoundingClientRect();
      if (!rect) return;
      const cx = e.clientX - rect.left - rect.width / 2;
      const cy = e.clientY - rect.top - rect.height / 2;
      zoomAt(e.deltaY < 0 ? 1.15 : 1 / 1.15, cx, cy);
    },
    [zoomAt],
  );

  // wheel 기본 스크롤 차단 (passive 이슈 회피 위해 native 리스너)
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const h = (e: WheelEvent) => e.preventDefault();
    el.addEventListener("wheel", h, { passive: false });
    return () => el.removeEventListener("wheel", h);
  }, []);

  const onMouseDown = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      if (e.button !== 0) return;
      e.preventDefault();
      setDragging(true);
      dragStart.current = { x: e.clientX, y: e.clientY, tx: t.tx, ty: t.ty };
    },
    [t.tx, t.ty],
  );
  const onMouseMove = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    if (!dragStart.current) return;
    const d = dragStart.current;
    setT((prev) => ({ ...prev, tx: d.tx + (e.clientX - d.x), ty: d.ty + (e.clientY - d.y) }));
  }, []);
  const endDrag = useCallback(() => {
    setDragging(false);
    dragStart.current = null;
  }, []);

  const onDoubleClick = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      const rect = containerRef.current?.getBoundingClientRect();
      if (!rect) return;
      if (Math.abs(t.scale - fitScale) > 1e-6) {
        fit();
        return;
      }
      const cx = e.clientX - rect.left - rect.width / 2;
      const cy = e.clientY - rect.top - rect.height / 2;
      zoomAt(2.5, cx, cy);
    },
    [t.scale, fitScale, fit, zoomAt],
  );

  return {
    containerRef,
    natural,
    setNatural,
    t,
    fitScale,
    dragging,
    fit,
    actual,
    zoomIn: () => zoomAt(1.3, 0, 0),
    zoomOut: () => zoomAt(1 / 1.3, 0, 0),
    handlers: { onWheel, onMouseDown, onMouseMove, onMouseUp: endDrag, onMouseLeave: endDrag, onDoubleClick },
  };
}

function safeName(label: string | undefined, fallback = "image") {
  const s = (label ?? "").replace(/[^\w가-힣.-]+/g, "_").slice(0, 60);
  return (s || fallback) + ".png";
}

function ToolBtn({
  title,
  onClick,
  children,
  dark,
  href,
  download,
}: {
  title: string;
  onClick?: () => void;
  children: React.ReactNode;
  dark?: boolean;
  href?: string;
  download?: string;
}) {
  const cls = clsx(
    "w-7 h-7 inline-flex items-center justify-center rounded-md border text-[12px]",
    dark
      ? "border-white/20 bg-white/10 text-white hover:bg-white/20"
      : "border-line bg-bg1 text-text-mute hover:border-ink hover:text-ink",
  );
  if (href) {
    return (
      <a href={href} download={download} title={title} className={cls} onClick={(e) => e.stopPropagation()}>
        {children}
      </a>
    );
  }
  return (
    <button type="button" title={title} onClick={onClick} className={cls}>
      {children}
    </button>
  );
}

function ViewerCore({
  src,
  label,
  downloadName,
  height,
  dark,
  onOpenLightbox,
  onClose,
  emptyText,
  onLoaded,
  className,
}: ImageViewerProps & { onOpenLightbox?: () => void; onClose?: () => void }) {
  const v = useViewerTransform(src);
  const dl = downloadName ?? safeName(label);
  const pct = Math.round(v.t.scale * 100);

  return (
    <div className={clsx("flex flex-col min-w-0 min-h-0", className)} style={{ height: height ?? "100%" }}>
      <div
        className={clsx(
          "flex items-center justify-between gap-2 px-2 py-1.5 text-[11px] flex-shrink-0",
          dark ? "text-white" : "text-text-mute",
        )}
      >
        <span className="truncate font-medium">
          {label ?? ""}
          {v.natural && (
            <span className={clsx("ml-2 font-mono text-[10px]", dark ? "text-white/60" : "text-text-faint")}>
              {v.natural.w}×{v.natural.h}
            </span>
          )}
        </span>
        <div className="flex items-center gap-1 flex-shrink-0">
          <span className={clsx("font-mono text-[10px] tabular-nums w-11 text-right", dark ? "text-white/70" : "text-text-faint")}>
            {pct}%
          </span>
          <ToolBtn title="축소" onClick={v.zoomOut} dark={dark}>
            <Minus size={12} />
          </ToolBtn>
          <ToolBtn title="확대" onClick={v.zoomIn} dark={dark}>
            <Plus size={12} />
          </ToolBtn>
          <ToolBtn title="화면에 맞춤" onClick={v.fit} dark={dark}>
            <Scan size={12} />
          </ToolBtn>
          <button
            type="button"
            title="100% (원본 크기)"
            onClick={v.actual}
            className={clsx(
              "h-7 px-1.5 inline-flex items-center justify-center rounded-md border text-[10px] font-mono font-semibold",
              dark ? "border-white/20 bg-white/10 text-white hover:bg-white/20" : "border-line bg-bg1 text-text-mute hover:border-ink hover:text-ink",
            )}
          >
            1:1
          </button>
          {src && (
            <ToolBtn title="이미지 저장" href={src} download={dl} dark={dark}>
              <Download size={12} />
            </ToolBtn>
          )}
          {onOpenLightbox && src && (
            <ToolBtn title="크게 보기" onClick={onOpenLightbox} dark={dark}>
              <Maximize2 size={12} />
            </ToolBtn>
          )}
          {onClose && (
            <ToolBtn title="닫기 (ESC)" onClick={onClose} dark={dark}>
              <X size={13} />
            </ToolBtn>
          )}
        </div>
      </div>
      <div
        ref={v.containerRef}
        className={clsx(
          "relative flex-1 min-h-0 overflow-hidden select-none",
          dark ? "" : "rounded-lg border border-line",
        )}
        style={{
          background: "repeating-conic-gradient(#161616 0% 25%, #1f1f1f 0% 50%) 50% / 16px 16px, #0e0e0e",
          cursor: src ? (v.dragging ? "grabbing" : "grab") : "default",
        }}
        {...(src ? v.handlers : {})}
      >
        {src ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={src}
            alt={label ?? ""}
            draggable={false}
            onLoad={(e) => {
              const el = e.currentTarget;
              if (el.naturalWidth > 0 && el.naturalHeight > 0) {
                v.setNatural({ w: el.naturalWidth, h: el.naturalHeight });
                onLoaded?.(el.naturalWidth, el.naturalHeight);
              }
            }}
            className="absolute left-1/2 top-1/2 max-w-none"
            style={{
              width: v.natural ? v.natural.w : undefined,
              height: v.natural ? v.natural.h : undefined,
              transform: `translate(-50%, -50%) translate(${v.t.tx}px, ${v.t.ty}px) scale(${v.t.scale})`,
              transformOrigin: "center center",
              transition: v.dragging ? "none" : "transform 0.08s ease-out",
              imageRendering: v.t.scale >= 3 ? "pixelated" : "auto",
              visibility: v.natural ? "visible" : "hidden",
            }}
          />
        ) : (
          <div className="absolute inset-0 flex items-center justify-center text-[12px] text-white/40">
            {emptyText ?? "이미지가 없습니다"}
          </div>
        )}
      </div>
      <div className={clsx("text-[9px] mt-0.5 text-right flex-shrink-0", dark ? "text-white/50" : "text-text-faint")}>
        휠 줌 · 드래그 이동 · 더블클릭 맞춤/확대
      </div>
    </div>
  );
}

export default function ImageViewer(props: ImageViewerProps) {
  const { lightbox = true, ...rest } = props;
  const [open, setOpen] = useState(false);

  useEffect(() => {
    if (!open) return;
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => {
      document.body.style.overflow = prev;
      window.removeEventListener("keydown", onKey);
    };
  }, [open]);

  return (
    <>
      <ViewerCore {...rest} onOpenLightbox={lightbox ? () => setOpen(true) : undefined} />
      {open && typeof document !== "undefined" &&
        createPortal(
          <div className="fixed inset-0 z-[100] bg-black/90 backdrop-blur-sm flex flex-col fade-in p-3" onClick={() => setOpen(false)}>
            <div className="flex-1 min-h-0" onClick={(e) => e.stopPropagation()}>
              <ViewerCore {...rest} dark height="100%" onClose={() => setOpen(false)} />
            </div>
          </div>,
          document.body,
        )}
    </>
  );
}

/** 외부 제어형 라이트박스 (갤러리용). */
export function Lightbox({
  src,
  label,
  onClose,
  onPrev,
  onNext,
  counter,
}: {
  src: string | null;
  label?: string;
  onClose: () => void;
  onPrev?: () => void;
  onNext?: () => void;
  counter?: string;
}) {
  useEffect(() => {
    if (!src) return;
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
      else if (e.key === "ArrowLeft") onPrev?.();
      else if (e.key === "ArrowRight") onNext?.();
    };
    window.addEventListener("keydown", onKey);
    return () => {
      document.body.style.overflow = prev;
      window.removeEventListener("keydown", onKey);
    };
  }, [src, onClose, onPrev, onNext]);

  if (!src || typeof document === "undefined") return null;
  return createPortal(
    <div className="fixed inset-0 z-[100] bg-black/90 backdrop-blur-sm flex flex-col fade-in p-3" onClick={onClose}>
      <div className="flex-1 min-h-0 relative" onClick={(e) => e.stopPropagation()}>
        <ViewerCore src={src} label={label} dark height="100%" onClose={onClose} />
        {onPrev && (
          <button
            type="button"
            onClick={onPrev}
            className="absolute left-2 top-1/2 -translate-y-1/2 w-10 h-10 rounded-full bg-white/10 hover:bg-white/25 text-white text-lg border border-white/20"
            aria-label="이전"
          >
            ‹
          </button>
        )}
        {onNext && (
          <button
            type="button"
            onClick={onNext}
            className="absolute right-2 top-1/2 -translate-y-1/2 w-10 h-10 rounded-full bg-white/10 hover:bg-white/25 text-white text-lg border border-white/20"
            aria-label="다음"
          >
            ›
          </button>
        )}
      </div>
      <div className="text-center text-[11px] text-white/60 py-1.5">
        {counter ? `${counter} · ` : ""}배경 클릭 / ESC 로 닫기{onPrev || onNext ? " · ←/→ 이동" : ""}
      </div>
    </div>,
    document.body,
  );
}
