"use client";

import { useState, type ReactNode } from "react";
import clsx from "clsx";
import { ChevronDown, RotateCcw } from "lucide-react";

export interface ParamSectionProps {
  title: string;
  subtitle?: string;
  defaultOpen?: boolean;
  /** 기본값 리셋 버튼 콜백 (없으면 버튼 숨김) */
  onReset?: () => void;
  /** 제목 우측 추가 요소 (예: enable 토글) */
  right?: ReactNode;
  badge?: ReactNode;
  children: ReactNode;
  className?: string;
}

export default function ParamSection({
  title,
  subtitle,
  defaultOpen = true,
  onReset,
  right,
  badge,
  children,
  className,
}: ParamSectionProps) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <section className={clsx("app-card overflow-hidden", className)}>
      <header className="flex items-center gap-2 px-3 py-2 bg-bg2/60 border-b border-line-soft">
        <button
          type="button"
          onClick={() => setOpen((o) => !o)}
          className="flex items-center gap-2 min-w-0 flex-1 text-left"
          aria-expanded={open}
        >
          <ChevronDown size={14} className={clsx("text-text-faint transition-transform flex-shrink-0", !open && "-rotate-90")} />
          <span className="text-[12px] font-semibold text-text truncate">{title}</span>
          {badge}
          {subtitle && <span className="text-[10px] text-text-faint truncate hidden xl:inline">{subtitle}</span>}
        </button>
        {right && <div className="flex-shrink-0" onClick={(e) => e.stopPropagation()}>{right}</div>}
        {onReset && (
          <button
            type="button"
            onClick={onReset}
            title="이 섹션을 기본값으로"
            className="flex items-center gap-1 h-6 px-1.5 rounded-md text-[10px] text-text-faint hover:text-text hover:bg-bg3/60 flex-shrink-0"
          >
            <RotateCcw size={11} />
            기본값
          </button>
        )}
      </header>
      {open && <div className="px-3 py-1 fade-in">{children}</div>}
    </section>
  );
}
