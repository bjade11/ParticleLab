"use client";

import clsx from "clsx";
import type { ReactNode } from "react";

export type ChipTone = "default" | "good" | "bad" | "mute" | "ink" | "amber";

export interface ChipProps {
  children: ReactNode;
  tone?: ChipTone;
  dot?: boolean;
  active?: boolean;
  disabled?: boolean;
  onClick?: () => void;
  title?: string;
  className?: string;
}

/** globals.css `.chip` 기반. onClick 이 있으면 버튼(선택형 칩). */
export default function Chip({ children, tone = "default", dot, active, disabled, onClick, title, className }: ChipProps) {
  const cls = clsx(
    "chip whitespace-nowrap",
    tone === "good" && "chip-good",
    tone === "bad" && "chip-bad",
    tone === "mute" && "chip-mute",
    tone === "ink" && "!text-ink !border-ink/30 !bg-ink-soft",
    tone === "amber" && "!text-amber !border-amber/30 !bg-amber-soft",
    onClick && !disabled && "cursor-pointer hover:border-[#191919] hover:text-text",
    active && "!bg-[#191919] !text-white !border-[#191919]",
    disabled && "opacity-40 cursor-not-allowed",
    className,
  );
  if (onClick) {
    return (
      <button type="button" className={cls} onClick={onClick} disabled={disabled} title={title} aria-pressed={active}>
        {dot && <span className="chip-dot" />}
        {children}
      </button>
    );
  }
  return (
    <span className={cls} title={title}>
      {dot && <span className="chip-dot" />}
      {children}
    </span>
  );
}
