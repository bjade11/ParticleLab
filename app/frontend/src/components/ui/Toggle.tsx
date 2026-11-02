"use client";

import clsx from "clsx";
import type { ReactNode } from "react";

export interface ToggleProps {
  label?: ReactNode;
  hint?: string;
  checked: boolean;
  onChange: (v: boolean) => void;
  disabled?: boolean;
  className?: string;
  /** 라벨 없이 스위치만 */
  bare?: boolean;
}

/** globals.css 의 checkbox→switch 스타일을 사용하는 토글 */
export default function Toggle({ label, hint, checked, onChange, disabled, className, bare }: ToggleProps) {
  const input = (
    <input
      type="checkbox"
      checked={checked}
      disabled={disabled}
      onChange={(e) => onChange(e.target.checked)}
      aria-label={typeof label === "string" ? label : undefined}
    />
  );
  if (bare || !label) return <span className={className}>{input}</span>;
  return (
    <label className={clsx("toss-row py-1.5 flex items-center justify-between gap-3 cursor-pointer", disabled && "cursor-not-allowed", className)}>
      <div className="min-w-0">
        <div className={clsx("text-[12px] text-text", disabled && "opacity-50")}>{label}</div>
        {hint && <div className="text-[10px] text-text-faint leading-tight mt-0.5">{hint}</div>}
      </div>
      {input}
    </label>
  );
}
