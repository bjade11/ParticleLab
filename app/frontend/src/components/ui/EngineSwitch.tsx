"use client";

import clsx from "clsx";
import type { EngineId } from "@/lib/types";

const OPTIONS: { value: EngineId; label: string; hint: string }[] = [
  { value: "watershed", label: "Watershed", hint: "Otsu + 거리변환 + 마커 기반 분할" },
  { value: "ai_hybrid", label: "AI Hybrid", hint: "U-Net × YOLO 하이브리드 AI 분할" },
];

export default function EngineSwitch({
  value,
  onChange,
  disabled,
  className,
  size = "md",
}: {
  value: EngineId;
  onChange: (v: EngineId) => void;
  disabled?: boolean;
  className?: string;
  size?: "sm" | "md";
}) {
  return (
    <div
      role="radiogroup"
      aria-label="분할 엔진"
      className={clsx("inline-flex w-full rounded-lg border border-line bg-bg2 p-0.5", disabled && "opacity-60", className)}
    >
      {OPTIONS.map((o) => {
        const active = o.value === value;
        return (
          <button
            key={o.value}
            type="button"
            role="radio"
            aria-checked={active}
            disabled={disabled}
            title={o.hint}
            onClick={() => onChange(o.value)}
            className={clsx(
              "flex-1 rounded-md font-semibold transition-colors",
              size === "sm" ? "h-7 text-[11px]" : "h-8 text-[12px]",
              active ? "bg-[#191919] text-white shadow-sm" : "text-text-mute hover:text-text hover:bg-bg1",
            )}
          >
            {o.label}
          </button>
        );
      })}
    </div>
  );
}
