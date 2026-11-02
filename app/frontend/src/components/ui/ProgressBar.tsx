"use client";

import clsx from "clsx";

export default function ProgressBar({
  value,
  indeterminate,
  tone = "ink",
  className,
  showLabel = true,
}: {
  /** 0~100 */
  value: number;
  indeterminate?: boolean;
  tone?: "ink" | "good" | "bad" | "dark";
  className?: string;
  showLabel?: boolean;
}) {
  const pct = Math.max(0, Math.min(100, Number.isFinite(value) ? value : 0));
  const color =
    tone === "good" ? "bg-good" : tone === "bad" ? "bg-bad" : tone === "dark" ? "bg-[#191919]" : "bg-ink";
  return (
    <div className={clsx("flex items-center gap-2", className)}>
      <div className="relative flex-1 h-2 rounded-full bg-bg3 overflow-hidden">
        {indeterminate ? (
          <div
            className={clsx("absolute inset-y-0 w-1/3 rounded-full", color)}
            style={{
              animation: "plabSlide 1.2s ease-in-out infinite",
            }}
          />
        ) : (
          <div className={clsx("h-full rounded-full transition-[width] duration-300", color)} style={{ width: `${pct}%` }} />
        )}
      </div>
      {showLabel && (
        <span className="text-[11px] font-mono tabular-nums text-text-mute w-10 text-right">{indeterminate ? "…" : `${pct.toFixed(0)}%`}</span>
      )}
    </div>
  );
}
