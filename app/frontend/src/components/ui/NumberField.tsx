"use client";

import { useEffect, useState } from "react";
import clsx from "clsx";

export interface NumberFieldProps {
  label?: string;
  hint?: string;
  value: number | null;
  onChange: (v: number) => void;
  min?: number;
  max?: number;
  step?: number;
  /** 정수만 */
  integer?: boolean;
  /** 홀수 강제 (커널 크기) */
  odd?: boolean;
  unit?: string;
  disabled?: boolean;
  /** 빈 값 허용 → onChange 에 NaN 대신 onNull 호출 */
  nullable?: boolean;
  onNull?: () => void;
  placeholder?: string;
  className?: string;
  inputClassName?: string;
  /** label 을 왼쪽, 입력을 오른쪽에 (행 레이아웃). 기본 true */
  row?: boolean;
}

export function coerceNumber(
  raw: number,
  opts: { min?: number; max?: number; integer?: boolean; odd?: boolean },
): number {
  let v = raw;
  if (!Number.isFinite(v)) return opts.min ?? 0;
  if (opts.integer || opts.odd) v = Math.round(v);
  if (opts.odd && v % 2 === 0) v = v + 1;
  if (opts.min !== undefined && v < opts.min) v = opts.min;
  if (opts.max !== undefined && v > opts.max) v = opts.max;
  if (opts.odd && v % 2 === 0) v = v - 1 >= (opts.min ?? 1) ? v - 1 : v + 1;
  return v;
}

export default function NumberField({
  label,
  hint,
  value,
  onChange,
  min,
  max,
  step,
  integer,
  odd,
  unit,
  disabled,
  nullable,
  onNull,
  placeholder,
  className,
  inputClassName,
  row = true,
}: NumberFieldProps) {
  const [text, setText] = useState(value === null ? "" : String(value));
  useEffect(() => {
    setText(value === null ? "" : String(value));
  }, [value]);

  const commit = () => {
    const s = text.trim();
    if (s === "") {
      if (nullable) {
        onNull?.();
        return;
      }
      setText(value === null ? "" : String(value));
      return;
    }
    const n = Number(s);
    if (!Number.isFinite(n)) {
      setText(value === null ? "" : String(value));
      return;
    }
    const v = coerceNumber(n, { min, max, integer, odd });
    setText(String(v));
    if (v !== value) onChange(v);
  };

  const input = (
    <div className="flex items-center gap-1">
      <input
        type="number"
        className={clsx(
          "h-7 rounded-md border px-2 text-[12px] font-mono tabular-nums text-right w-[88px] disabled:opacity-50",
          inputClassName,
        )}
        value={text}
        min={min}
        max={max}
        step={step ?? (integer || odd ? 1 : "any")}
        disabled={disabled}
        placeholder={placeholder}
        onChange={(e) => setText(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => {
          if (e.key === "Enter") (e.target as HTMLInputElement).blur();
        }}
      />
      {unit && <span className="text-[10px] text-text-faint w-8">{unit}</span>}
    </div>
  );

  if (!label) return <div className={className}>{input}</div>;

  return (
    <div className={clsx("toss-row py-1.5", row ? "flex items-center justify-between gap-3" : "flex flex-col gap-1", className)}>
      <div className="min-w-0">
        <div className={clsx("text-[12px] text-text", disabled && "opacity-50")}>{label}</div>
        {hint && <div className="text-[10px] text-text-faint leading-tight mt-0.5">{hint}</div>}
      </div>
      {input}
    </div>
  );
}
