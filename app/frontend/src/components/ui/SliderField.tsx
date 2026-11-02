"use client";

import { useEffect, useState } from "react";
import clsx from "clsx";
import { coerceNumber } from "./NumberField";

export interface SliderFieldProps {
  label: string;
  hint?: string;
  value: number;
  onChange: (v: number) => void;
  min: number;
  max: number;
  step?: number;
  integer?: boolean;
  odd?: boolean;
  unit?: string;
  disabled?: boolean;
  /** 숫자 입력은 슬라이더 범위 밖도 허용 */
  allowOutOfRange?: boolean;
  /** 소수 표시 자릿수 (기본 step 에서 유도) */
  decimals?: number;
  className?: string;
}

function decimalsOf(step: number) {
  const s = String(step);
  const i = s.indexOf(".");
  return i < 0 ? 0 : s.length - i - 1;
}

export default function SliderField({
  label,
  hint,
  value,
  onChange,
  min,
  max,
  step = 1,
  integer,
  odd,
  unit,
  disabled,
  allowOutOfRange,
  decimals,
  className,
}: SliderFieldProps) {
  const dec = decimals ?? decimalsOf(step);
  const fmt = (v: number) => (Number.isFinite(v) ? v.toFixed(dec) : "");
  const [text, setText] = useState(fmt(value));
  useEffect(() => setText(fmt(value)), [value, dec]); // eslint-disable-line react-hooks/exhaustive-deps

  const apply = (n: number) => {
    const v = coerceNumber(n, {
      min: allowOutOfRange ? undefined : min,
      max: allowOutOfRange ? undefined : max,
      integer,
      odd,
    });
    if (v !== value) onChange(v);
    setText(fmt(v));
  };

  const commitText = () => {
    const n = Number(text);
    if (!Number.isFinite(n)) {
      setText(fmt(value));
      return;
    }
    apply(n);
  };

  const sliderVal = Math.max(min, Math.min(max, value));

  return (
    <div className={clsx("toss-row py-1.5", className)}>
      <div className="flex items-center justify-between gap-2">
        <div className="min-w-0">
          <div className={clsx("text-[12px] text-text", disabled && "opacity-50")}>{label}</div>
          {hint && <div className="text-[10px] text-text-faint leading-tight mt-0.5">{hint}</div>}
        </div>
        <div className="flex items-center gap-1 flex-shrink-0">
          <input
            type="number"
            className="knob-num h-6 w-[72px] rounded-md border px-1.5 text-[12px] font-mono tabular-nums text-right disabled:opacity-50"
            value={text}
            step={odd ? 2 : step}
            disabled={disabled}
            onChange={(e) => setText(e.target.value)}
            onBlur={commitText}
            onKeyDown={(e) => {
              if (e.key === "Enter") (e.target as HTMLInputElement).blur();
            }}
          />
          {unit && <span className="text-[10px] text-text-faint w-7">{unit}</span>}
        </div>
      </div>
      <input
        type="range"
        className="mt-0.5"
        min={min}
        max={max}
        step={odd ? 2 : step}
        value={sliderVal}
        disabled={disabled}
        onChange={(e) => apply(Number(e.target.value))}
      />
    </div>
  );
}
