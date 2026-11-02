"use client";

import clsx from "clsx";

export interface SelectOption<T extends string> {
  value: T;
  label: string;
}

export interface SelectFieldProps<T extends string> {
  label?: string;
  hint?: string;
  value: T;
  options: readonly SelectOption<T>[];
  onChange: (v: T) => void;
  disabled?: boolean;
  className?: string;
  selectClassName?: string;
}

export default function SelectField<T extends string>({
  label,
  hint,
  value,
  options,
  onChange,
  disabled,
  className,
  selectClassName,
}: SelectFieldProps<T>) {
  const sel = (
    <select
      className={clsx("h-7 rounded-md border px-2 text-[12px] min-w-[120px] disabled:opacity-50", selectClassName)}
      value={value}
      disabled={disabled}
      onChange={(e) => onChange(e.target.value as T)}
    >
      {options.map((o) => (
        <option key={o.value} value={o.value}>
          {o.label}
        </option>
      ))}
    </select>
  );
  if (!label) return <div className={className}>{sel}</div>;
  return (
    <div className={clsx("toss-row py-1.5 flex items-center justify-between gap-3", className)}>
      <div className="min-w-0">
        <div className={clsx("text-[12px] text-text", disabled && "opacity-50")}>{label}</div>
        {hint && <div className="text-[10px] text-text-faint leading-tight mt-0.5">{hint}</div>}
      </div>
      {sel}
    </div>
  );
}
