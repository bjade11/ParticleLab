"use client";

import { useMemo, useState, type ReactNode } from "react";
import clsx from "clsx";
import { ArrowDown, ArrowUp, ArrowUpDown } from "lucide-react";

export interface DataTableColumn<Row> {
  key: string;
  label: ReactNode;
  /** 셀 값 추출 (정렬·표시 공용). 기본 row[key] */
  accessor?: (row: Row) => unknown;
  /** 표시 포매터 */
  render?: (value: unknown, row: Row) => ReactNode;
  align?: "left" | "right" | "center";
  width?: number | string;
  /** 숫자 소수 자릿수 (number 값에만) */
  decimals?: number;
  sortable?: boolean;
  className?: string;
}

export interface DataTableProps<Row> {
  rows: Row[];
  columns: DataTableColumn<Row>[];
  rowKey?: (row: Row, i: number) => string | number;
  /** 최대 표시 행 (기본 500) */
  maxRows?: number;
  /** 컨테이너 최대 높이 (스크롤) */
  maxHeight?: number | string;
  dense?: boolean;
  emptyText?: string;
  onRowClick?: (row: Row) => void;
  selectedKey?: string | number | null;
  className?: string;
  stickyHeader?: boolean;
  defaultSort?: { key: string; dir: "asc" | "desc" };
}

function defaultFormat(v: unknown, decimals?: number): ReactNode {
  if (v === null || v === undefined) return <span className="text-text-faint">–</span>;
  if (typeof v === "number") {
    if (!Number.isFinite(v)) return <span className="text-text-faint">–</span>;
    if (decimals !== undefined) return v.toFixed(decimals);
    return Number.isInteger(v) ? String(v) : v.toFixed(Math.abs(v) >= 100 ? 1 : 3);
  }
  if (typeof v === "boolean") return v ? "예" : "아니오";
  return String(v);
}

function compare(a: unknown, b: unknown): number {
  const an = a === null || a === undefined || (typeof a === "number" && Number.isNaN(a));
  const bn = b === null || b === undefined || (typeof b === "number" && Number.isNaN(b));
  if (an && bn) return 0;
  if (an) return 1;
  if (bn) return -1;
  if (typeof a === "number" && typeof b === "number") return a - b;
  return String(a).localeCompare(String(b), "ko", { numeric: true });
}

export default function DataTable<Row>({
  rows,
  columns,
  rowKey,
  maxRows = 500,
  maxHeight = 360,
  dense,
  emptyText = "데이터가 없습니다",
  onRowClick,
  selectedKey,
  className,
  stickyHeader = true,
  defaultSort,
}: DataTableProps<Row>) {
  const [sort, setSort] = useState<{ key: string; dir: "asc" | "desc" } | null>(defaultSort ?? null);

  const acc = (col: DataTableColumn<Row>, row: Row): unknown =>
    col.accessor ? col.accessor(row) : (row as Record<string, unknown>)[col.key];

  const sorted = useMemo(() => {
    const limited = rows.slice(0, maxRows);
    if (!sort) return limited;
    const col = columns.find((c) => c.key === sort.key);
    if (!col) return limited;
    const dirMul = sort.dir === "asc" ? 1 : -1;
    return limited
      .map((r, i) => ({ r, i }))
      .sort((x, y) => {
        const c = compare(acc(col, x.r), acc(col, y.r));
        return c !== 0 ? c * dirMul : x.i - y.i;
      })
      .map((x) => x.r);
  }, [rows, maxRows, sort, columns]); // eslint-disable-line react-hooks/exhaustive-deps

  const toggleSort = (key: string) => {
    setSort((prev) => {
      if (!prev || prev.key !== key) return { key, dir: "asc" };
      if (prev.dir === "asc") return { key, dir: "desc" };
      return null;
    });
  };

  const truncated = rows.length > maxRows;

  return (
    <div className={clsx("app-card overflow-hidden", className)}>
      <div className="overflow-auto" style={{ maxHeight }}>
        <table className="w-full text-[11px] border-collapse">
          <thead className={clsx(stickyHeader && "sticky top-0 z-10")}>
            <tr className="bg-bg2 text-text-mute">
              {columns.map((c) => {
                const sortable = c.sortable !== false;
                const active = sort?.key === c.key;
                return (
                  <th
                    key={c.key}
                    style={{ width: c.width }}
                    className={clsx(
                      "font-semibold px-2 border-b border-line whitespace-nowrap select-none",
                      dense ? "py-1" : "py-1.5",
                      c.align === "right" ? "text-right" : c.align === "center" ? "text-center" : "text-left",
                      sortable && "cursor-pointer hover:text-text",
                      c.className,
                    )}
                    onClick={sortable ? () => toggleSort(c.key) : undefined}
                  >
                    <span className="inline-flex items-center gap-1">
                      {c.label}
                      {sortable &&
                        (active ? (
                          sort?.dir === "asc" ? (
                            <ArrowUp size={10} className="text-ink" />
                          ) : (
                            <ArrowDown size={10} className="text-ink" />
                          )
                        ) : (
                          <ArrowUpDown size={10} className="text-text-faint/60" />
                        ))}
                    </span>
                  </th>
                );
              })}
            </tr>
          </thead>
          <tbody>
            {sorted.length === 0 && (
              <tr>
                <td colSpan={columns.length} className="px-3 py-6 text-center text-text-faint">
                  {emptyText}
                </td>
              </tr>
            )}
            {sorted.map((row, i) => {
              const k = rowKey ? rowKey(row, i) : i;
              const selected = selectedKey !== undefined && selectedKey !== null && k === selectedKey;
              return (
                <tr
                  key={k}
                  className={clsx(
                    "toss-row",
                    onRowClick && "cursor-pointer hover:bg-bg2",
                    selected && "bg-ink-soft",
                  )}
                  onClick={onRowClick ? () => onRowClick(row) : undefined}
                >
                  {columns.map((c) => {
                    const v = acc(c, row);
                    return (
                      <td
                        key={c.key}
                        className={clsx(
                          "px-2 font-mono tabular-nums whitespace-nowrap",
                          dense ? "py-0.5" : "py-1",
                          c.align === "right" ? "text-right" : c.align === "center" ? "text-center" : "text-left",
                          c.className,
                        )}
                      >
                        {c.render ? c.render(v, row) : defaultFormat(v, c.decimals)}
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {truncated && (
        <div className="px-2 py-1 text-[10px] text-text-faint border-t border-line-soft bg-bg2/50">
          {rows.length.toLocaleString()}행 중 처음 {maxRows}행만 표시
        </div>
      )}
    </div>
  );
}
