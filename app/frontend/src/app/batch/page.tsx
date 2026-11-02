"use client";

// 배치 분석 (/batch) — 폴더(서버/업로드) → job → 진행률 → 결과(다운로드/Summary/갤러리).

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import clsx from "clsx";
import {
  ChevronLeft,
  Copy,
  FileSpreadsheet,
  FolderOpen,
  FolderSearch,
  HardDrive,
  Loader2,
  Package,
  Play,
  Presentation,
  RefreshCw,
  Square,
  X,
} from "lucide-react";

import type { BatchJob, BatchJobSummary, BatchResult, FsList, SummaryRow } from "@/lib/types";
import { basename, cancelJob, createBatchJob, fsDrives, fsList, getJob, jobFileUrl, jobZipUrl, listJobs, type InputMode } from "@/lib/api";
import { useAnalysisParams } from "@/lib/params";
import ParamsPanel from "@/components/params/ParamsPanel";
import EngineSwitch from "@/components/ui/EngineSwitch";
import Dropzone, { isImageFile } from "@/components/ui/Dropzone";
import Chip from "@/components/ui/Chip";
import ProgressBar from "@/components/ui/ProgressBar";
import DataTable, { type DataTableColumn } from "@/components/ui/DataTable";
import { Lightbox } from "@/components/ui/ImageViewer";

const POLL_MS = 700;
const EXCEL_NAME = "particle_analysis_result.xlsx";
const PPTX_NAME = "particle_analysis_result.pptx";

// ───────────────────────── 갤러리 카테고리 ─────────────────────────

interface GalleryItem {
  relpath: string;
  label: string;
}

const CATEGORY_DEFS: { id: string; label: string; imageKeys?: string[]; sampleKeys?: string[]; extraKeys?: string[] }[] = [
  { id: "labeled_before", label: "라벨 Before", imageKeys: ["labeled_before"] },
  { id: "labeled_after", label: "라벨 After", imageKeys: ["labeled_after"] },
  { id: "comparison", label: "비교", imageKeys: ["comparison"] },
  { id: "contour", label: "윤곽선", imageKeys: ["contour_before", "contour_after"] },
  { id: "hist", label: "히스토그램", sampleKeys: ["hist_diameter", "hist_circularity", "hist_aspect"] },
  { id: "summary", label: "요약카드", sampleKeys: ["summary_before", "summary_after"] },
  { id: "boundary", label: "경계", imageKeys: ["boundary_before", "boundary_after"] },
  { id: "aggl", label: "응집도", imageKeys: ["aggl_before", "aggl_after"] },
  { id: "dbscan", label: "DBSCAN", imageKeys: ["dbscan_before", "dbscan_after"] },
  { id: "normal_dist", label: "정규분포", extraKeys: ["normal_dist"] },
];

function buildGallery(r: BatchResult): { id: string; label: string; items: GalleryItem[] }[] {
  const out: { id: string; label: string; items: GalleryItem[] }[] = [];
  const usedImageKeys = new Set<string>();
  const usedSampleKeys = new Set<string>();
  const usedExtraKeys = new Set<string>();

  for (const def of CATEGORY_DEFS) {
    const items: GalleryItem[] = [];
    for (const k of def.imageKeys ?? []) {
      usedImageKeys.add(k);
      for (const im of r.images ?? []) {
        const rel = im.files?.[k];
        if (rel) items.push({ relpath: rel, label: `${im.filename} · ${k}` });
      }
    }
    for (const k of def.sampleKeys ?? []) {
      usedSampleKeys.add(k);
      for (const [sample, files] of Object.entries(r.sample_files ?? {})) {
        const rel = files?.[k];
        if (rel) items.push({ relpath: rel, label: `${sample} · ${k}` });
      }
    }
    for (const k of def.extraKeys ?? []) {
      usedExtraKeys.add(k);
      for (const rel of r.extra_files?.[k] ?? []) items.push({ relpath: rel, label: `${k} · ${basename(rel)}` });
    }
    if (items.length) out.push({ id: def.id, label: def.label, items });
  }

  // 정의되지 않은 키는 그대로 탭으로
  const extraImageKeys = new Set<string>();
  for (const im of r.images ?? []) for (const k of Object.keys(im.files ?? {})) if (!usedImageKeys.has(k)) extraImageKeys.add(k);
  for (const k of extraImageKeys) {
    const items = (r.images ?? []).flatMap((im) => (im.files?.[k] ? [{ relpath: im.files[k], label: `${im.filename} · ${k}` }] : []));
    if (items.length) out.push({ id: `img:${k}`, label: k, items });
  }
  const extraSampleKeys = new Set<string>();
  for (const files of Object.values(r.sample_files ?? {})) for (const k of Object.keys(files ?? {})) if (!usedSampleKeys.has(k)) extraSampleKeys.add(k);
  for (const k of extraSampleKeys) {
    const items = Object.entries(r.sample_files ?? {}).flatMap(([s, files]) => (files?.[k] ? [{ relpath: files[k], label: `${s} · ${k}` }] : []));
    if (items.length) out.push({ id: `smp:${k}`, label: k, items });
  }
  for (const [k, rels] of Object.entries(r.extra_files ?? {})) {
    if (usedExtraKeys.has(k) || !rels?.length) continue;
    out.push({ id: `ext:${k}`, label: k, items: rels.map((rel) => ({ relpath: rel, label: `${k} · ${basename(rel)}` })) });
  }
  return out;
}

// ───────────────────────── Summary 표 ─────────────────────────

// 백엔드 Summary 시트 컬럼: Sample, Total_Particles_{Before,After}, D10/D50/D90/Span/Mean/Std/Mode/CV_{Before,After}
const PREFERRED_SUMMARY_COLS: Array<[string, string]> = [
  ["Sample", "샘플"],
  ["Total_Particles_Before", "입자수 (Before)"],
  ["Total_Particles_After", "입자수 (After)"],
  ["D10_After", "D10"],
  ["D50_After", "D50"],
  ["D90_After", "D90"],
  ["Span_After", "Span"],
  ["Mean_After", "Mean"],
  ["Std_After", "Std"],
  ["CV_After", "CV (%)"],
];

function summaryColumns(rows: SummaryRow[]): DataTableColumn<SummaryRow>[] {
  if (!rows.length) return [];
  const keys = Object.keys(rows[0]);
  const lower = new Map(keys.map((k) => [k.toLowerCase().replace(/[\s_()%]/g, ""), k]));
  const picked: Array<[string, string]> = [];
  for (const [want, label] of PREFERRED_SUMMARY_COLS) {
    const k = lower.get(want.toLowerCase().replace(/[\s_()%]/g, ""));
    if (k && !picked.some(([pk]) => pk === k)) picked.push([k, label]);
  }
  // 선호 컬럼이 거의 없으면 전체 표시
  const cols: Array<[string, string]> = picked.length >= 3 ? picked : keys.map((k) => [k, k]);
  return cols.map(([k, label]) => ({
    key: k,
    label,
    align: typeof rows[0][k] === "number" ? "right" : "left",
    render: (v) => {
      if (typeof v === "number") return Number.isInteger(v) ? String(v) : v.toFixed(3);
      if (v === null || v === undefined || v === "") return <span className="text-text-faint">–</span>;
      return String(v);
    },
  }));
}

// ───────────────────────── 폴더 탐색 모달 ─────────────────────────

function FolderBrowser({ initial, onSelect, onClose }: { initial: string; onSelect: (path: string) => void; onClose: () => void }) {
  const [drives, setDrives] = useState<string[]>([]);
  const [listing, setListing] = useState<FsList | null>(null);
  const [path, setPath] = useState(initial);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const load = useCallback(async (p: string) => {
    setBusy(true);
    setErr(null);
    try {
      const r = await fsList(p);
      setListing(r);
      setPath(r.path);
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  }, []);

  useEffect(() => {
    fsDrives()
      .then((d) => {
        setDrives(d);
        load(initial || d[0] || "");
      })
      .catch((e) => {
        setErr((e as Error).message);
        if (initial) load(initial);
      });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  if (typeof document === "undefined") return null;
  return createPortal(
    <div className="fixed inset-0 z-[90] bg-black/40 backdrop-blur-[2px] flex items-center justify-center p-4 fade-in" onClick={onClose}>
      <div className="app-card w-full max-w-[720px] max-h-[80vh] flex flex-col shadow-lg" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center justify-between px-4 py-2.5 border-b border-line">
          <div className="text-[13px] font-semibold flex items-center gap-2">
            <FolderSearch size={15} /> 서버 폴더 탐색
          </div>
          <button type="button" className="w-7 h-7 rounded-md hover:bg-bg3 inline-flex items-center justify-center text-text-mute" onClick={onClose}>
            <X size={14} />
          </button>
        </div>
        <div className="px-4 py-2 flex items-center gap-2 border-b border-line-soft">
          {drives.map((d) => (
            <Chip key={d} active={listing?.path?.toLowerCase().startsWith(d.toLowerCase())} onClick={() => load(d)}>
              <HardDrive size={11} /> {d}
            </Chip>
          ))}
          <div className="flex-1 flex items-center gap-1">
            <input
              type="text"
              className="h-7 flex-1 rounded-md border px-2 text-[11px] font-mono"
              value={path}
              onChange={(e) => setPath(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && load(path)}
            />
            <button type="button" className="h-7 px-2 rounded-md border border-line text-[11px] hover:border-line-strong" onClick={() => load(path)}>
              이동
            </button>
          </div>
        </div>
        <div className="flex-1 min-h-0 grid grid-cols-[1fr_220px]">
          <div className="overflow-y-auto border-r border-line-soft">
            {listing?.parent && (
              <button type="button" className="w-full text-left px-4 py-1.5 text-[12px] hover:bg-bg2 flex items-center gap-2 toss-row" onClick={() => load(listing.parent as string)}>
                <ChevronLeft size={13} className="text-text-faint" /> ..
              </button>
            )}
            {busy && (
              <div className="px-4 py-3 text-[11px] text-text-faint flex items-center gap-2">
                <Loader2 size={12} className="animate-spin" /> 불러오는 중…
              </div>
            )}
            {!busy && listing?.dirs.length === 0 && <div className="px-4 py-3 text-[11px] text-text-faint">하위 폴더 없음</div>}
            {!busy &&
              listing?.dirs.map((d) => (
                <button key={d.path} type="button" className="w-full text-left px-4 py-1.5 text-[12px] hover:bg-bg2 flex items-center gap-2 toss-row" onClick={() => load(d.path)} onDoubleClick={() => onSelect(d.path)}>
                  <FolderOpen size={13} className="text-amber flex-shrink-0" />
                  <span className="truncate">{d.name}</span>
                </button>
              ))}
            {err && <div className="px-4 py-2 text-[11px] text-bad">{err}</div>}
          </div>
          <div className="overflow-y-auto p-3 text-[11px]">
            <div className="text-text-mute mb-1">
              이미지 <span className="font-semibold text-text font-mono">{listing?.n_images ?? 0}</span>장
            </div>
            <ul className="space-y-0.5 text-[10px] font-mono text-text-faint">
              {(listing?.images ?? []).map((n) => (
                <li key={n} className="truncate" title={n}>
                  {n}
                </li>
              ))}
              {listing && listing.n_images > listing.images.length && <li>… 외 {listing.n_images - listing.images.length}장</li>}
            </ul>
          </div>
        </div>
        <div className="px-4 py-2.5 border-t border-line flex items-center justify-between gap-2">
          <div className="text-[10px] text-text-faint font-mono truncate">{listing?.path}</div>
          <div className="flex gap-2">
            <button type="button" className="h-8 px-3 rounded-md border border-line text-[12px] hover:border-line-strong" onClick={onClose}>
              취소
            </button>
            <button type="button" className="btn-primary h-8 px-3 text-[12px]" disabled={!listing} onClick={() => listing && onSelect(listing.path)}>
              이 폴더 선택 {listing ? `(${listing.n_images}장)` : ""}
            </button>
          </div>
        </div>
      </div>
    </div>,
    document.body,
  );
}

// ───────────────────────── 페이지 ─────────────────────────

function statusTone(s: string): "good" | "bad" | "ink" | "mute" | "amber" {
  switch (s) {
    case "done":
      return "good";
    case "error":
      return "bad";
    case "running":
      return "ink";
    case "cancelled":
      return "amber";
    default:
      return "mute";
  }
}
const STATUS_KO: Record<string, string> = { queued: "대기", running: "실행 중", done: "완료", error: "오류", cancelled: "취소됨" };

function fmtBytes(n: number) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} KB`;
  if (n < 1024 * 1024 * 1024) return `${(n / 1024 / 1024).toFixed(1)} MB`;
  return `${(n / 1024 / 1024 / 1024).toFixed(2)} GB`;
}

function fmtTime(v: string | number | null | undefined) {
  if (v === null || v === undefined || v === "") return "";
  const d = typeof v === "number" ? new Date(v < 1e12 ? v * 1000 : v) : new Date(v);
  return Number.isNaN(d.getTime()) ? String(v) : d.toLocaleString("ko-KR", { hour12: false });
}

export default function BatchPage() {
  const { params, setParams, loaded } = useAnalysisParams();

  const [inputMode, setInputMode] = useState<InputMode>("server_path");
  const [inputPath, setInputPath] = useState("");
  const [pathInfo, setPathInfo] = useState<{ n: number; err: string | null } | null>(null);
  const [browserOpen, setBrowserOpen] = useState(false);
  const [files, setFiles] = useState<File[]>([]);
  const [outputDir, setOutputDir] = useState("");
  const [jobName, setJobName] = useState("");

  const [jobId, setJobId] = useState<string | null>(null);
  const [job, setJob] = useState<BatchJob | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [jobs, setJobs] = useState<BatchJobSummary[]>([]);
  const [copied, setCopied] = useState(false);

  const [gallerySel, setGallerySel] = useState<string | null>(null);
  const [lightboxIdx, setLightboxIdx] = useState<number | null>(null);
  const logRef = useRef<HTMLDivElement>(null);

  // 서버 경로 → 이미지 수 (디바운스)
  useEffect(() => {
    if (inputMode !== "server_path" || !inputPath.trim()) {
      setPathInfo(null);
      return;
    }
    const t = setTimeout(async () => {
      try {
        const r = await fsList(inputPath.trim());
        setPathInfo({ n: r.n_images, err: null });
      } catch (e) {
        setPathInfo({ n: 0, err: (e as Error).message });
      }
    }, 400);
    return () => clearTimeout(t);
  }, [inputPath, inputMode]);

  const refreshJobs = useCallback(async () => {
    try {
      const list = await listJobs();
      setJobs([...list].sort((a, b) => String(b.created_at ?? "").localeCompare(String(a.created_at ?? ""))));
    } catch {
      /* 백엔드 미기동 등 — 무시 */
    }
  }, []);

  useEffect(() => {
    refreshJobs();
  }, [refreshJobs]);

  // 폴링 (700ms, 종료 상태에서 중단)
  useEffect(() => {
    if (!jobId) return;
    let alive = true;
    let timer: ReturnType<typeof setTimeout> | null = null;
    const tick = async () => {
      try {
        const j = await getJob(jobId);
        if (!alive) return;
        setJob(j);
        if (j.status === "done" || j.status === "error" || j.status === "cancelled") {
          refreshJobs();
          return;
        }
      } catch (e) {
        if (!alive) return;
        setError(`job 조회 실패: ${(e as Error).message}`);
        return;
      }
      if (alive) timer = setTimeout(tick, POLL_MS);
    };
    tick();
    return () => {
      alive = false;
      if (timer) clearTimeout(timer);
    };
  }, [jobId, refreshJobs]);

  // 로그 자동 스크롤
  useEffect(() => {
    const el = logRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [job?.log?.length]);

  const imageFiles = useMemo(() => files.filter(isImageFile), [files]);
  const totalBytes = useMemo(() => imageFiles.reduce((s, f) => s + f.size, 0), [imageFiles]);

  const canStart =
    loaded && !submitting && (inputMode === "server_path" ? inputPath.trim().length > 0 && !(pathInfo?.err) : imageFiles.length > 0) && job?.status !== "running" && job?.status !== "queued";

  const start = async () => {
    if (!canStart) return;
    setSubmitting(true);
    setError(null);
    setJob(null);
    setGallerySel(null);
    try {
      const r = await createBatchJob({
        params,
        inputMode,
        inputPath: inputPath.trim(),
        outputDir: outputDir.trim(),
        jobName: jobName.trim(),
        files: imageFiles,
      });
      setJobId(r.job_id);
      refreshJobs();
    } catch (e) {
      setError(`job 생성 실패: ${(e as Error).message}`);
    } finally {
      setSubmitting(false);
    }
  };

  const cancel = async () => {
    if (!jobId) return;
    try {
      await cancelJob(jobId);
    } catch (e) {
      setError(`취소 실패: ${(e as Error).message}`);
    }
  };

  const openJob = (id: string) => {
    setError(null);
    setGallerySel(null);
    setJob(null);
    setJobId(id);
  };

  const result = job?.result ?? null;
  const gallery = useMemo(() => (result ? buildGallery(result) : []), [result]);
  const activeCat = gallery.find((g) => g.id === gallerySel) ?? gallery[0];
  const summaryCols = useMemo(() => summaryColumns(result?.samples ?? []), [result]);

  const excelRel = result?.excel_path ? basename(result.excel_path) || EXCEL_NAME : EXCEL_NAME;
  const pptxRel = result?.pptx_path ? basename(result.pptx_path) || PPTX_NAME : null;

  const copyOutput = async () => {
    const dir = result?.output_dir ?? job?.output_dir;
    if (!dir) return;
    try {
      await navigator.clipboard.writeText(dir);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      window.prompt("출력 폴더 경로", dir);
    }
  };

  const running = job?.status === "running" || job?.status === "queued";

  return (
    <main className="mx-auto max-w-[1800px] px-4 py-3">
      <div className="grid gap-3" style={{ gridTemplateColumns: "320px minmax(0,1fr) minmax(360px, 420px)" }}>
        {/* ── 좌: 입력 ── */}
        <aside className="flex flex-col gap-3">
          <section className="app-card p-3 flex flex-col gap-2.5">
            <div className="text-[12px] font-semibold text-text">입력</div>
            <div className="inline-flex w-full rounded-lg border border-line bg-bg2 p-0.5">
              {(
                [
                  { v: "server_path", l: "서버 폴더" },
                  { v: "upload", l: "업로드" },
                ] as { v: InputMode; l: string }[]
              ).map((o) => (
                <button
                  key={o.v}
                  type="button"
                  className={clsx("flex-1 h-7 rounded-md text-[11px] font-semibold", inputMode === o.v ? "bg-[#191919] text-white" : "text-text-mute hover:text-text")}
                  onClick={() => setInputMode(o.v)}
                >
                  {o.l}
                </button>
              ))}
            </div>

            {inputMode === "server_path" ? (
              <>
                <div className="flex gap-1">
                  <input
                    type="text"
                    className="h-8 flex-1 rounded-md border px-2 text-[11px] font-mono min-w-0"
                    placeholder="C:\data\sem_images"
                    value={inputPath}
                    onChange={(e) => setInputPath(e.target.value)}
                  />
                  <button type="button" className="h-8 px-2.5 rounded-md border border-line text-[11px] hover:border-line-strong inline-flex items-center gap-1" onClick={() => setBrowserOpen(true)} title="폴더 탐색">
                    <FolderSearch size={13} />
                  </button>
                </div>
                <div className="text-[10px] text-text-faint min-h-[14px]">
                  {pathInfo?.err ? (
                    <span className="text-bad">{pathInfo.err}</span>
                  ) : pathInfo ? (
                    <>
                      이미지 <span className="font-mono font-semibold text-text">{pathInfo.n}</span>장 (비재귀)
                    </>
                  ) : (
                    "서버에서 접근 가능한 절대경로"
                  )}
                </div>
              </>
            ) : (
              <>
                <Dropzone directory onFiles={(fs) => setFiles(fs)} filled={files.length > 0} compact={files.length > 0} accept=".jpg,.jpeg,.png,.tif,.tiff,.bmp,image/*">
                  {files.length > 0 ? (
                    <div className="w-full text-left flex items-center justify-between gap-2">
                      <div>
                        <div className="text-[12px] font-medium text-text">
                          이미지 <span className="font-mono">{imageFiles.length}</span>장
                          {files.length !== imageFiles.length && <span className="text-text-faint"> (총 {files.length}개 파일)</span>}
                        </div>
                        <div className="text-[10px] text-text-faint">{fmtBytes(totalBytes)} · 클릭하여 다른 폴더 선택</div>
                      </div>
                      <button
                        type="button"
                        className="w-6 h-6 rounded-md hover:bg-bg3 inline-flex items-center justify-center text-text-faint"
                        onClick={(e) => {
                          e.stopPropagation();
                          setFiles([]);
                        }}
                      >
                        <X size={12} />
                      </button>
                    </div>
                  ) : undefined}
                </Dropzone>
                {imageFiles.length > 0 && (
                  <div className="max-h-24 overflow-y-auto text-[10px] font-mono text-text-faint px-1 space-y-0.5">
                    {imageFiles.slice(0, 30).map((f) => (
                      <div key={f.name + f.size} className="truncate">
                        {f.name}
                      </div>
                    ))}
                    {imageFiles.length > 30 && <div>… 외 {imageFiles.length - 30}장</div>}
                  </div>
                )}
              </>
            )}
          </section>

          <section className="app-card p-3 flex flex-col gap-2">
            <label className="flex flex-col gap-1">
              <span className="text-[11px] text-text-mute">출력 폴더 (선택)</span>
              <input type="text" className="h-8 rounded-md border px-2 text-[11px] font-mono" placeholder="비우면 runs/<일시>_<엔진>" value={outputDir} onChange={(e) => setOutputDir(e.target.value)} />
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-[11px] text-text-mute">job 이름 (선택)</span>
              <input type="text" className="h-8 rounded-md border px-2 text-[11px]" placeholder="예: 샘플A 1차" value={jobName} onChange={(e) => setJobName(e.target.value)} />
            </label>
          </section>

          <section className="app-card p-3 flex flex-col gap-2">
            <div className="text-[12px] font-semibold text-text">분할 엔진</div>
            <EngineSwitch value={params.engine} onChange={(v) => setParams({ ...params, engine: v })} />
          </section>

          <div className="flex gap-2">
            <button type="button" className="btn-primary flex-1 h-10 text-[13px] inline-flex items-center justify-center gap-2" disabled={!canStart} onClick={start}>
              {submitting ? <Loader2 size={14} className="animate-spin" /> : <Play size={14} />}
              {submitting ? "업로드 중…" : "분석 시작"}
            </button>
            {running && (
              <button type="button" className="h-10 px-3 rounded-md border border-bad/40 text-bad text-[12px] hover:bg-bad-soft inline-flex items-center gap-1" onClick={cancel}>
                <Square size={12} /> 취소
              </button>
            )}
          </div>

          {error && <div className="app-card p-2.5 border-bad/30 bg-bad-soft text-[11px] text-bad break-words">{error}</div>}

          {/* 최근 job */}
          <section className="app-card">
            <div className="flex items-center justify-between px-3 py-2 border-b border-line-soft">
              <span className="text-[12px] font-semibold">최근 job</span>
              <button type="button" className="w-6 h-6 rounded-md hover:bg-bg3 inline-flex items-center justify-center text-text-faint" title="새로고침" onClick={refreshJobs}>
                <RefreshCw size={12} />
              </button>
            </div>
            <div className="max-h-[260px] overflow-y-auto">
              {jobs.length === 0 && <div className="px-3 py-3 text-[11px] text-text-faint">아직 job 이 없습니다 (서버 재시작 시 초기화)</div>}
              {jobs.map((j) => (
                <button
                  key={j.job_id}
                  type="button"
                  className={clsx("w-full text-left px-3 py-2 toss-row hover:bg-bg2 flex items-center gap-2", j.job_id === jobId && "bg-ink-soft")}
                  onClick={() => openJob(j.job_id)}
                >
                  <div className="min-w-0 flex-1">
                    <div className="text-[11px] font-medium truncate">{j.name || j.job_id}</div>
                    <div className="text-[10px] text-text-faint truncate font-mono">
                      {j.n_images}장 · {fmtTime(j.created_at)}
                    </div>
                  </div>
                  <Chip tone={statusTone(j.status)} className="!py-0 !px-1.5 !text-[9px]">
                    {STATUS_KO[j.status] ?? j.status}
                    {j.status === "running" ? ` ${Math.round(j.progress)}%` : ""}
                  </Chip>
                </button>
              ))}
            </div>
          </section>
        </aside>

        {/* ── 중앙: 진행/결과 ── */}
        <section className="flex flex-col gap-3 min-w-0">
          {!job && !jobId && (
            <div className="app-card p-8 text-center text-[12px] text-text-faint">
              입력 폴더와 파라미터를 설정하고 <span className="font-semibold text-text">분석 시작</span>을 누르세요. 결과는 이 영역에 표시됩니다.
            </div>
          )}

          {jobId && !job && (
            <div className="app-card p-6 text-[12px] text-text-faint flex items-center gap-2">
              <Loader2 size={13} className="animate-spin" /> job 불러오는 중…
            </div>
          )}

          {job && (
            <div className="app-card p-3 flex flex-col gap-2">
              <div className="flex items-center gap-2">
                <Chip tone={statusTone(job.status)} dot>
                  {STATUS_KO[job.status] ?? job.status}
                </Chip>
                <span className="text-[12px] font-semibold truncate">{job.name || job.job_id}</span>
                <span className="text-[10px] text-text-faint font-mono ml-auto">{job.job_id}</span>
              </div>
              <ProgressBar value={job.progress} indeterminate={job.status === "queued"} tone={job.status === "error" ? "bad" : job.status === "done" ? "good" : "dark"} />
              <div className="text-[11px] text-text-mute truncate">{job.message}</div>
              {job.error && <div className="text-[11px] text-bad whitespace-pre-wrap break-words bg-bad-soft rounded-md p-2">{job.error}</div>}
              {job.log?.length > 0 && (
                <details open={running} className="text-[10px]">
                  <summary className="cursor-pointer text-text-faint select-none">로그 ({job.log.length})</summary>
                  <div ref={logRef} className="mt-1 max-h-40 overflow-y-auto font-mono text-text-mute bg-bg2 rounded-md p-2 space-y-0.5">
                    {job.log.map((l, i) => (
                      <div key={i} className="whitespace-pre-wrap break-all">
                        {l}
                      </div>
                    ))}
                  </div>
                </details>
              )}
            </div>
          )}

          {result && jobId && (
            <>
              {/* 다운로드 바 */}
              <div className="app-card p-2.5 flex items-center gap-2 flex-wrap">
                <a href={jobFileUrl(jobId, excelRel)} className="btn-primary h-8 px-3 text-[12px] inline-flex items-center gap-1.5" download>
                  <FileSpreadsheet size={13} /> Excel
                </a>
                {pptxRel ? (
                  <a href={jobFileUrl(jobId, pptxRel)} className="h-8 px-3 rounded-md border border-line text-[12px] font-medium hover:border-line-strong inline-flex items-center gap-1.5" download>
                    <Presentation size={13} /> PPT
                  </a>
                ) : (
                  <span className="h-8 px-3 rounded-md border border-line text-[12px] text-text-faint inline-flex items-center gap-1.5 opacity-60" title="PPT 미생성">
                    <Presentation size={13} /> PPT 없음
                  </span>
                )}
                <a href={jobZipUrl(jobId)} className="h-8 px-3 rounded-md border border-line text-[12px] font-medium hover:border-line-strong inline-flex items-center gap-1.5" download>
                  <Package size={13} /> 전체 zip
                </a>
                <button type="button" className="h-8 px-3 rounded-md border border-line text-[12px] font-medium hover:border-line-strong inline-flex items-center gap-1.5" onClick={copyOutput}>
                  <Copy size={13} /> {copied ? "복사됨" : "출력 폴더 경로 복사"}
                </button>
                <span className="text-[10px] font-mono text-text-faint truncate ml-auto" title={result.output_dir}>
                  {result.output_dir}
                </span>
              </div>

              {/* Summary */}
              <div className="flex flex-col gap-1.5">
                <div className="text-[12px] font-semibold px-0.5">
                  샘플 Summary <span className="text-text-faint font-normal">({result.samples?.length ?? 0})</span>
                </div>
                <DataTable<SummaryRow> rows={result.samples ?? []} columns={summaryCols} rowKey={(r, i) => `${String(r[summaryCols[0]?.key ?? ""] ?? "")}-${i}`} maxHeight={300} dense emptyText="Summary 행이 없습니다" />
              </div>

              {/* 갤러리 */}
              <div className="flex flex-col gap-1.5">
                <div className="flex items-center gap-1.5 flex-wrap">
                  <span className="text-[12px] font-semibold px-0.5 mr-1">결과 갤러리</span>
                  {gallery.map((g) => (
                    <Chip key={g.id} active={activeCat?.id === g.id} onClick={() => setGallerySel(g.id)}>
                      {g.label} <span className="font-mono text-[9px] opacity-70">{g.items.length}</span>
                    </Chip>
                  ))}
                  {gallery.length === 0 && <span className="text-[11px] text-text-faint">출력 이미지가 없습니다</span>}
                </div>
                {activeCat && (
                  <div className="app-card p-2 grid gap-2" style={{ gridTemplateColumns: "repeat(auto-fill, minmax(150px, 1fr))" }}>
                    {activeCat.items.map((it, i) => (
                      <button key={it.relpath} type="button" className="group text-left flex flex-col gap-1" onClick={() => setLightboxIdx(i)} title={it.label}>
                        <div className="aspect-[4/3] w-full rounded-md overflow-hidden border border-line bg-black/90 group-hover:border-ink">
                          {/* eslint-disable-next-line @next/next/no-img-element */}
                          <img src={jobFileUrl(jobId, it.relpath)} alt={it.label} loading="lazy" className="w-full h-full object-contain" />
                        </div>
                        <div className="text-[10px] text-text-mute truncate">{it.label}</div>
                      </button>
                    ))}
                  </div>
                )}
              </div>

              {/* 경고 */}
              {result.warnings?.length > 0 && (
                <div className="app-card p-3 border-amber/30 bg-amber-soft">
                  <div className="text-[12px] font-semibold text-amber mb-1">경고 ({result.warnings.length})</div>
                  <ul className="text-[11px] text-text-mute space-y-0.5 max-h-40 overflow-y-auto">
                    {result.warnings.map((w, i) => (
                      <li key={i} className="whitespace-pre-wrap break-words">
                        · {w}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </>
          )}
        </section>

        {/* ── 우: 파라미터 ── */}
        <aside className={clsx("min-w-0 overflow-y-auto pr-0.5", !loaded && "opacity-50")} style={{ maxHeight: "calc(100vh - 72px)" }}>
          <ParamsPanel params={params} onChange={setParams} mode="batch" disabled={running} />
        </aside>
      </div>

      {browserOpen && (
        <FolderBrowser
          initial={inputPath}
          onClose={() => setBrowserOpen(false)}
          onSelect={(p) => {
            setInputPath(p);
            setBrowserOpen(false);
          }}
        />
      )}

      {activeCat && lightboxIdx !== null && jobId && (
        <Lightbox
          src={jobFileUrl(jobId, activeCat.items[lightboxIdx].relpath)}
          label={activeCat.items[lightboxIdx].label}
          counter={`${lightboxIdx + 1} / ${activeCat.items.length}`}
          onClose={() => setLightboxIdx(null)}
          onPrev={lightboxIdx > 0 ? () => setLightboxIdx(lightboxIdx - 1) : undefined}
          onNext={lightboxIdx < activeCat.items.length - 1 ? () => setLightboxIdx(lightboxIdx + 1) : undefined}
        />
      )}
    </main>
  );
}
