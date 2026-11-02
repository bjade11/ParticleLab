"use client";

// 튜너 (/) — 단일 이미지로 파라미터 조정 + 미리보기.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import clsx from "clsx";
import { ArrowRight, Download, Loader2, Play, Upload, X } from "lucide-react";

import type { AnalysisParams, DValues, ParticleRow, TunerSegmentResponse, TunerViewKey } from "@/lib/types";
import { tunerPreprocess, tunerSegment } from "@/lib/api";
import { downloadParamsJson, importParamsJson, saveParams, useAnalysisParams } from "@/lib/params";
import ParamsPanel from "@/components/params/ParamsPanel";
import ImageViewer from "@/components/ui/ImageViewer";
import EngineSwitch from "@/components/ui/EngineSwitch";
import Dropzone, { isImageFile } from "@/components/ui/Dropzone";
import Chip from "@/components/ui/Chip";
import DataTable, { type DataTableColumn } from "@/components/ui/DataTable";
import SelectField from "@/components/ui/SelectField";
import NumberField from "@/components/ui/NumberField";

const VIEWS: { key: TunerViewKey; label: string; wsOnly?: boolean }[] = [
  { key: "original", label: "원본" },
  { key: "preprocessed", label: "전처리" },
  { key: "binary", label: "이진화", wsOnly: true },
  { key: "distance", label: "거리맵", wsOnly: true },
  { key: "markers", label: "마커", wsOnly: true },
  { key: "labels", label: "라벨" },
  { key: "labeled_before", label: "Before 라벨" },
  { key: "labeled_after", label: "After 라벨" },
  { key: "contour_before", label: "윤곽선" },
];

const PARTICLE_COLS: DataTableColumn<ParticleRow>[] = [
  { key: "Label", label: "Label", align: "right", width: 56 },
  { key: "Diameter (μm)", label: "직경 (μm)", align: "right", decimals: 3 },
  { key: "Circularity", label: "원형도", align: "right", decimals: 3 },
  { key: "Aspect Ratio", label: "종횡비", align: "right", decimals: 3 },
  { key: "Area (μm^2)", label: "면적 (μm²)", align: "right", decimals: 3 },
  { key: "Solidity", label: "Solidity", align: "right", decimals: 3 },
  { key: "Mean Intensity", label: "평균 밝기", align: "right", decimals: 1 },
  { key: "x", label: "x", align: "right", decimals: 1 },
  { key: "y", label: "y", align: "right", decimals: 1 },
];

function fmt(v: number | undefined | null, d = 3) {
  return typeof v === "number" && Number.isFinite(v) ? v.toFixed(d) : "–";
}

function DStat({ label, before, after, d = 3 }: { label: string; before?: number; after?: number; d?: number }) {
  return (
    <div className="flex flex-col items-start px-3 py-1.5 border-r border-line-soft last:border-r-0 min-w-[92px]">
      <span className="text-[10px] text-text-faint">{label}</span>
      <span className="text-[13px] font-semibold font-mono tabular-nums text-text">{fmt(after, d)}</span>
      <span className="text-[10px] font-mono tabular-nums text-text-faint">before {fmt(before, d)}</span>
    </div>
  );
}

export default function TunerPage() {
  const router = useRouter();
  const { params, setParams, loaded } = useAnalysisParams();

  const [file, setFile] = useState<File | null>(null);
  const [fileUrl, setFileUrl] = useState<string | null>(null);
  const [result, setResult] = useState<TunerSegmentResponse | null>(null);
  const [previewPng, setPreviewPng] = useState<string | null>(null);
  const [view, setView] = useState<TunerViewKey>("original");
  const [segBusy, setSegBusy] = useState(false);
  const [preBusy, setPreBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [tableSide, setTableSide] = useState<"before" | "after">("after");
  const [notice, setNotice] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const preTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const preAbort = useRef<AbortController | null>(null);
  const segAbort = useRef<AbortController | null>(null);

  // 파일 → object URL
  useEffect(() => {
    if (!file) {
      setFileUrl(null);
      return;
    }
    const u = URL.createObjectURL(file);
    setFileUrl(u);
    return () => URL.revokeObjectURL(u);
  }, [file]);

  const onFiles = (fs: File[]) => {
    const f = fs.find(isImageFile) ?? fs[0];
    if (!f) return;
    setFile(f);
    setResult(null);
    setPreviewPng(null);
    setError(null);
    setView("original");
  };

  // ── 실시간 전처리 미리보기 (300ms 디바운스)
  const requestPreview = useCallback(
    (p: AnalysisParams, f: File | null) => {
      if (!f) return;
      if (preTimer.current) clearTimeout(preTimer.current);
      preTimer.current = setTimeout(async () => {
        preAbort.current?.abort();
        const ctrl = new AbortController();
        preAbort.current = ctrl;
        setPreBusy(true);
        try {
          const r = await tunerPreprocess(f, p, ctrl.signal);
          if (ctrl.signal.aborted) return;
          setPreviewPng(r.preview_png);
          setView("preprocessed");
          setError(null);
        } catch (e) {
          if ((e as Error).name === "AbortError") return;
          setError(`전처리 미리보기 실패: ${(e as Error).message}`);
        } finally {
          if (!ctrl.signal.aborted) setPreBusy(false);
        }
      }, 300);
    },
    [],
  );

  // 파일이 새로 올라오면 1회 미리보기
  useEffect(() => {
    if (file && loaded) requestPreview(params, file);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [file, loaded]);

  useEffect(
    () => () => {
      if (preTimer.current) clearTimeout(preTimer.current);
      preAbort.current?.abort();
      segAbort.current?.abort();
    },
    [],
  );

  const onSemChange = useCallback((next: AnalysisParams) => requestPreview(next, file), [requestPreview, file]);

  // ── 분할 실행
  const runSegment = async () => {
    if (!file || segBusy) return;
    segAbort.current?.abort();
    const ctrl = new AbortController();
    segAbort.current = ctrl;
    setSegBusy(true);
    setError(null);
    try {
      const r = await tunerSegment(file, params, ctrl.signal);
      if (ctrl.signal.aborted) return;
      setResult(r);
      setView("labeled_after");
    } catch (e) {
      if ((e as Error).name === "AbortError") return;
      setError(`분할 실패: ${(e as Error).message}`);
    } finally {
      if (!ctrl.signal.aborted) setSegBusy(false);
    }
  };

  const cancelSegment = () => {
    segAbort.current?.abort();
    setSegBusy(false);
  };

  // ── JSON 내보내기 / 불러오기
  const onImportFile = async (f: File) => {
    try {
      const text = await f.text();
      const { params: p, source } = importParamsJson(text, params);
      setParams(p);
      setNotice(source === "legacy" ? "레거시 데스크톱 설정을 변환해 불러왔습니다." : "파라미터를 불러왔습니다.");
      if (file) requestPreview(p, file);
      setTimeout(() => setNotice(null), 3500);
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const goBatch = () => {
    saveParams(params);
    router.push("/batch");
  };

  // ── 현재 뷰 이미지
  const isWs = params.engine === "watershed";
  const viewSrc = useMemo((): string | null => {
    const v = result?.views;
    switch (view) {
      case "original":
        return fileUrl ?? v?.original ?? null;
      case "preprocessed":
        return previewPng ?? v?.preprocessed ?? null;
      default:
        return v?.[view] ?? null;
    }
  }, [view, fileUrl, previewPng, result]);

  const viewAvailable = (k: TunerViewKey): boolean => {
    if (k === "original") return !!fileUrl;
    if (k === "preprocessed") return !!previewPng || !!result?.views.preprocessed;
    return !!result?.views[k];
  };

  const dBefore: DValues | undefined = result?.d_values_before;
  const dAfter: DValues | undefined = result?.d_values_after;
  const rows = tableSide === "after" ? result?.particles_after ?? [] : result?.particles_before ?? [];

  return (
    <main className="mx-auto max-w-[1800px] px-4 py-3">
      <div className="grid gap-3" style={{ gridTemplateColumns: "320px minmax(0,1fr) minmax(360px, 420px)" }}>
        {/* ── 좌: 입력 ── */}
        <aside className="flex flex-col gap-3">
          <section className="app-card p-3 flex flex-col gap-3">
            <div className="text-[12px] font-semibold text-text">이미지</div>
            <Dropzone onFiles={onFiles} accept=".jpg,.jpeg,.png,.tif,.tiff,.bmp,image/*" filled={!!file} compact={!!file}>
              {file ? (
                <div className="flex items-center gap-2 w-full text-left">
                  {fileUrl && (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img src={fileUrl} alt="" className="w-12 h-12 object-cover rounded-md border border-line flex-shrink-0 bg-black" />
                  )}
                  <div className="min-w-0 flex-1">
                    <div className="text-[12px] font-medium text-text truncate">{file.name}</div>
                    <div className="text-[10px] text-text-faint">
                      {(file.size / 1024).toFixed(0)} KB
                      {result?.shape && ` · ${result.shape[1]}×${result.shape[0]}`}
                    </div>
                  </div>
                  <button
                    type="button"
                    className="w-6 h-6 rounded-md hover:bg-bg3 inline-flex items-center justify-center text-text-faint"
                    title="제거"
                    onClick={(e) => {
                      e.stopPropagation();
                      setFile(null);
                      setResult(null);
                      setPreviewPng(null);
                    }}
                  >
                    <X size={12} />
                  </button>
                </div>
              ) : undefined}
            </Dropzone>
          </section>

          <section className="app-card p-3 flex flex-col gap-2">
            <div className="text-[12px] font-semibold text-text">분할 엔진</div>
            <EngineSwitch value={params.engine} onChange={(v) => setParams({ ...params, engine: v })} />
            <div className="text-[10px] text-text-faint">
              {isWs ? "Otsu 이진화 + 거리변환 + 마커 기반 watershed" : "U-Net++ × YOLO11 하이브리드 (서버 기본 가중치 사용)"}
            </div>
          </section>

          <section className="app-card p-3 flex flex-col">
            <div className="text-[12px] font-semibold text-text mb-1">스케일</div>
            <SelectField
              label="모드"
              value={params.scale.mode}
              options={[
                { value: "auto", label: "자동 (파일명)" },
                { value: "manual", label: "수동" },
              ]}
              onChange={(v) => setParams({ ...params, scale: { ...params.scale, mode: v } })}
            />
            {params.scale.mode === "manual" ? (
              <NumberField
                label="μm/px"
                value={params.scale.pixel_to_micro}
                min={0.0001}
                step={0.001}
                onChange={(v) => setParams({ ...params, scale: { ...params.scale, pixel_to_micro: v } })}
              />
            ) : (
              <NumberField
                label="기본값 (매칭 실패 시)"
                unit="μm/px"
                value={params.scale.auto_default}
                min={0.0001}
                step={0.01}
                onChange={(v) => setParams({ ...params, scale: { ...params.scale, auto_default: v } })}
              />
            )}
            {result && (
              <div className="text-[10px] text-text-faint mt-1">
                적용된 스케일: <span className="font-mono">{result.pixel_to_micro.toFixed(4)}</span> μm/px
              </div>
            )}
          </section>

          <div className="flex gap-2">
            <button type="button" className="btn-primary flex-1 h-10 text-[13px] inline-flex items-center justify-center gap-2" disabled={!file || segBusy} onClick={runSegment}>
              {segBusy ? <Loader2 size={14} className="animate-spin" /> : <Play size={14} />}
              {segBusy ? "분할 중…" : "분할 실행"}
            </button>
            {segBusy && (
              <button type="button" className="h-10 px-3 rounded-md border border-line text-[12px] text-text-mute hover:text-text" onClick={cancelSegment}>
                취소
              </button>
            )}
          </div>

          {result && (
            <div className="text-[10px] text-text-faint font-mono px-1">
              전처리 {result.timing_ms.preprocess?.toFixed(0)}ms · 분할 {result.timing_ms.segment?.toFixed(0)}ms · 측정 {result.timing_ms.measure?.toFixed(0)}ms
            </div>
          )}

          {error && (
            <div className="app-card p-2.5 border-bad/30 bg-bad-soft text-[11px] text-bad break-words">
              {error}
            </div>
          )}
          {notice && <div className="app-card p-2.5 border-good/30 bg-good-soft text-[11px] text-good fade-in">{notice}</div>}
        </aside>

        {/* ── 중앙: 뷰어 ── */}
        <section className="flex flex-col gap-2 min-w-0">
          <div className="flex items-center gap-1.5 flex-wrap">
            {VIEWS.map((v) => {
              const disabled = (v.wsOnly && !isWs) || !viewAvailable(v.key);
              return (
                <Chip key={v.key} active={view === v.key} disabled={disabled} onClick={() => !disabled && setView(v.key)} title={v.wsOnly && !isWs ? "AI 엔진에서는 제공되지 않음" : undefined}>
                  {v.label}
                </Chip>
              );
            })}
            {preBusy && (
              <span className="chip chip-mute ml-auto">
                <Loader2 size={11} className="animate-spin" /> 전처리 미리보기
              </span>
            )}
          </div>
          <div className="app-card p-2" style={{ height: "calc(100vh - 300px)", minHeight: 420 }}>
            <ImageViewer
              src={viewSrc}
              label={`${VIEWS.find((v) => v.key === view)?.label ?? view}${file ? ` — ${file.name}` : ""}`}
              downloadName={file ? `${file.name.replace(/\.[^.]+$/, "")}_${view}.png` : undefined}
              emptyText={file ? "이 뷰는 아직 없습니다. '분할 실행'을 누르세요." : "좌측에 SEM 이미지를 올리세요"}
            />
          </div>

          {/* ── 하단 결과 바 ── */}
          <div className="app-card">
            <div className="flex items-stretch flex-wrap border-b border-line-soft">
              <div className="flex flex-col px-3 py-1.5 border-r border-line-soft min-w-[110px]">
                <span className="text-[10px] text-text-faint">입자 수</span>
                <span className="text-[13px] font-semibold font-mono tabular-nums">
                  {result ? result.n_after : "–"}
                  <span className="text-[10px] text-text-faint font-normal"> / before {result ? result.n_before : "–"}</span>
                </span>
              </div>
              <DStat label="D10 (μm)" before={dBefore?.d10} after={dAfter?.d10} />
              <DStat label="D50 (μm)" before={dBefore?.d50} after={dAfter?.d50} />
              <DStat label="D90 (μm)" before={dBefore?.d90} after={dAfter?.d90} />
              <DStat label="Span" before={dBefore?.span} after={dAfter?.span} />
              <DStat label="Mean (μm)" before={dBefore?.mean} after={dAfter?.mean} />
              <DStat label="CV" before={dBefore?.cv} after={dAfter?.cv} />
              <div className="ml-auto flex items-center gap-1.5 px-3 py-1.5">
                <button
                  type="button"
                  className="h-7 px-2.5 rounded-md border border-line text-[11px] font-medium text-text-mute hover:text-text hover:border-line-strong inline-flex items-center gap-1"
                  onClick={() => downloadParamsJson(params)}
                >
                  <Download size={12} /> 파라미터 JSON 내보내기
                </button>
                <button
                  type="button"
                  className="h-7 px-2.5 rounded-md border border-line text-[11px] font-medium text-text-mute hover:text-text hover:border-line-strong inline-flex items-center gap-1"
                  onClick={() => fileInputRef.current?.click()}
                >
                  <Upload size={12} /> JSON 불러오기
                </button>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".json,application/json"
                  className="hidden"
                  onChange={(e) => {
                    const f = e.target.files?.[0];
                    if (f) onImportFile(f);
                    e.target.value = "";
                  }}
                />
                <button type="button" className="btn-primary h-7 px-3 text-[11px] inline-flex items-center gap-1" onClick={goBatch}>
                  배치 분석으로 <ArrowRight size={12} />
                </button>
              </div>
            </div>
            <div className="flex items-center gap-2 px-3 py-1.5">
              <span className="text-[11px] text-text-mute">입자 표</span>
              <Chip active={tableSide === "after"} onClick={() => setTableSide("after")}>
                After {result ? `(${result.particles_after.length})` : ""}
              </Chip>
              <Chip active={tableSide === "before"} onClick={() => setTableSide("before")}>
                Before {result ? `(${result.particles_before.length})` : ""}
              </Chip>
            </div>
            <DataTable<ParticleRow>
              rows={rows}
              columns={PARTICLE_COLS}
              rowKey={(r, i) => `${r.Label}-${i}`}
              maxRows={500}
              maxHeight={240}
              dense
              className="!border-0 !rounded-none border-t border-line-soft"
              emptyText={result ? "입자가 없습니다" : "분할 실행 후 입자 목록이 표시됩니다"}
              defaultSort={{ key: "Diameter (μm)", dir: "desc" }}
            />
          </div>
        </section>

        {/* ── 우: 파라미터 ── */}
        <aside className={clsx("min-w-0 overflow-y-auto pr-0.5", !loaded && "opacity-50")} style={{ maxHeight: "calc(100vh - 72px)" }}>
          <ParamsPanel params={params} onChange={setParams} mode="tuner" onSemChange={onSemChange} showScale={false} />
        </aside>
      </div>
    </main>
  );
}
