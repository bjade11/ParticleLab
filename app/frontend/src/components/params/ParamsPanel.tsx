"use client";

// AnalysisParams 전 섹션 편집 패널. 튜너/배치 공용.
// mode="tuner": 공간분석·출력 섹션 숨김. mode="batch": 전부 표시.

import { useCallback } from "react";
import clsx from "clsx";
import type {
  AfterFilterParams,
  AnalysisParams,
  EdgeMethod,
  FilterMode,
  HybridParams,
  MarkerMethod,
  MarkerParams,
  MeasureParams,
  MorphologyParams,
  OutputParams,
  ScaleParams,
  SemParams,
  SpatialParams,
  WatershedParams,
} from "@/lib/types";
import { DEFAULT_MORPHOLOGY, DEFAULT_PARAMS, deepClone, oddify } from "@/lib/params";
import NumberField from "@/components/ui/NumberField";
import SliderField from "@/components/ui/SliderField";
import Toggle from "@/components/ui/Toggle";
import SelectField from "@/components/ui/SelectField";
import ParamSection from "@/components/ui/ParamSection";
import Chip from "@/components/ui/Chip";

export interface ParamsPanelProps {
  params: AnalysisParams;
  onChange: (next: AnalysisParams) => void;
  mode: "tuner" | "batch";
  /** SEM 전처리(및 블러 커널) 변경 시 호출 — 튜너 실시간 미리보기 디바운스 지점 */
  onSemChange?: (next: AnalysisParams) => void;
  /** 스케일 섹션 표시 (기본 true) */
  showScale?: boolean;
  disabled?: boolean;
  className?: string;
}

const EDGE_METHODS: { value: EdgeMethod; label: string }[] = [
  { value: "laplacian", label: "Laplacian" },
  { value: "sobelx", label: "Sobel X" },
  { value: "sobely", label: "Sobel Y" },
  { value: "scharr", label: "Scharr" },
];
const MARKER_METHODS: { value: MarkerMethod; label: string }[] = [
  { value: "peak_local_max", label: "peak_local_max (국소 최대)" },
  { value: "dilate_threshold", label: "dilate_threshold (거리 임계)" },
  { value: "erosion", label: "erosion (침식)" },
  { value: "blob_detection", label: "blob_detection (LoG 블롭)" },
];
const FILTER_MODES: { value: FilterMode; label: string }[] = [
  { value: "value", label: "값" },
  { value: "percentile", label: "백분위" },
];
const DEVICES = [
  { value: "auto", label: "auto" },
  { value: "cuda", label: "cuda" },
  { value: "cpu", label: "cpu" },
];

function TextRow({
  label,
  hint,
  value,
  onChange,
  placeholder,
  disabled,
}: {
  label: string;
  hint?: string;
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
  disabled?: boolean;
}) {
  return (
    <div className="toss-row py-1.5 flex items-center justify-between gap-3">
      <div className="min-w-0">
        <div className="text-[12px] text-text">{label}</div>
        {hint && <div className="text-[10px] text-text-faint leading-tight mt-0.5">{hint}</div>}
      </div>
      <input
        type="text"
        className="h-7 rounded-md border px-2 text-[11px] font-mono w-[160px]"
        value={value}
        placeholder={placeholder}
        disabled={disabled}
        onChange={(e) => onChange(e.target.value)}
      />
    </div>
  );
}

function SubHeading({ children }: { children: React.ReactNode }) {
  return <div className="text-[10px] font-semibold text-text-faint tracking-wide2 uppercase pt-2 pb-0.5">{children}</div>;
}

export default function ParamsPanel({ params, onChange, mode, onSemChange, showScale = true, disabled, className }: ParamsPanelProps) {
  const p = params;
  const isWs = p.engine === "watershed";

  const patch = useCallback(
    <K extends keyof AnalysisParams>(key: K, part: Partial<AnalysisParams[K]>, sem = false) => {
      const cur = p[key];
      const next: AnalysisParams = {
        ...p,
        [key]: typeof cur === "object" && cur !== null ? { ...(cur as object), ...(part as object) } : part,
      } as AnalysisParams;
      onChange(next);
      if (sem) onSemChange?.(next);
    },
    [p, onChange, onSemChange],
  );

  const setScale = (part: Partial<ScaleParams>) => patch("scale", part);
  const setSem = (part: Partial<SemParams>) => patch("sem", part, true);
  const setWs = (part: Partial<WatershedParams>, sem = false) => patch("watershed", part, sem);
  const setMarker = (part: Partial<MarkerParams>) => setWs({ marker: { ...p.watershed.marker, ...part } });
  const setMorph = (op: keyof MorphologyParams, part: Partial<MorphologyParams[keyof MorphologyParams]>) => {
    const m = p.watershed.morphology ?? deepClone(DEFAULT_MORPHOLOGY);
    setWs({ morphology: { ...m, [op]: { ...m[op], ...part } } });
  };
  const setHy = (part: Partial<HybridParams>) => patch("hybrid", part);
  const setMe = (part: Partial<MeasureParams>) => patch("measure", part);
  const setAf = (part: Partial<AfterFilterParams>) => patch("after_filter", part);
  const setSp = (part: Partial<SpatialParams>) => patch("spatial", part);
  const setOut = (part: Partial<OutputParams>) => patch("output", part);

  const resetSection = (key: keyof AnalysisParams, sem = false) => {
    const next = { ...p, [key]: deepClone(DEFAULT_PARAMS[key]) } as AnalysisParams;
    onChange(next);
    if (sem) onSemChange?.(next);
  };

  const ws = p.watershed;
  const hy = p.hybrid;
  const me = p.measure;
  const af = p.after_filter;
  const sp = p.spatial;
  const out = p.output;
  const morphOn = ws.morphology !== null;

  return (
    <div className={clsx("flex flex-col gap-2", disabled && "opacity-60 pointer-events-none", className)}>
      {/* ── SEM 전처리 ── */}
      <ParamSection
        title="SEM 전처리"
        subtitle="crop → B/C → gamma → CLAHE → USM"
        onReset={() => resetSection("sem", true)}
        badge={mode === "tuner" ? <Chip tone="ink" className="!py-0 !px-1.5 !text-[9px]">실시간</Chip> : undefined}
      >
        <SliderField
          label="하단 크롭 비율"
          hint="0~1, 하단 스케일바 제거"
          value={p.sem.bottom_crop_ratio}
          min={0}
          max={0.3}
          step={0.001}
          onChange={(v) => setSem({ bottom_crop_ratio: v })}
        />
        <SliderField label="밝기" hint="가산 오프셋" value={p.sem.brightness} min={-100} max={100} step={1} integer onChange={(v) => setSem({ brightness: v })} />
        <SliderField label="대비" hint="승산 계수 (1.0 = 원본)" value={p.sem.contrast} min={0.1} max={3} step={0.05} onChange={(v) => setSem({ contrast: v })} />
        <SliderField label="감마" hint="1.0 = 원본, <1 밝게, >1 어둡게" value={p.sem.gamma} min={0.1} max={3} step={0.05} onChange={(v) => setSem({ gamma: v })} />
        <Toggle label="CLAHE" hint="지역 대비 제한 히스토그램 평활화" checked={p.sem.use_clahe} onChange={(v) => setSem({ use_clahe: v })} />
        <SliderField label="CLAHE clip" value={p.sem.clahe_clip} min={0.5} max={10} step={0.1} disabled={!p.sem.use_clahe} onChange={(v) => setSem({ clahe_clip: v })} />
        <SliderField label="CLAHE tile" hint="타일 격자 크기" value={p.sem.clahe_tile} min={2} max={32} step={1} integer disabled={!p.sem.use_clahe} onChange={(v) => setSem({ clahe_tile: v })} />
        <Toggle label="언샤프 마스크 (USM)" hint="엣지 선명화" checked={p.sem.use_usm} onChange={(v) => setSem({ use_usm: v })} />
        <SliderField label="USM amount" value={p.sem.usm_amount} min={0} max={3} step={0.05} disabled={!p.sem.use_usm} onChange={(v) => setSem({ usm_amount: v })} />
        <SliderField label="USM radius" value={p.sem.usm_radius} min={0.5} max={10} step={0.1} disabled={!p.sem.use_usm} onChange={(v) => setSem({ usm_radius: v })} />
      </ParamSection>

      {/* ── 분할: Watershed ── */}
      {isWs && (
        <ParamSection title="분할 — Watershed" subtitle="블러 → Otsu → 모폴로지 → EDT → 마커 → watershed" onReset={() => resetSection("watershed", true)}>
          <SubHeading>블러 (GaussianBlur, Otsu 직전 1회)</SubHeading>
          <div className="toss-row py-1.5 flex items-center justify-between gap-3">
            <div>
              <div className="text-[12px] text-text">블러 커널 (x, y)</div>
              <div className="text-[10px] text-text-faint leading-tight mt-0.5">홀수. 튜너 미리보기에도 반영</div>
            </div>
            <div className="flex items-center gap-1">
              <NumberField
                value={ws.blur_kernel[0]}
                min={1}
                max={51}
                odd
                inputClassName="!w-[56px]"
                onChange={(v) => setWs({ blur_kernel: [oddify(v), ws.blur_kernel[1]] }, true)}
              />
              <span className="text-text-faint text-[11px]">×</span>
              <NumberField
                value={ws.blur_kernel[1]}
                min={1}
                max={51}
                odd
                inputClassName="!w-[56px]"
                onChange={(v) => setWs({ blur_kernel: [ws.blur_kernel[0], oddify(v)] }, true)}
              />
            </div>
          </div>

          <SubHeading>엣지 강화</SubHeading>
          <Toggle label="엣지 강화" hint="블러 전에 엣지 응답을 가산" checked={ws.enhance_edges} onChange={(v) => setWs({ enhance_edges: v })} />
          <SelectField label="엣지 방법" value={ws.edge.method} options={EDGE_METHODS} disabled={!ws.enhance_edges} onChange={(v) => setWs({ edge: { ...ws.edge, method: v } })} />
          <SliderField label="엣지 가중치" value={ws.edge.weight} min={0} max={2} step={0.05} disabled={!ws.enhance_edges} onChange={(v) => setWs({ edge: { ...ws.edge, weight: v } })} />
          <NumberField label="엣지 커널" hint="홀수 (1·3·5·7)" value={ws.edge.kernel} min={1} max={7} odd disabled={!ws.enhance_edges} onChange={(v) => setWs({ edge: { ...ws.edge, kernel: v } })} />

          <SubHeading>모폴로지 (open → close → erode → dilate)</SubHeading>
          <Toggle
            label="모폴로지 사용"
            hint="끄면 생략 (morphology = null)"
            checked={morphOn}
            onChange={(v) => setWs({ morphology: v ? deepClone(DEFAULT_MORPHOLOGY) : null })}
          />
          {morphOn &&
            (["open", "close", "erode", "dilate"] as (keyof MorphologyParams)[]).map((op) => {
              const m = ws.morphology as MorphologyParams;
              const labels: Record<keyof MorphologyParams, string> = { open: "Open", close: "Close", erode: "Erode", dilate: "Dilate" };
              return (
                <div key={op} className="toss-row py-1.5 flex items-center justify-between gap-3">
                  <div className="text-[12px] text-text">{labels[op]}</div>
                  <div className="flex items-center gap-2">
                    <span className="text-[10px] text-text-faint">커널</span>
                    <NumberField value={m[op].kernel} min={1} max={31} integer inputClassName="!w-[56px]" onChange={(v) => setMorph(op, { kernel: v })} />
                    <span className="text-[10px] text-text-faint">반복</span>
                    <NumberField value={m[op].iterations} min={0} max={20} integer inputClassName="!w-[56px]" onChange={(v) => setMorph(op, { iterations: v })} />
                  </div>
                </div>
              );
            })}

          <SubHeading>마커</SubHeading>
          <SelectField label="마커 방법" value={ws.marker_method} options={MARKER_METHODS} selectClassName="!min-w-[190px]" onChange={(v) => setWs({ marker_method: v })} />
          {ws.marker_method === "peak_local_max" && (
            <SliderField label="peak_min_distance" hint="국소 최대 간 최소 거리 (px)" value={ws.marker.peak_min_distance} min={1} max={50} step={1} integer onChange={(v) => setMarker({ peak_min_distance: v })} />
          )}
          {ws.marker_method === "dilate_threshold" && (
            <>
              <SliderField label="distance_threshold" hint="dist > thr × max 인 영역을 마커로" value={ws.marker.distance_threshold} min={0.01} max={0.99} step={0.01} onChange={(v) => setMarker({ distance_threshold: v })} />
              <NumberField label="dilate_kernel" value={ws.marker.dilate_kernel} min={1} max={31} integer onChange={(v) => setMarker({ dilate_kernel: v })} />
              <NumberField label="dilate_iter" value={ws.marker.dilate_iter} min={0} max={20} integer onChange={(v) => setMarker({ dilate_iter: v })} />
            </>
          )}
          {ws.marker_method === "erosion" && (
            <>
              <NumberField label="erosion_kernel" value={ws.marker.erosion_kernel} min={1} max={31} integer onChange={(v) => setMarker({ erosion_kernel: v })} />
              <NumberField label="erosion_iter" value={ws.marker.erosion_iter} min={0} max={20} integer onChange={(v) => setMarker({ erosion_iter: v })} />
            </>
          )}
          {ws.marker_method === "blob_detection" && (
            <>
              <NumberField label="blob_min_sigma" value={ws.marker.blob_min_sigma} min={0.1} step={0.1} onChange={(v) => setMarker({ blob_min_sigma: v })} />
              <NumberField label="blob_max_sigma" value={ws.marker.blob_max_sigma} min={0.1} step={0.5} onChange={(v) => setMarker({ blob_max_sigma: v })} />
              <NumberField label="blob_num_sigma" value={ws.marker.blob_num_sigma} min={1} max={50} integer onChange={(v) => setMarker({ blob_num_sigma: v })} />
              <NumberField label="blob_threshold" value={ws.marker.blob_threshold} min={0} max={1} step={0.01} onChange={(v) => setMarker({ blob_threshold: v })} />
            </>
          )}

          <SubHeading>후처리</SubHeading>
          <SliderField label="min_size" hint="remove_small_objects (px²)" value={ws.min_size} min={0} max={500} step={1} integer allowOutOfRange onChange={(v) => setWs({ min_size: v })} />
        </ParamSection>
      )}

      {/* ── 분할: AI Hybrid ── */}
      {!isWs && (
        <ParamSection title="분할 — AI Hybrid" subtitle="U-Net × YOLO" onReset={() => resetSection("hybrid")}>
          <SubHeading>U-Net 임계</SubHeading>
          <SliderField label="high_thresh" hint="확정 전경 임계" value={hy.high_thresh} min={0} max={1} step={0.01} onChange={(v) => setHy({ high_thresh: v })} />
          <SliderField label="low_thresh" hint="후보 전경 임계" value={hy.low_thresh} min={0} max={1} step={0.01} onChange={(v) => setHy({ low_thresh: v })} />
          <SliderField label="boundary_thickness" hint="경계 두께 (px)" value={hy.boundary_thickness} min={0} max={15} step={1} integer onChange={(v) => setHy({ boundary_thickness: v })} />
          <SliderField label="min_anchor_area" hint="앵커 최소 면적 (px²)" value={hy.min_anchor_area} min={0} max={500} step={1} integer allowOutOfRange onChange={(v) => setHy({ min_anchor_area: v })} />
          <SliderField label="balance_floor" value={hy.balance_floor} min={0} max={1} step={0.01} onChange={(v) => setHy({ balance_floor: v })} />
          <SliderField label="tau_scale" value={hy.tau_scale} min={0.1} max={5} step={0.05} onChange={(v) => setHy({ tau_scale: v })} />

          <SubHeading>YOLO</SubHeading>
          <SliderField label="yolo_conf" hint="신뢰도 임계" value={hy.yolo_conf} min={0.01} max={1} step={0.01} onChange={(v) => setHy({ yolo_conf: v })} />
          <SliderField label="yolo_iou" hint="NMS IoU" value={hy.yolo_iou} min={0.05} max={1} step={0.01} onChange={(v) => setHy({ yolo_iou: v })} />
          <NumberField label="yolo_imgsz" hint="추론 해상도 (32 배수)" value={hy.yolo_imgsz} min={128} max={4096} step={32} integer onChange={(v) => setHy({ yolo_imgsz: v })} />
          <SliderField label="yolo_round_strength" hint="마스크 둥글림 강도" value={hy.yolo_round_strength} min={0} max={10} step={1} integer onChange={(v) => setHy({ yolo_round_strength: v })} />
          <Toggle label="겹침 제거" hint="yolo_remove_overlaps" checked={hy.yolo_remove_overlaps} onChange={(v) => setHy({ yolo_remove_overlaps: v })} />
          <Toggle label="포함된 객체 제거" hint="yolo_remove_enclosed" checked={hy.yolo_remove_enclosed} onChange={(v) => setHy({ yolo_remove_enclosed: v })} />
          <SliderField label="enclosed_ratio" hint="포함 판정 비율" value={hy.enclosed_ratio} min={0.5} max={1} step={0.01} disabled={!hy.yolo_remove_enclosed} onChange={(v) => setHy({ enclosed_ratio: v })} />

          <SubHeading>추론</SubHeading>
          <NumberField
            label="unet_imgsz"
            hint="비우면 원본 크기"
            value={hy.unet_imgsz}
            min={64}
            max={4096}
            step={32}
            integer
            nullable
            placeholder="원본"
            onNull={() => setHy({ unet_imgsz: null })}
            onChange={(v) => setHy({ unet_imgsz: v })}
          />
          <SelectField label="device" value={hy.device} options={DEVICES} onChange={(v) => setHy({ device: v })} />
          <TextRow label="U-Net 가중치" hint='"basic" 또는 서버 절대경로' value={hy.unet_weights} placeholder="basic" onChange={(v) => setHy({ unet_weights: v || "basic" })} />
          <TextRow label="YOLO 가중치" hint='"basic" 또는 서버 절대경로' value={hy.yolo_weights} placeholder="basic" onChange={(v) => setHy({ yolo_weights: v || "basic" })} />
        </ParamSection>
      )}

      {/* ── 특성 필터 (measure) ── */}
      <ParamSection title="특성 필터 (measure)" subtitle="라벨맵 → regionprops 단계" onReset={() => resetSection("measure")}>
        <Toggle label="경계 접촉 입자 제외" hint="이미지 가장자리에 닿은 입자 제거" checked={me.filter_border_particles} onChange={(v) => setMe({ filter_border_particles: v })} />
        <SliderField label="최소 윤곽 점 수" hint="min_contour_points" value={me.min_contour_points} min={0} max={200} step={1} integer allowOutOfRange onChange={(v) => setMe({ min_contour_points: v })} />
        <NumberField label="최소 직경" hint="min_diameter" unit="μm" value={me.min_diameter} min={0} step={0.01} onChange={(v) => setMe({ min_diameter: v })} />
        <Toggle label="선형 윤곽 제외" hint="직선에 가까운 윤곽 제거" checked={me.exclude_linear_contour} onChange={(v) => setMe({ exclude_linear_contour: v })} />
        <SliderField label="선형 윤곽 백분위" hint="linear_contour_percentile" value={me.linear_contour_percentile} min={0} max={50} step={0.5} disabled={!me.exclude_linear_contour} onChange={(v) => setMe({ linear_contour_percentile: v })} />
        <Toggle label="어두운 입자 제외" hint="평균 밝기 하위 백분위 제거" checked={me.exclude_dark_particle} onChange={(v) => setMe({ exclude_dark_particle: v })} />
        <SliderField label="어두운 입자 백분위" hint="dark_particle_percentile" value={me.dark_particle_percentile} min={0} max={50} step={0.5} disabled={!me.exclude_dark_particle} onChange={(v) => setMe({ dark_particle_percentile: v })} />
        <SliderField label="곡률 백분위" hint="curvature_percentile, 0 이면 끔" value={me.curvature_percentile} min={0} max={50} step={0.5} onChange={(v) => setMe({ curvature_percentile: v })} />
      </ParamSection>

      {/* ── After 필터 ── */}
      <ParamSection title="After 필터" subtitle="Before → After" onReset={() => resetSection("after_filter")}>
        <SliderField label="직경 하위 백분위 제거" hint="diameter_lower_percentile (0 = 끔)" value={af.diameter_lower_percentile} min={0} max={50} step={0.5} onChange={(v) => setAf({ diameter_lower_percentile: v })} />
        <SliderField label="직경 상위 백분위 제거" hint="diameter_upper_percentile (0 = 끔)" value={af.diameter_upper_percentile} min={0} max={50} step={0.5} onChange={(v) => setAf({ diameter_upper_percentile: v })} />
        <SubHeading>원형도</SubHeading>
        <Toggle label="원형도 필터 사용" hint="끄면 원형도로 제외하지 않음" checked={af.enable_circularity} onChange={(v) => setAf({ enable_circularity: v })} />
        {af.enable_circularity && (
          <>
            <SelectField label="원형도 기준" value={af.circularity_mode} options={FILTER_MODES} onChange={(v) => setAf({ circularity_mode: v })} />
            {af.circularity_mode === "value" ? (
              <SliderField label="원형도 최소값" hint="이 값 미만 제외" value={af.circularity_value} min={0} max={1} step={0.01} onChange={(v) => setAf({ circularity_value: v })} />
            ) : (
              <SliderField label="원형도 하위 백분위" hint="하위 n% 제외" value={af.circularity_percentile} min={0} max={50} step={0.5} onChange={(v) => setAf({ circularity_percentile: v })} />
            )}
          </>
        )}
        <SubHeading>종횡비</SubHeading>
        <Toggle label="종횡비 필터 사용" hint="끄면 종횡비로 제외하지 않음" checked={af.enable_aspect_ratio} onChange={(v) => setAf({ enable_aspect_ratio: v })} />
        {af.enable_aspect_ratio && (
          <>
            <SelectField label="종횡비 기준" value={af.aspect_ratio_mode} options={FILTER_MODES} onChange={(v) => setAf({ aspect_ratio_mode: v })} />
            {af.aspect_ratio_mode === "value" ? (
              <SliderField label="종횡비 최대값" hint="이 값 초과 제외" value={af.aspect_ratio_value} min={1} max={5} step={0.05} allowOutOfRange onChange={(v) => setAf({ aspect_ratio_value: v })} />
            ) : (
              <SliderField label="종횡비 상위 백분위" hint="상위 n% 제외" value={af.aspect_ratio_percentile} min={0} max={50} step={0.5} onChange={(v) => setAf({ aspect_ratio_percentile: v })} />
            )}
          </>
        )}
      </ParamSection>

      {/* ── 스케일 ── */}
      {showScale && (
        <ParamSection title="스케일" subtitle="μm/px" onReset={() => resetSection("scale")} defaultOpen={mode === "batch"}>
          <SelectField
            label="모드"
            hint="auto: 파일명의 배율(…k)에서 추출"
            value={p.scale.mode}
            options={[
              { value: "auto", label: "자동 (파일명)" },
              { value: "manual", label: "수동" },
            ]}
            onChange={(v) => setScale({ mode: v })}
          />
          <NumberField label="pixel_to_micro" hint={p.scale.mode === "manual" ? "수동 μm/px" : "auto 실패 시 fallback"} unit="μm/px" value={p.scale.pixel_to_micro} min={0.0001} step={0.001} onChange={(v) => setScale({ pixel_to_micro: v })} />
          <NumberField label="auto_default" hint="파일명 매칭 실패 시 값" unit="μm/px" value={p.scale.auto_default} min={0.0001} step={0.01} disabled={p.scale.mode !== "auto"} onChange={(v) => setScale({ auto_default: v })} />
        </ParamSection>
      )}

      {mode === "batch" && (
        <>
          {/* ── 공간 분석 ── */}
          <ParamSection title="공간 분석" subtitle="경계 접촉 · NND · DBSCAN" onReset={() => resetSection("spatial")} defaultOpen={false}>
            <Toggle label="경계 접촉 분석" hint="입자 간 접촉 쌍 탐지" checked={sp.enable_boundary} onChange={(v) => setSp({ enable_boundary: v })} />
            <NumberField label="접촉 임계" hint="boundary_contact_threshold" unit="px" value={sp.boundary_contact_threshold} min={0} step={0.1} disabled={!sp.enable_boundary} onChange={(v) => setSp({ boundary_contact_threshold: v })} />
            <Toggle label="최근접 거리 (NND) / 응집도" hint="enable_nnd" checked={sp.enable_nnd} onChange={(v) => setSp({ enable_nnd: v })} />
            <Toggle label="DBSCAN 군집" hint="enable_dbscan" checked={sp.enable_dbscan} onChange={(v) => setSp({ enable_dbscan: v })} />
            <Toggle label="eps 자동" hint="평균 직경 × factor" checked={sp.dbscan_eps_auto} disabled={!sp.enable_dbscan} onChange={(v) => setSp({ dbscan_eps_auto: v })} />
            {sp.dbscan_eps_auto ? (
              <NumberField label="eps factor" value={sp.dbscan_eps_factor} min={0.1} step={0.1} disabled={!sp.enable_dbscan} onChange={(v) => setSp({ dbscan_eps_factor: v })} />
            ) : (
              <NumberField label="eps 값" unit="μm" value={sp.dbscan_eps_value} min={0.01} step={0.1} disabled={!sp.enable_dbscan} onChange={(v) => setSp({ dbscan_eps_value: v })} />
            )}
            <Toggle label="min_samples 자동" hint="입자 수 × factor" checked={sp.dbscan_min_samples_auto} disabled={!sp.enable_dbscan} onChange={(v) => setSp({ dbscan_min_samples_auto: v })} />
            {sp.dbscan_min_samples_auto ? (
              <NumberField label="min_samples factor" value={sp.dbscan_min_samples_factor} min={0.0001} step={0.005} disabled={!sp.enable_dbscan} onChange={(v) => setSp({ dbscan_min_samples_factor: v })} />
            ) : (
              <NumberField label="min_samples 값" value={sp.dbscan_min_samples_value} min={1} integer disabled={!sp.enable_dbscan} onChange={(v) => setSp({ dbscan_min_samples_value: v })} />
            )}
          </ParamSection>

          {/* ── 출력 ── */}
          <ParamSection title="출력 옵션" onReset={() => resetSection("output")} defaultOpen={false}>
            <Toggle label="PPT 생성" hint="particle_analysis_result.pptx" checked={out.make_ppt} onChange={(v) => setOut({ make_ppt: v })} />
            <Toggle label="윤곽선 전용 이미지" hint="labeled_images/*_contour_only" checked={out.make_contour_only} onChange={(v) => setOut({ make_contour_only: v })} />
            <Toggle label="정규분포 플롯" hint="normal_dist/" checked={out.make_normal_dist} onChange={(v) => setOut({ make_normal_dist: v })} />
            <SliderField label="JPEG 품질" value={out.jpeg_quality} min={50} max={100} step={1} integer onChange={(v) => setOut({ jpeg_quality: v })} />
          </ParamSection>
        </>
      )}
    </div>
  );
}
