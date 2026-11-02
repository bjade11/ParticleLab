"use client";

// 입자 분석 — Lab. 단순 한글 한 페이지 UI.
// 레거시 / YOLO (다파일 + 다파일 GT) / U-Net 모두 백엔드와 직접 연결.

import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import type { MutableRefObject } from "react";

import { API_BASE } from "@/lib/api";

type Mode = "legacy" | "yolo" | "unet" | "hybrid" | "stats" | "augment";

type StatsRow = {
  value: number;
  yolo: Metrics | null;
  unet_low: Metrics | null;
  hybrid: Metrics | null;
};

type StatsResult = {
  params: string;
  sweep_mode?: "hybrid" | "yolo" | "unet";
  sweep_param: string;
  sweep_values: number[];
  metric_keys: (keyof Metrics)[];
  rows: StatsRow[];
  chart_pngs: Record<string, string>;
  excel_data_url?: string;
  excel_size_bytes?: number;
  n_images: number;
  n_with_gt: number;
  unet_method?: string;
  unet_image_size?: number;
  yolo_model_name?: string;
};

type ResearchSweepRow = { value: number; metrics: Metrics | null };
type ResearchGrid = {
  x_param: string;
  y_param: string;
  x_values: number[];
  y_values: number[];
  metrics_by_xy: (Metrics | null)[][];
};
type ResearchResult = {
  operating: Record<string, number | boolean>;
  sweep_specs: Record<string, number[]>;
  sweeps: Record<string, ResearchSweepRow[]>;
  grid: ResearchGrid | null;
  panel_figure_png: string;
  heatmap_pngs: Record<string, string>;
  excel_data_url?: string;
  excel_size_bytes?: number;
  unet_method?: string;
  unet_image_size?: number;
  yolo_model_name?: string;
  n_images: number;
  n_with_gt: number;
};
type ResearchJob = {
  job_id: string;
  status: "pending" | "running" | "done" | "error";
  stage: string;
  progress: number;
  current_step: number;
  total_steps: number;
  current_param: string | null;
  current_value: string | number | null;
  current_image: string | null;
  elapsed_sec: number;
  eta_sec: number | null;
  n_images: number;
  n_with_gt: number;
  n_sweep_params: number;
  has_grid: boolean;
  unet_method?: string;
  unet_image_size?: number;
  yolo_model_name?: string;
  error: string | null;
  result: ResearchResult | null;
};

type StatsJob = {
  job_id: string;
  status: "pending" | "running" | "done" | "error";
  stage: string;            // queued / loading_weights / preparing / sweeping / rendering / done
  progress: number;          // 0~1
  current_step: number;
  total_steps: number;
  current_value: number | null;
  current_image: string | null;
  elapsed_sec: number;
  eta_sec: number | null;
  sweep_mode: "hybrid" | "yolo" | "unet";
  sweep_param: string;
  n_sweep_values: number;
  n_images: number;
  n_with_gt: number;
  unet_method?: string;
  unet_image_size?: number;
  yolo_model_name?: string;
  error: string | null;
  result: StatsResult | null;
};

type Metrics = {
  Precision: number;
  Recall: number;
  Accuracy: number;
  IoU: number;
  F1: number;
  "mAP@0.5": number;
  ParticleCountError: number;
  n_gt_inst: number;
  n_pred_inst: number;
};

type YoloItem = {
  filename: string;
  shape: [number, number];
  n_detections: number;
  mean_conf: number;
  overlay_png: string;
  mask_png: string;
  metrics?: Metrics;
  gt_png?: string;
  gt_filename?: string;
};

type LegacyItem = {
  filename: string;
  shape: [number, number];
  pred_png: string;
  pred_pixels: number;
  metrics?: Metrics;
  gt_png?: string;
  gt_filename?: string;
};

type LegacyResult = {
  params: string;
  n_images: number;
  n_with_gt: number;
  results: LegacyItem[];
  summary?: Metrics & { n_with_gt: number };
};

type YoloResult = {
  params: string;
  weights_source?: string;
  n_images: number;
  total_detections: number;
  n_with_gt: number;
  results: YoloItem[];
  summary?: Metrics & { n_with_gt: number };
  result_id?: string;
  excel_data_url?: string;
  excel_size_bytes?: number;
  excel_error?: string;
};

type YoloTrainResult = {
  model_id: string;
  base_arch: string;
  imgsz: number;
  n_pairs: number;
  n_unmatched: number;
  elapsed_sec: number;
  device: string;
  history: { epoch: number; train_loss: number; val_loss: number }[];
  params: string;
  model_bytes: number;
};

type YoloTrainProgress = {
  job_id: string;
  status: "pending" | "running" | "done" | "error";
  progress: number;
  epoch: number;
  total_epochs: number;
  batch?: number;
  total_batches?: number;
  train_loss: number | null;
  val_loss: number | null;
  elapsed_sec: number;
  eta_sec: number | null;
  error: string | null;
  result: YoloTrainResult | null;
  history_len: number;
  last_history: { epoch: number; train_loss: number; val_loss: number } | null;
  n_pairs: number;
  n_unmatched: number;
  base_arch: string;
};

type UnetItem = {
  filename: string;
  shape: [number, number];
  pred_pixels: number;
  filtered_pixels: number;
  prob_png: string;
  prob_color_png: string;
  prob_filtered_png: string;
  prob_gray_filtered_png: string;
  mask_png: string;
  mask_color_png: string;
  metrics?: Metrics;
  gt_png?: string;
  gt_filename?: string;
};

type UnetResult = {
  method: string;
  image_size: number;
  threshold: number;
  min_prob: number;
  weights_source: string;
  params: string;
  n_images: number;
  n_with_gt: number;
  results: UnetItem[];
  summary?: Metrics & { n_with_gt: number };
  excel_data_url?: string;
  excel_size_bytes?: number;
  excel_error?: string;
};

type UnetTrainResult = {
  model_id: string;
  method: string;
  image_size: number;
  n_pairs: number;
  n_unmatched: number;
  best_val_loss: number;
  elapsed_sec: number;
  device: string;
  history: { epoch: number; train_loss: number; val_loss: number }[];
  params: string;
  model_bytes: number;
};

type Particle = {
  id: number;       // 1, 2, ...
  label: string;    // "1", "2", ... ("p" 접두사 제거)
  area_px: number;  // pixel count
  cy: number;
  cx: number;
};

// 모듈 레벨 캐시 — 같은 base 이미지를 여러 번 디코드하지 않도록.
const _baseImageCache = new Map<string, Promise<HTMLImageElement | null>>();
function loadImageCached(src: string): Promise<HTMLImageElement | null> {
  if (!src) return Promise.resolve(null);
  const hit = _baseImageCache.get(src);
  if (hit) return hit;
  const p = new Promise<HTMLImageElement | null>((resolve) => {
    const im = new Image();
    im.onload = () => resolve(im);
    im.onerror = () => resolve(null);
    im.src = src;
  });
  _baseImageCache.set(src, p);
  return p;
}

/**
 * 인스턴스 컬러 이미지 위에 입자별 일련번호 (1, 2, ...) 를 캔버스로 그려 반환.
 * scale=1.0 이 기본값. 더 키우면 글자 ↑.  실시간 슬라이더 조절용.
 */
async function renderNumberedImage(
  baseSrc: string,
  particles: Particle[],
  scale: number,
): Promise<string | null> {
  if (!baseSrc) return null;
  const im = await loadImageCached(baseSrc);
  if (!im) return null;
  const canvas = document.createElement("canvas");
  canvas.width = im.naturalWidth || im.width;
  canvas.height = im.naturalHeight || im.height;
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  ctx.drawImage(im, 0, 0);
  ctx.textBaseline = "middle";
  ctx.textAlign = "center";
  (ctx as any).textRendering = "geometricPrecision";

  for (const p of particles) {
    const eqRadius = Math.sqrt(Math.max(1, p.area_px) / Math.PI);
    // 면적 반경 비례 + 사용자 scale
    const baseSize = Math.max(8, Math.min(72, eqRadius * 0.62));
    const sizePx = baseSize * scale;
    if (sizePx < 5) continue;
    ctx.font = `bold ${sizePx}px Inter, "GmarketSans", "Apple SD Gothic Neo", sans-serif`;
    const txt = String(p.id);
    // 흰 테두리 + 검정 글씨 — 어떤 배경 색에서도 읽힘
    ctx.lineJoin = "round";
    ctx.lineWidth = Math.max(2, sizePx * 0.22);
    ctx.strokeStyle = "rgba(255,255,255,0.95)";
    ctx.strokeText(txt, p.cx, p.cy);
    ctx.fillStyle = "#111111";
    ctx.fillText(txt, p.cx, p.cy);
  }
  return canvas.toDataURL("image/png");
}

type HybridItem = {
  filename: string;
  shape: [number, number];
  n_instances: number;
  n_yolo: number;
  K_unet_anchor: number;
  tau: number;
  overlay_png: string;
  mask_png: string;
  instances_png: string;
  instances_numbered_png?: string;
  particles?: Particle[];
  diag_prob_png: string;
  diag_high_png: string;
  diag_close_png: string;
  diag_boundary_png: string;
  diag_separator_png: string;
  diag_overlap_png: string;
  diag_hist_png?: string;
  hist_csv?: string;
  pixel_csv?: string;
  pixel_csv_rows?: number;
  metrics?: Metrics;
  metrics_yolo?: Metrics;
  metrics_unet_low?: Metrics;
  metrics_hybrid?: Metrics;
  gt_png?: string;
  gt_filename?: string;
};

type HybridResult = {
  params: string;
  n_images: number;
  total_instances: number;
  n_with_gt: number;
  results: HybridItem[];
  high_thresh?: number;
  low_thresh?: number;
  summary?: Metrics & { n_with_gt: number };
  summary_yolo?: Metrics & { n_with_gt: number };
  summary_unet_low?: Metrics & { n_with_gt: number };
  summary_hybrid?: Metrics & { n_with_gt: number };
  excel_data_url?: string;
  excel_size_bytes?: number;
  excel_error?: string;
};

type UnetTrainProgress = {
  job_id: string;
  status: "pending" | "running" | "done" | "error";
  progress: number;          // 0~1
  epoch: number;
  total_epochs: number;
  train_loss: number | null;
  val_loss: number | null;
  elapsed_sec: number;
  eta_sec: number | null;
  error: string | null;
  result: UnetTrainResult | null;
  history_len: number;
  last_history: { epoch: number; train_loss: number; val_loss: number } | null;
  n_pairs: number;
  n_unmatched: number;
  device_request: string;
};

export default function Page() {
  const [health, setHealth] = useState<"확인중" | "정상" | "끊김">("확인중");

  useEffect(() => {
    fetch(`${API_BASE}/api/health`)
      .then((r) => r.ok ? setHealth("정상") : setHealth("끊김"))
      .catch(() => setHealth("끊김"));
  }, []);

  // ── 기본 가중치 자동 로드 (basic_pt/) ──
  // 첫 페이지 진입 시 서버에 있는 default U-Net / YOLO .pt 를 fetch 해서
  // Hybrid/Stats 슬롯에 자동으로 채움. 사용자가 다른 파일을 올리면 즉시 교체됨.
  // File 객체 레퍼런스로 "이게 기본 모델인지" 판별 → UI 배지 + "기본값으로 되돌리기".
  const [defaultUnetFile, setDefaultUnetFile] = useState<File | null>(null);
  const [defaultYoloFile, setDefaultYoloFile] = useState<File | null>(null);
  const [defaultLoadStatus, setDefaultLoadStatus] = useState<"loading" | "ok" | "fail">("loading");
  useEffect(() => {
    let cancelled = false;
    async function loadOne(which: "unet" | "yolo"): Promise<File | null> {
      try {
        const r = await fetch(`${API_BASE}/api/basic_pt/${which}`);
        if (!r.ok) return null;
        const blob = await r.blob();
        const name = which === "unet" ? "U-net++.pt" : "yolo_11.pt";
        return new File([blob], name, { type: "application/octet-stream" });
      } catch { return null; }
    }
    (async () => {
      const [u, y] = await Promise.all([loadOne("unet"), loadOne("yolo")]);
      if (cancelled) return;
      if (u) {
        setDefaultUnetFile(u);
        setHybridUnetWeights(u);
        setStatsUnetWeights(u);
      }
      if (y) {
        setDefaultYoloFile(y);
        setHybridYoloWeights(y);
        setStatsYoloWeights(y);
      }
      setDefaultLoadStatus(u || y ? "ok" : "fail");
    })();
    return () => { cancelled = true; };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Right (params) column width — drag handle resizes it. Persisted in localStorage.
  const [rightW, setRightW] = useState<number>(420);
  const resizingRef = useRef(false);
  useEffect(() => {
    try {
      const s = localStorage.getItem("plab.rightW");
      if (s) {
        const v = parseInt(s, 10);
        if (Number.isFinite(v) && v >= 320 && v <= 820) setRightW(v);
      }
    } catch { /* ignore */ }
  }, []);
  useEffect(() => {
    const onMove = (e: MouseEvent) => {
      if (!resizingRef.current) return;
      const next = window.innerWidth - e.clientX - 4;
      const clamped = Math.max(320, Math.min(820, next));
      setRightW(clamped);
    };
    const onUp = () => {
      if (!resizingRef.current) return;
      resizingRef.current = false;
      document.body.classList.remove("is-resizing");
      try { localStorage.setItem("plab.rightW", String(rightW)); } catch { /* ignore */ }
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
  }, [rightW]);
  const startResize = () => {
    resizingRef.current = true;
    document.body.classList.add("is-resizing");
  };

  // v2: 추론 전용 — mode 는 항상 "hybrid"
  const [mode, setMode] = useState<Mode>("hybrid");
  void setMode;

  // 다파일 입력 (legacy / unet predict 공유)
  const [predictImages, setPredictImages] = useState<File[]>([]);
  const [predictGts, setPredictGts] = useState<File[]>([]);
  const [legacyActiveIdx, setLegacyActiveIdx] = useState(0);
  const [unetActiveIdx, setUnetActiveIdx] = useState(0);

  // YOLO 다파일
  const [yoloImages, setYoloImages] = useState<File[]>([]);
  const [yoloGts, setYoloGts] = useState<File[]>([]);

  const [weightsFile, setWeightsFile] = useState<File | null>(null);

  // 레거시
  const [invert, setInvert] = useState(false);
  const [minSize, setMinSize] = useState(8);
  const [morphKernel, setMorphKernel] = useState(0);
  const [useWatershed, setUseWatershed] = useState(true);
  const [watershedDist, setWatershedDist] = useState(5);

  // YOLO 사브탭
  const [yoloTab, setYoloTab] = useState<"train" | "predict">("predict");

  // YOLO 추론
  const [yoloConf, setYoloConf] = useState(0.15);
  const [yoloIou, setYoloIou] = useState(0.5);
  const [yoloImgsz, setYoloImgsz] = useState(640);
  const [yoloSmooth, setYoloSmooth] = useState(0);
  const [yoloRound, setYoloRound] = useState(0);
  const [yoloNoOverlap, setYoloNoOverlap] = useState(false);
  const [yoloDevice, setYoloDevice] = useState("auto");
  const [yoloUseTrained, setYoloUseTrained] = useState(false);

  // YOLO 학습
  const [yoloTrainImages, setYoloTrainImages] = useState<File[]>([]);
  const [yoloTrainMasks, setYoloTrainMasks] = useState<File[]>([]);
  const [yoloDatasetPath, setYoloDatasetPath] = useState("");  // 서버 디스크 폴더 (multipart 회피)
  const [yoloBaseArch, setYoloBaseArch] = useState("yolo11n-seg.pt");
  const [yoloTrainEpochs, setYoloTrainEpochs] = useState(50);
  const [yoloTrainBatch, setYoloTrainBatch] = useState(8);
  const [yoloTrainImgsz, setYoloTrainImgsz] = useState(640);
  const [yoloTrainValFrac, setYoloTrainValFrac] = useState(0.2);
  const [yoloTrainDevice, setYoloTrainDevice] = useState<"auto" | "cpu" | "mps" | "cuda">("auto");
  const [yoloTrainRes, setYoloTrainRes] = useState<YoloTrainResult | null>(null);
  const [yoloTrainProgress, setYoloTrainProgress] = useState<YoloTrainProgress | null>(null);
  const yoloPollTimer: MutableRefObject<number | null> = useRef(null);

  // Hybrid 모드 (U-Net + YOLO fusion)
  const [hybridImages, setHybridImages] = useState<File[]>([]);
  const [hybridGts, setHybridGts] = useState<File[]>([]);
  const [hybridUnetWeights, setHybridUnetWeights] = useState<File | null>(null);
  const [hybridYoloWeights, setHybridYoloWeights] = useState<File | null>(null);
  const [hybridYoloPath, setHybridYoloPath] = useState("");
  const [hybridHighThresh, setHybridHighThresh] = useState(0.75);
  const [hybridLowThresh, setHybridLowThresh] = useState(0.40);
  const [hybridBoundaryThk, setHybridBoundaryThk] = useState(3);
  const [hybridMinArea, setHybridMinArea] = useState(10);
  const [hybridBalance, setHybridBalance] = useState(0.25);
  const [hybridTauScale, setHybridTauScale] = useState(1.0);
  const [hybridYoloConf, setHybridYoloConf] = useState(0.15);
  const [hybridYoloIou, setHybridYoloIou] = useState(0.5);
  const [hybridYoloImgsz, setHybridYoloImgsz] = useState(640);
  const [hybridYoloRound, setHybridYoloRound] = useState(3);
  const [hybridYoloNoOverlap, setHybridYoloNoOverlap] = useState(true);
  const [hybridYoloRemoveEnclosed, setHybridYoloRemoveEnclosed] = useState(true);
  const [hybridEnclosedRatio, setHybridEnclosedRatio] = useState(0.9);
  const [hybridDevice, setHybridDevice] = useState("auto");
  const [hybridRes, setHybridRes] = useState<HybridResult | null>(null);
  const [hybridActiveIdx, setHybridActiveIdx] = useState(0);
  const [hybridDiagView, setHybridDiagView] = useState<"original" | "overlay" | "instances" | "numbered" | "mask" | "prob" | "high" | "close" | "boundary" | "separator" | "color" | "hist">("overlay");
  const [hybridLabelScale, setHybridLabelScale] = useState<number>(1.0);
  // 번호 모드 클라이언트 렌더링 — Page 레벨에서 캐시 (HybridView, 다운로드 패널 모두 공유)
  const [hybridNumberedSrc, setHybridNumberedSrc] = useState<string>("");
  const [hybridNumberedBusy, setHybridNumberedBusy] = useState(false);

  // 통계 (stats) — sweep 모드
  const [statsImages, setStatsImages] = useState<File[]>([]);
  const [statsGts, setStatsGts] = useState<File[]>([]);
  const [statsUnetWeights, setStatsUnetWeights] = useState<File | null>(null);
  const [statsYoloWeights, setStatsYoloWeights] = useState<File | null>(null);
  const [statsYoloPath, setStatsYoloPath] = useState("");
  const [statsSweepMode, setStatsSweepMode] = useState<"hybrid" | "yolo" | "unet" | "research">("hybrid");
  // 연구 모드 — 4개 파라미터 sweep 범위 (CSV)
  const [resSweepHigh, setResSweepHigh] = useState("0.65,0.70,0.75,0.80,0.85");
  const [resSweepLow, setResSweepLow] = useState("0.30,0.35,0.40,0.45,0.50");
  const [resSweepBalance, setResSweepBalance] = useState("0.15,0.20,0.25,0.30,0.35");
  const [resSweepRound, setResSweepRound] = useState("1,2,3,4,5");
  const [resEnableGrid, setResEnableGrid] = useState(true);
  const [resGridX, setResGridX] = useState<"high_thresh" | "low_thresh" | "balance_floor" | "yolo_round_strength">("low_thresh");
  const [resGridY, setResGridY] = useState<"high_thresh" | "low_thresh" | "balance_floor" | "yolo_round_strength">("high_thresh");
  const [researchJob, setResearchJob] = useState<ResearchJob | null>(null);
  const [researchRes, setResearchRes] = useState<ResearchResult | null>(null);
  const researchPollTimer: MutableRefObject<number | null> = useRef(null);
  const [researchActiveHeatmap, setResearchActiveHeatmap] = useState<keyof Metrics>("ParticleCountError");
  const [statsSweepParam, setStatsSweepParam] = useState<string>("high_thresh");
  const [statsSweepFrom, setStatsSweepFrom] = useState(0.5);
  const [statsSweepTo, setStatsSweepTo] = useState(0.95);
  const [statsSweepStep, setStatsSweepStep] = useState(0.05);
  // 고정값 (sweep 안 하는 파라미터)
  const [statsHigh, setStatsHigh] = useState(0.75);
  const [statsLow, setStatsLow] = useState(0.40);
  const [statsThickness, setStatsThickness] = useState(3);
  const [statsMinArea, setStatsMinArea] = useState(10);
  const [statsBalance, setStatsBalance] = useState(0.25);
  const [statsTauScale, setStatsTauScale] = useState(1.0);
  const [statsRound, setStatsRound] = useState(3);
  const [statsYoloNoOverlap, setStatsYoloNoOverlap] = useState(true);
  const [statsYoloConf, setStatsYoloConf] = useState(0.15);
  const [statsYoloIou, setStatsYoloIou] = useState(0.5);
  const [statsYoloImgsz, setStatsYoloImgsz] = useState(640);
  const [statsDevice, setStatsDevice] = useState("auto");
  const [statsRes, setStatsRes] = useState<StatsResult | null>(null);
  const [statsJob, setStatsJob] = useState<StatsJob | null>(null);
  const statsPollTimer: MutableRefObject<number | null> = useRef(null);
  const [statsActiveChart, setStatsActiveChart] = useState<keyof Metrics>("F1");

  // 증강 모드
  const [augImages, setAugImages] = useState<File[]>([]);
  const [augMasks, setAugMasks] = useState<File[]>([]);
  const [augNPer, setAugNPer] = useState(5);
  const [augBrightness, setAugBrightness] = useState(30);
  const [augContrast, setAugContrast] = useState(0.2);
  const [augNoise, setAugNoise] = useState(0);
  const [augSuffix, setAugSuffix] = useState("_aug{n}");
  const [augIncludeOrig, setAugIncludeOrig] = useState(false);
  const [augSeed, setAugSeed] = useState(0);
  const [augPreview, setAugPreview] = useState<{
    original_png: string; augmented_png: string;
    mask_original_png?: string; mask_augmented_png?: string;
    shape: [number, number];
  } | null>(null);
  const [augBusy, setAugBusy] = useState(false);

  // U-Net
  const [unetTab, setUnetTab] = useState<"train" | "predict">("predict");

  // 추론
  const [unetThreshold, setUnetThreshold] = useState(0.5);
  const [unetMinProb, setUnetMinProb] = useState(0.0);
  const [unetImgsz, setUnetImgsz] = useState<number | "">("");
  const [unetUseTrained, setUnetUseTrained] = useState(false);
  const [unetProbView, setUnetProbView] = useState<"color" | "gray" | "filtered_color" | "filtered_gray">("color");

  // 학습
  const [unetTrainImages, setUnetTrainImages] = useState<File[]>([]);
  const [unetTrainMasks, setUnetTrainMasks] = useState<File[]>([]);
  const [unetMethod, setUnetMethod] = useState<"U-Net" | "U-Net++">("U-Net++");
  const [unetEpochs, setUnetEpochs] = useState(30);
  const [unetBatch, setUnetBatch] = useState(4);
  const [unetLr, setUnetLr] = useState(0.001);
  const [unetTrainImgSize, setUnetTrainImgSize] = useState(256);
  const [unetValFrac, setUnetValFrac] = useState(0.2);
  const [unetAugment, setUnetAugment] = useState(true);
  const [unetUseDice, setUnetUseDice] = useState(true);
  const [unetDiceWeight, setUnetDiceWeight] = useState(0.5);
  const [unetWmapMode, setUnetWmapMode] = useState<"none" | "distance" | "intensity">("none");
  const [unetW0, setUnetW0] = useState(10);
  const [unetSigma, setUnetSigma] = useState(7);
  const [unetGamma, setUnetGamma] = useState(2);
  const [unetTrainRes, setUnetTrainRes] = useState<UnetTrainResult | null>(null);
  const [unetTrainProgress, setUnetTrainProgress] = useState<UnetTrainProgress | null>(null);
  const unetPollTimer: MutableRefObject<number | null> = useRef(null);
  const [unetDevice, setUnetDevice] = useState<"auto" | "cpu" | "mps" | "cuda">("auto");

  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [legacyRes, setLegacyRes] = useState<LegacyResult | null>(null);
  const [yoloRes, setYoloRes] = useState<YoloResult | null>(null);
  const [unetRes, setUnetRes] = useState<UnetResult | null>(null);
  const [yoloActiveIdx, setYoloActiveIdx] = useState(0);

  // ── 추론 진행 추정 (서버측 잡 progress 가 없는 단일 POST endpoint 용) ──
  const [inferStartMs, setInferStartMs] = useState<number | null>(null);
  const [inferTick, setInferTick] = useState(0);
  const [inferTotal, setInferTotal] = useState<number>(0);
  const [inferLabel, setInferLabel] = useState<string>("");
  useEffect(() => {
    if (inferStartMs === null) return;
    const id = window.setInterval(() => setInferTick(t => t + 1), 300);
    return () => window.clearInterval(id);
  }, [inferStartMs]);

  // ── Hybrid 번호 모드 클라이언트 렌더링 ──
  // active item / labelScale 변경 시 짧은 디바운스 후 재렌더링.
  useEffect(() => {
    const res = hybridRes;
    if (!res) { setHybridNumberedSrc(""); return; }
    const it = res.results[Math.min(hybridActiveIdx, res.results.length - 1)];
    if (!it || !it.instances_png || !(it.particles && it.particles.length > 0)) {
      setHybridNumberedSrc("");
      return;
    }
    let cancelled = false;
    setHybridNumberedBusy(true);
    const id = window.setTimeout(async () => {
      const url = await renderNumberedImage(it.instances_png, it.particles ?? [], hybridLabelScale);
      if (!cancelled) {
        setHybridNumberedSrc(url ?? "");
        setHybridNumberedBusy(false);
      }
    }, 40);
    return () => { cancelled = true; window.clearTimeout(id); setHybridNumberedBusy(false); };
  }, [hybridRes, hybridActiveIdx, hybridLabelScale]);

  // ── Hybrid 원본 이미지 (드롭존에서 받은 File → blob URL) ──
  const [hybridOriginals, setHybridOriginals] = useState<Record<string, string>>({});
  useEffect(() => {
    const m: Record<string, string> = {};
    hybridImages.forEach(f => {
      try { m[f.name] = URL.createObjectURL(f); } catch { /* ignore */ }
    });
    setHybridOriginals(m);
    return () => {
      Object.values(m).forEach(u => { try { URL.revokeObjectURL(u); } catch { /* ignore */ } });
    };
  }, [hybridImages]);

  function reset() {
    setError(null);
    setLegacyRes(null);
    setYoloRes(null);
    setUnetRes(null);
    setHybridRes(null);
    setStatsRes(null);
    setStatsJob(null);
    setResearchRes(null);
    setResearchJob(null);
    setYoloActiveIdx(0);
    setLegacyActiveIdx(0);
    setUnetActiveIdx(0);
    setHybridActiveIdx(0);
  }

  function stopPolling() {
    if (unetPollTimer.current !== null) {
      window.clearTimeout(unetPollTimer.current);
      unetPollTimer.current = null;
    }
    if (yoloPollTimer.current !== null) {
      window.clearTimeout(yoloPollTimer.current);
      yoloPollTimer.current = null;
    }
    if (statsPollTimer.current !== null) {
      window.clearTimeout(statsPollTimer.current);
      statsPollTimer.current = null;
    }
    if (researchPollTimer.current !== null) {
      window.clearTimeout(researchPollTimer.current);
      researchPollTimer.current = null;
    }
  }

  async function pollResearchJob(jobId: string) {
    try {
      const r = await fetch(`${API_BASE}/api/run/research/${jobId}`);
      if (!r.ok) { setError(`${r.status}: ${await r.text()}`); setRunning(false); return; }
      const job: ResearchJob = await r.json();
      setResearchJob(job);
      if (job.status === "done" && job.result) {
        setResearchRes(job.result);
        setRunning(false);
        return;
      }
      if (job.status === "error") {
        setError(job.error || "연구 잡 에러");
        setRunning(false);
        return;
      }
      researchPollTimer.current = window.setTimeout(() => pollResearchJob(jobId), 700);
    } catch (e: any) {
      setError(e?.message ?? String(e));
      setRunning(false);
    }
  }

  async function pollStatsJob(jobId: string) {
    try {
      const r = await fetch(`${API_BASE}/api/run/stats/${jobId}`);
      if (!r.ok) {
        setError(`${r.status}: ${await r.text()}`);
        setRunning(false);
        return;
      }
      const job: StatsJob = await r.json();
      setStatsJob(job);
      if (job.status === "done" && job.result) {
        setStatsRes(job.result);
        setRunning(false);
        return;
      }
      if (job.status === "error") {
        setError(job.error || "통계 잡 에러");
        setRunning(false);
        return;
      }
      // 진행 중 — 다시 폴링
      statsPollTimer.current = window.setTimeout(() => pollStatsJob(jobId), 700);
    } catch (e: any) {
      setError(e?.message ?? String(e));
      setRunning(false);
    }
  }

  // 컴포넌트 unmount 시 폴링 정리
  useEffect(() => () => stopPolling(), []);

  async function runAugPreview() {
    if (augImages.length === 0) {
      setError("미리보기를 위해 이미지 1장을 먼저 추가해주세요.");
      return;
    }
    setError(null);
    setAugBusy(true);
    try {
      const fd = new FormData();
      fd.append("image", augImages[0]);
      // 매칭되는 마스크 있으면 같이 보냄 (같은 stem 검색)
      const matchMask = augMasks.find(m => {
        const i = augImages[0].name.replace(/\.[^/.]+$/, "");
        const j = m.name.replace(/\.[^/.]+$/, "");
        const strip = (s: string) =>
          s.replace(/_(image|img|orig|label|mask|gt)$/i, "");
        return j === i || strip(j) === strip(i);
      });
      if (matchMask) fd.append("mask", matchMask);
      // 미리보기는 max 값을 그대로 적용 (눈에 잘 보이는 결정적 변형)
      fd.append("brightness", String(augBrightness));
      fd.append("contrast", String(1 + augContrast));
      fd.append("noise_std", String(augNoise));
      fd.append("seed", String(augSeed));
      const r = await fetch(`${API_BASE}/api/augment/preview`, {
        method: "POST", body: fd,
      });
      if (!r.ok) throw new Error(`${r.status}: ${await r.text()}`);
      setAugPreview(await r.json());
    } catch (e: any) {
      setError(e?.message ?? String(e));
    } finally {
      setAugBusy(false);
    }
  }

  async function runAugmentDownload() {
    if (augImages.length === 0) {
      setError("이미지를 1장 이상 추가해주세요.");
      return;
    }
    if (!augSuffix.includes("{n}")) {
      setError("파일명 접미사에 {n} 자리표시자가 있어야 합니다.");
      return;
    }
    setError(null);
    setAugBusy(true);
    try {
      const fd = new FormData();
      for (const f of augImages) fd.append("images", f);
      for (const f of augMasks) fd.append("masks", f);
      fd.append("n_per_image", String(augNPer));
      fd.append("brightness_max", String(augBrightness));
      fd.append("contrast_range", String(augContrast));
      fd.append("noise_std_max", String(augNoise));
      fd.append("suffix_template", augSuffix);
      fd.append("include_original", String(augIncludeOrig));
      fd.append("seed", String(augSeed));
      const r = await fetch(`${API_BASE}/api/augment/run`, {
        method: "POST", body: fd,
      });
      if (!r.ok) throw new Error(`${r.status}: ${await r.text()}`);
      // 응답을 blob 으로 받아 다운로드 트리거
      const blob = await r.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "augmented.zip";
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    } catch (e: any) {
      setError(e?.message ?? String(e));
    } finally {
      setAugBusy(false);
    }
  }

  async function runYoloTrain() {
    if (!yoloDatasetPath.trim() && (yoloTrainImages.length < 2 || yoloTrainMasks.length === 0)) {
      setError("학습 이미지 2장 이상 + 정답 마스크 또는 데이터셋 경로가 필요합니다.");
      return;
    }
    stopPolling();
    setError(null);
    setRunning(true);
    setYoloTrainRes(null);
    setYoloTrainProgress(null);
    try {
      const fd = new FormData();
      // dataset_path 가 있으면 multipart 업로드 생략 (서버 디스크 폴더 사용)
      if (yoloDatasetPath.trim()) {
        fd.append("dataset_path", yoloDatasetPath.trim());
      } else {
        for (const f of yoloTrainImages) fd.append("images", f);
        for (const f of yoloTrainMasks) fd.append("masks", f);
      }
      fd.append("base_arch", yoloBaseArch);
      fd.append("epochs", String(yoloTrainEpochs));
      fd.append("batch_size", String(yoloTrainBatch));
      fd.append("imgsz", String(yoloTrainImgsz));
      fd.append("val_fraction", String(yoloTrainValFrac));
      fd.append("device", yoloTrainDevice);
      const r = await fetch(`${API_BASE}/api/run/yolo/train`, {
        method: "POST", body: fd,
      });
      if (!r.ok) throw new Error(`${r.status}: ${await r.text()}`);
      const start: { job_id: string; n_pairs: number; total_epochs: number } = await r.json();

      const poll = async () => {
        try {
          const pr = await fetch(`${API_BASE}/api/run/yolo/train/${start.job_id}`);
          if (!pr.ok) throw new Error(`${pr.status}`);
          const j: YoloTrainProgress = await pr.json();
          setYoloTrainProgress(j);
          if (j.status === "done" && j.result) {
            setYoloTrainRes(j.result);
            setYoloUseTrained(true);
            setRunning(false);
            return;
          }
          if (j.status === "error") {
            setError(j.error || "학습 중 오류");
            setRunning(false);
            return;
          }
        } catch {
          /* network blip */
        }
        yoloPollTimer.current = window.setTimeout(poll, 1500);
      };
      poll();
    } catch (e: any) {
      setError(e?.message ?? String(e));
      setRunning(false);
    }
  }

  async function runUnetTrain() {
    if (unetTrainImages.length === 0 || unetTrainMasks.length === 0) {
      setError("학습 이미지와 정답 마스크를 1장 이상 추가해주세요.");
      return;
    }
    stopPolling();
    setError(null);
    setRunning(true);
    setUnetTrainRes(null);
    setUnetTrainProgress(null);
    try {
      const fd = new FormData();
      for (const f of unetTrainImages) fd.append("images", f);
      for (const f of unetTrainMasks) fd.append("masks", f);
      fd.append("method", unetMethod);
      fd.append("epochs", String(unetEpochs));
      fd.append("batch_size", String(unetBatch));
      fd.append("lr", String(unetLr));
      fd.append("image_size", String(unetTrainImgSize));
      fd.append("val_fraction", String(unetValFrac));
      fd.append("augment", String(unetAugment));
      fd.append("use_dice", String(unetUseDice));
      fd.append("dice_weight", String(unetDiceWeight));
      fd.append("wmap_mode", unetWmapMode);
      fd.append("w0", String(unetW0));
      fd.append("sigma", String(unetSigma));
      fd.append("gamma", String(unetGamma));
      fd.append("device", unetDevice);
      const r = await fetch(`${API_BASE}/api/run/unet/train`, {
        method: "POST", body: fd,
      });
      if (!r.ok) throw new Error(`${r.status}: ${await r.text()}`);
      const start: { job_id: string; n_pairs: number; total_epochs: number } = await r.json();

      const poll = async () => {
        try {
          const pr = await fetch(`${API_BASE}/api/run/unet/train/${start.job_id}`);
          if (!pr.ok) throw new Error(`${pr.status}`);
          const j: UnetTrainProgress = await pr.json();
          setUnetTrainProgress(j);
          if (j.status === "done" && j.result) {
            setUnetTrainRes(j.result);
            setUnetUseTrained(true);
            setRunning(false);
            return;  // 폴링 중단
          }
          if (j.status === "error") {
            setError(j.error || "학습 중 오류");
            setRunning(false);
            return;
          }
        } catch {
          // 네트워크 일시 실패 — 무시하고 다음 틱에 재시도
        }
        unetPollTimer.current = window.setTimeout(poll, 1500);
      };
      poll();
    } catch (e: any) {
      setError(e?.message ?? String(e));
      setRunning(false);
    }
  }

  async function run() {
    setError(null);
    setRunning(true);
    setLegacyRes(null);
    setYoloRes(null);
    setUnetRes(null);
    setHybridRes(null);
    setStatsRes(null);
    // 추론 진행 추정 시작 — 모드별 이미지 수 + 라벨
    const startTotals: Record<string, number> = {
      hybrid: hybridImages.length,
      yolo: yoloImages.length,
      unet: predictImages.length,
      legacy: predictImages.length,
    };
    const startLabels: Record<string, string> = {
      hybrid: "Hybrid 추론",
      yolo: "YOLO 추론",
      unet: "U-Net 추론",
      legacy: "레거시 후처리",
    };
    setInferTotal(startTotals[mode] ?? 0);
    setInferLabel(startLabels[mode] ?? "처리");
    setInferStartMs(mode === "stats" ? null : Date.now());
    setInferTick(0);
    try {
      const fd = new FormData();
      let url = "";

      if (mode === "stats") {
        if (statsImages.length === 0) throw new Error("이미지를 1장 이상 추가.");
        if (statsGts.length === 0) throw new Error("정답 마스크 필요 (평가 지표 산출용).");
        const isResearch = statsSweepMode === "research";
        const needUnet = isResearch || statsSweepMode === "hybrid" || statsSweepMode === "unet";
        const needYolo = isResearch || statsSweepMode === "hybrid" || statsSweepMode === "yolo";
        if (needUnet && !statsUnetWeights) throw new Error("U-Net 가중치(.pt) 필요.");
        if (needYolo && !statsYoloWeights && !statsYoloPath.trim())
          throw new Error("YOLO 가중치(.pt) 또는 서버 경로 필요.");

        if (isResearch) {
          // ─── 연구 모드 — 별도 endpoint ───
          url = `${API_BASE}/api/run/research`;
          if (statsUnetWeights) fd.append("unet_weights", statsUnetWeights);
          if (statsYoloPath.trim()) fd.append("yolo_weights_path", statsYoloPath.trim());
          else if (statsYoloWeights) fd.append("yolo_weights", statsYoloWeights);
          for (const f of statsImages) fd.append("images", f);
          for (const f of statsGts) fd.append("gts", f);
          // 운용값 (sweep 중심)
          fd.append("high_thresh", String(statsHigh));
          fd.append("low_thresh", String(statsLow));
          fd.append("boundary_thickness", String(statsThickness));
          fd.append("min_anchor_area", String(statsMinArea));
          fd.append("balance_floor", String(statsBalance));
          fd.append("tau_scale", String(statsTauScale));
          fd.append("yolo_round_strength", String(statsRound));
          fd.append("yolo_remove_overlaps", String(statsYoloNoOverlap));
          fd.append("yolo_conf", String(statsYoloConf));
          fd.append("yolo_iou", String(statsYoloIou));
          fd.append("yolo_imgsz", String(statsYoloImgsz));
          fd.append("device", statsDevice);
          // sweep 범위
          fd.append("sweep_high_thresh", resSweepHigh);
          fd.append("sweep_low_thresh", resSweepLow);
          fd.append("sweep_balance_floor", resSweepBalance);
          fd.append("sweep_yolo_round_strength", resSweepRound);
          // 격자
          fd.append("enable_grid", String(resEnableGrid));
          fd.append("grid_x_param", resGridX);
          fd.append("grid_y_param", resGridY);
        } else {

        url = `${API_BASE}/api/run/stats`;
        if (needUnet && statsUnetWeights) fd.append("unet_weights", statsUnetWeights);
        if (needYolo) {
          if (statsYoloPath.trim()) {
            fd.append("yolo_weights_path", statsYoloPath.trim());
          } else if (statsYoloWeights) {
            fd.append("yolo_weights", statsYoloWeights);
          }
        }
        for (const f of statsImages) fd.append("images", f);
        for (const f of statsGts) fd.append("gts", f);
        fd.append("sweep_mode", statsSweepMode);
        fd.append("sweep_param", statsSweepParam);
        fd.append("sweep_from", String(statsSweepFrom));
        fd.append("sweep_to", String(statsSweepTo));
        fd.append("sweep_step", String(statsSweepStep));
        fd.append("high_thresh", String(statsHigh));
        fd.append("low_thresh", String(statsLow));
        fd.append("boundary_thickness", String(statsThickness));
        fd.append("min_anchor_area", String(statsMinArea));
        fd.append("balance_floor", String(statsBalance));
        fd.append("tau_scale", String(statsTauScale));
        fd.append("yolo_round_strength", String(statsRound));
        fd.append("yolo_remove_overlaps", String(statsYoloNoOverlap));
        fd.append("yolo_conf", String(statsYoloConf));
        fd.append("yolo_iou", String(statsYoloIou));
        fd.append("yolo_imgsz", String(statsYoloImgsz));
        fd.append("device", statsDevice);
        }  // end of isResearch=false (stats)
      } else if (mode === "hybrid") {
        if (hybridImages.length === 0) throw new Error("이미지를 1장 이상 추가해주세요.");
        if (!hybridUnetWeights) throw new Error("U-Net 가중치(.pt) 를 선택해주세요.");
        if (!hybridYoloWeights && !hybridYoloPath.trim())
          throw new Error("YOLO 가중치(.pt) 또는 서버 경로를 입력해주세요.");
        url = `${API_BASE}/api/run/hybrid`;
        fd.append("unet_weights", hybridUnetWeights);
        if (hybridYoloPath.trim()) {
          fd.append("yolo_weights_path", hybridYoloPath.trim());
        } else if (hybridYoloWeights) {
          fd.append("yolo_weights", hybridYoloWeights);
        }
        for (const f of hybridImages) fd.append("images", f);
        for (const f of hybridGts) fd.append("gts", f);
        fd.append("high_thresh", String(hybridHighThresh));
        fd.append("low_thresh", String(hybridLowThresh));
        fd.append("boundary_thickness", String(hybridBoundaryThk));
        fd.append("min_anchor_area", String(hybridMinArea));
        fd.append("balance_floor", String(hybridBalance));
        fd.append("tau_scale", String(hybridTauScale));
        fd.append("yolo_conf", String(hybridYoloConf));
        fd.append("yolo_iou", String(hybridYoloIou));
        fd.append("yolo_imgsz", String(hybridYoloImgsz));
        fd.append("yolo_round_strength", String(hybridYoloRound));
        fd.append("yolo_remove_overlaps", String(hybridYoloNoOverlap));
        fd.append("yolo_remove_enclosed", String(hybridYoloRemoveEnclosed));
        fd.append("enclosed_ratio", String(hybridEnclosedRatio));
        fd.append("device", hybridDevice);
      } else if (mode === "legacy") {
        if (predictImages.length === 0) throw new Error("이미지를 1장 이상 추가해주세요.");
        url = `${API_BASE}/api/run/legacy`;
        for (const f of predictImages) fd.append("images", f);
        for (const f of predictGts) fd.append("gts", f);
        fd.append("invert", String(invert));
        fd.append("min_size", String(minSize));
        fd.append("morph_kernel", String(morphKernel));
        fd.append("use_watershed", String(useWatershed));
        fd.append("watershed_min_distance", String(watershedDist));
      } else if (mode === "yolo") {
        const useTrained = yoloUseTrained && yoloTrainRes;
        if (!useTrained && !weightsFile)
          throw new Error("가중치(.pt) 를 선택하거나 학습된 모델을 사용해주세요.");
        if (yoloImages.length === 0) throw new Error("이미지를 1장 이상 추가해주세요.");
        url = `${API_BASE}/api/run/yolo`;
        if (useTrained) {
          fd.append("model_id", yoloTrainRes!.model_id);
        } else if (weightsFile) {
          fd.append("weights", weightsFile);
        }
        for (const f of yoloImages) fd.append("images", f);
        for (const f of yoloGts) fd.append("gts", f);
        fd.append("conf", String(yoloConf));
        fd.append("iou", String(yoloIou));
        fd.append("imgsz", String(yoloImgsz));
        fd.append("smooth", String(yoloSmooth));
        fd.append("round_strength", String(yoloRound));
        fd.append("remove_overlaps", String(yoloNoOverlap));
        fd.append("device", yoloDevice);
      } else {
        // U-Net 추론
        const useTrained = unetUseTrained && unetTrainRes;
        if (!useTrained && !weightsFile)
          throw new Error("가중치(.pt) 를 선택하거나 학습된 모델을 사용해주세요.");
        if (predictImages.length === 0)
          throw new Error("이미지를 1장 이상 추가해주세요.");
        url = `${API_BASE}/api/run/unet`;
        if (useTrained) {
          fd.append("model_id", unetTrainRes!.model_id);
        } else if (weightsFile) {
          fd.append("weights", weightsFile);
        }
        for (const f of predictImages) fd.append("images", f);
        for (const f of predictGts) fd.append("gts", f);
        fd.append("threshold", String(unetThreshold));
        fd.append("min_prob", String(unetMinProb));
        if (unetImgsz !== "") fd.append("image_size", String(unetImgsz));
      }

      const r = await fetch(url, { method: "POST", body: fd });
      if (!r.ok) throw new Error(`${r.status}: ${await r.text()}`);
      const json = await r.json();
      if (mode === "legacy") { setLegacyRes(json); setLegacyActiveIdx(0); setRunning(false); }
      else if (mode === "yolo") { setYoloRes(json); setYoloActiveIdx(0); setRunning(false); }
      else if (mode === "hybrid") { setHybridRes(json); setHybridActiveIdx(0); setRunning(false); }
      else if (mode === "stats" && statsSweepMode === "research") {
        // 연구 모드 — polling /api/run/research
        setResearchJob({
          job_id: json.job_id,
          status: "pending",
          stage: "queued",
          progress: 0,
          current_step: 0,
          total_steps: json.total_steps ?? 0,
          current_param: null,
          current_value: null,
          current_image: null,
          elapsed_sec: 0,
          eta_sec: null,
          n_images: json.n_images,
          n_with_gt: json.n_with_gt,
          n_sweep_params: json.n_sweep_params,
          has_grid: json.has_grid,
          error: null,
          result: null,
        });
        pollResearchJob(json.job_id);
      }
      else if (mode === "stats") {
        // job_id 받음 → polling 시작 (setRunning 은 polling 종료 시 false)
        setStatsJob({
          job_id: json.job_id,
          status: "pending",
          stage: "queued",
          progress: 0,
          current_step: 0,
          total_steps: json.total_steps ?? 0,
          current_value: null,
          current_image: null,
          elapsed_sec: 0,
          eta_sec: null,
          sweep_mode: json.sweep_mode,
          sweep_param: json.sweep_param,
          n_sweep_values: json.n_sweep_values,
          n_images: json.n_images,
          n_with_gt: json.n_with_gt,
          error: null,
          result: null,
        });
        pollStatsJob(json.job_id);
      }
      else { setUnetRes(json); setUnetActiveIdx(0); setRunning(false); }
    } catch (e: any) {
      setError(e?.message ?? String(e));
      setRunning(false);
    } finally {
      setInferStartMs(null);
    }
  }

  const canRun =
    !running &&
    (mode === "legacy"
      ? predictImages.length > 0
      : mode === "yolo" && yoloTab === "predict"
        ? (!!weightsFile || (yoloUseTrained && !!yoloTrainRes))
          && yoloImages.length > 0
        : mode === "unet" && unetTab === "predict"
          ? (!!weightsFile || (unetUseTrained && !!unetTrainRes))
            && predictImages.length > 0
          : mode === "hybrid"
            ? hybridImages.length > 0 && !!hybridUnetWeights
              && (!!hybridYoloWeights || hybridYoloPath.trim() !== "")
            : mode === "stats"
              ? (() => {
                  const isResearch = statsSweepMode === "research";
                  const needU = isResearch || statsSweepMode === "hybrid" || statsSweepMode === "unet";
                  const needY = isResearch || statsSweepMode === "hybrid" || statsSweepMode === "yolo";
                  const okU = !needU || !!statsUnetWeights;
                  const okY = !needY || (!!statsYoloWeights || statsYoloPath.trim() !== "");
                  if (isResearch) {
                    return statsImages.length > 0 && statsGts.length > 0 && okU && okY;
                  }
                  return statsImages.length > 0 && statsGts.length > 0
                    && okU && okY
                    && statsSweepStep > 0 && statsSweepTo > statsSweepFrom;
                })()
              : false);

  const canTrainUnet =
    !running && unetTrainImages.length > 0 && unetTrainMasks.length > 0;
  const canTrainYolo =
    !running && (yoloDatasetPath.trim() !== "" ||
      (yoloTrainImages.length >= 2 && yoloTrainMasks.length > 0));

  return (
    <div className="min-h-screen text-[#14161b] font-sans overflow-x-hidden">
      <header className="sticky top-0 z-30 bg-white border-b border-[#f1f1ef]">
        <div className="w-full px-5 lg:px-6 py-3 flex items-center justify-between gap-4 flex-wrap">
          <div className="flex items-center gap-3">
            <div
              aria-hidden
              className="w-8 h-8 rounded-lg flex items-center justify-center text-white font-bold text-[15px] bg-[#191919]"
              style={{ letterSpacing: "-0.02em" }}
            >
              P
            </div>
            <div className="leading-tight">
              <h1 className="text-[17px] font-semibold tracking-tight text-[#191919] flex items-baseline gap-1.5">
                Particle Lab
                <span className="text-[10px] font-semibold text-[#9ca3af] align-middle">v2</span>
              </h1>
              <p className="text-[11.5px] text-[#9ca3af]">SEM 입자 분할 콘솔</p>
            </div>
          </div>
          <div className="flex items-center gap-3 text-[11.5px]">
            <span
              className="inline-flex items-center gap-1.5 text-[#6b7280]"
              title={
                defaultLoadStatus === "ok"
                  ? "basic_pt/ 의 U-Net + YOLO 기본 모델이 자동 로드되었습니다"
                  : defaultLoadStatus === "fail"
                    ? "기본 모델 로드 실패 — 직접 .pt 파일을 올려주세요"
                    : "기본 모델 로드 중..."
              }
            >
              <span
                className={
                  "w-1.5 h-1.5 rounded-full " +
                  (defaultLoadStatus === "ok"
                    ? "bg-[#009f50]"
                    : defaultLoadStatus === "fail"
                      ? "bg-[#e8483c]"
                      : "bg-[#9ca3af] dot-pulse")
                }
              />
              <span className="hidden sm:inline">
                기본 모델 {defaultLoadStatus === "ok" ? "로드됨" : defaultLoadStatus === "fail" ? "없음" : "로드 중"}
              </span>
            </span>
            <span className="text-[#e6e6e2]">|</span>
            <span className="inline-flex items-center gap-1.5 text-[#6b7280]">
              <span
                className={
                  "w-1.5 h-1.5 rounded-full " +
                  (health === "정상"
                    ? "bg-[#009f50]"
                    : health === "끊김"
                      ? "bg-[#e8483c]"
                      : "bg-[#9ca3af] dot-pulse")
                }
              />
              {health === "정상" ? "백엔드 정상" : health === "끊김" ? "백엔드 끊김" : "확인 중"}
            </span>
            <code className="text-[10.5px] text-[#9ca3af] font-mono hidden md:inline">{API_BASE}</code>
          </div>
        </div>
      </header>

      <div
        className="w-full px-4 lg:px-6 py-4 lg:py-5 grid grid-cols-1 lg:grid-cols-[300px_minmax(0,1fr)_var(--right-w)] gap-4 lg:gap-5"
        style={{ ["--right-w" as any]: `${rightW}px` }}
      >
        {/* ── 좌측 ── */}
        <section className="space-y-3 min-w-0">
          {/* v2: mode selector 제거 — Hybrid 추론 전용 */}

          {/* YOLO / U-Net 사브탭 (v2 에서는 사용 안 함, 코드 유지) */}
          {(mode === "unet" || mode === "yolo") && (
            <div className="app-card p-2">
              <div className="grid grid-cols-2 gap-1 text-xs">
                {([
                  ["train", "학습"],
                  ["predict", "추론"],
                ] as const).map(([id, label]) => {
                  const cur = mode === "unet" ? unetTab : yoloTab;
                  const setCur = mode === "unet" ? setUnetTab : setYoloTab;
                  return (
                    <button
                      key={id}
                      onClick={() => setCur(id as "train" | "predict")}
                      className={
                        "px-3 py-1.5 rounded font-semibold " +
                        (cur === id
                          ? "bg-[#1f5fd6] text-white"
                          : "bg-gray-100 text-gray-700 hover:bg-gray-200")
                      }
                    >
                      {label}
                    </button>
                  );
                })}
              </div>
            </div>
          )}

          <div className="app-card p-4 space-y-3">
            <h2 className="text-sm font-semibold">입력 파일</h2>

            {/* 가중치 슬롯 (YOLO 추론) */}
            {mode === "yolo" && yoloTab === "predict" && (
              <>
                {yoloTrainRes && (
                  <label className="flex items-center justify-between text-xs
                                    bg-blue-50 border border-blue-200 rounded p-2">
                    <span>
                      <b>방금 학습한 모델 사용</b>
                      <br />
                      <span className="text-[10px] text-gray-600">
                        {yoloTrainRes.base_arch} · {yoloTrainRes.imgsz}px ·
                        {yoloTrainRes.history.length} 에포크 · {yoloTrainRes.n_pairs}쌍
                      </span>
                    </span>
                    <input
                      type="checkbox"
                      checked={yoloUseTrained}
                      onChange={(e) => setYoloUseTrained(e.target.checked)}
                    />
                  </label>
                )}
                {!yoloUseTrained && (
                  <SingleFileDropzone
                    label="YOLO 가중치 (.pt)"
                    hint="ultralytics 학습 결과 best.pt"
                    file={weightsFile}
                    onChange={setWeightsFile}
                    accept=".pt,application/octet-stream"
                  />
                )}
              </>
            )}
            {mode === "unet" && unetTab === "predict" && (
              <>
                {unetTrainRes && (
                  <label className="flex items-center justify-between text-xs
                                    bg-blue-50 border border-blue-200 rounded p-2">
                    <span>
                      <b>방금 학습한 모델 사용</b>
                      <br />
                      <span className="text-[10px] text-gray-600">
                        {unetTrainRes.method} · {unetTrainRes.image_size}px
                        · best_val {unetTrainRes.best_val_loss.toFixed(4)}
                      </span>
                    </span>
                    <input
                      type="checkbox"
                      checked={unetUseTrained}
                      onChange={(e) => setUnetUseTrained(e.target.checked)}
                    />
                  </label>
                )}
                {!unetUseTrained && (
                  <SingleFileDropzone
                    label="U-Net 모델 (.pt)"
                    hint="unet_trainer.serialize_model 형식"
                    file={weightsFile}
                    onChange={setWeightsFile}
                    accept=".pt,application/octet-stream"
                  />
                )}
              </>
            )}

            {/* 입력 이미지 / 마스크 슬롯 */}
            {mode === "yolo" && yoloTab === "predict" ? (
              <>
                <MultiFileDropzone
                  files={yoloImages}
                  setFiles={setYoloImages}
                  label="추론할 이미지"
                  emptyText="이미지를 끌어다 놓거나 클릭"
                  emptyHint="여러 장 선택 가능"
                />
                <MultiFileDropzone
                  files={yoloGts}
                  setFiles={setYoloGts}
                  label="정답 마스크 (선택)"
                  emptyText="GT 마스크 끌어다 놓거나 클릭"
                  emptyHint="파일명 일치 시 자동 매칭"
                />
              </>
            ) : mode === "yolo" && yoloTab === "train" ? (
              <>
                <div style={{
                  marginBottom: 12, padding: 10, background: "#f0f8ff",
                  border: "1px solid #b0d4ff", borderRadius: 6, fontSize: 12,
                }}>
                  <div style={{ fontWeight: 600, marginBottom: 4 }}>
                    💡 서버 디스크 경로 (선택 — 643장 multipart 업로드 회피)
                  </div>
                  <input
                    type="text"
                    value={yoloDatasetPath}
                    onChange={(e) => setYoloDatasetPath(e.target.value)}
                    placeholder="예: /workspace/dataset (images/ + masks/ 또는 *_image / *_label)"
                    style={{
                      width: "100%", padding: "6px 8px", fontSize: 12,
                      border: "1px solid #ccc", borderRadius: 4, fontFamily: "monospace",
                    }}
                  />
                  <div style={{ marginTop: 4, color: "#666", fontSize: 11 }}>
                    경로를 입력하면 아래 파일 업로드는 무시됨. 비워두면 평소처럼 업로드 모드.
                  </div>
                </div>
                <MultiFileDropzone
                  files={yoloTrainImages}
                  setFiles={setYoloTrainImages}
                  label={yoloDatasetPath.trim() ? "학습 이미지 (디스크 경로 사용 중 — 무시됨)" : "학습 이미지 (2장 이상)"}
                  emptyText="원본 SEM 이미지 끌어다 놓기"
                  emptyHint="train/val 분할에 최소 2장 필요"
                />
                <MultiFileDropzone
                  files={yoloTrainMasks}
                  setFiles={setYoloTrainMasks}
                  label="정답 이진 마스크"
                  emptyText="GT 마스크 끌어다 놓기"
                  emptyHint="파일명 매칭. 자동으로 폴리곤 라벨로 변환"
                />
              </>
            ) : mode === "unet" && unetTab === "train" ? (
              <>
                <MultiFileDropzone
                  files={unetTrainImages}
                  setFiles={setUnetTrainImages}
                  label="학습 이미지"
                  emptyText="원본 SEM 이미지 끌어다 놓기"
                  emptyHint="여러 장 선택 가능"
                />
                <MultiFileDropzone
                  files={unetTrainMasks}
                  setFiles={setUnetTrainMasks}
                  label="정답 이진 마스크"
                  emptyText="GT 마스크 끌어다 놓기"
                  emptyHint="파일명 매칭. 예: foo_image.png ↔ foo_label.png"
                />
              </>
            ) : mode === "hybrid" ? (
              <>
                <SingleFileDropzone
                  label="U-Net 가중치 (.pt)"
                  hint="unet_trainer.serialize_model 형식"
                  file={hybridUnetWeights}
                  onChange={setHybridUnetWeights}
                  accept=".pt,application/octet-stream"
                  isDefault={!!defaultUnetFile && hybridUnetWeights === defaultUnetFile}
                  onRevertToDefault={defaultUnetFile && hybridUnetWeights !== defaultUnetFile
                    ? () => setHybridUnetWeights(defaultUnetFile)
                    : undefined}
                />
                <SingleFileDropzone
                  label="YOLO 가중치 (.pt)"
                  hint="ultralytics 학습 결과 best.pt"
                  file={hybridYoloWeights}
                  onChange={setHybridYoloWeights}
                  accept=".pt,application/octet-stream"
                  isDefault={!!defaultYoloFile && hybridYoloWeights === defaultYoloFile}
                  onRevertToDefault={defaultYoloFile && hybridYoloWeights !== defaultYoloFile
                    ? () => setHybridYoloWeights(defaultYoloFile)
                    : undefined}
                />
                <MultiFileDropzone
                  files={hybridImages}
                  setFiles={setHybridImages}
                  label="추론할 이미지"
                  emptyText="이미지 끌어다 놓거나 클릭"
                  emptyHint="여러 장 가능"
                />
                <MultiFileDropzone
                  files={hybridGts}
                  setFiles={setHybridGts}
                  label="정답 마스크 (선택)"
                  emptyText="GT 마스크 끌어다 놓기"
                  emptyHint="파일명 매칭 시 자동 지표"
                />
              </>
            ) : mode === "stats" ? (
              <>
                {(statsSweepMode === "hybrid" || statsSweepMode === "unet" || statsSweepMode === "research") && (
                  <SingleFileDropzone
                    label="U-Net 가중치 (.pt)"
                    hint="unet_trainer.serialize_model 형식"
                    file={statsUnetWeights}
                    onChange={setStatsUnetWeights}
                    accept=".pt,application/octet-stream"
                    isDefault={!!defaultUnetFile && statsUnetWeights === defaultUnetFile}
                    onRevertToDefault={defaultUnetFile && statsUnetWeights !== defaultUnetFile
                      ? () => setStatsUnetWeights(defaultUnetFile)
                      : undefined}
                  />
                )}
                {(statsSweepMode === "hybrid" || statsSweepMode === "yolo" || statsSweepMode === "research") && (
                  <SingleFileDropzone
                    label="YOLO 가중치 (.pt)"
                    hint="ultralytics 학습 결과 best.pt"
                    file={statsYoloWeights}
                    onChange={setStatsYoloWeights}
                    accept=".pt,application/octet-stream"
                    isDefault={!!defaultYoloFile && statsYoloWeights === defaultYoloFile}
                    onRevertToDefault={defaultYoloFile && statsYoloWeights !== defaultYoloFile
                      ? () => setStatsYoloWeights(defaultYoloFile)
                      : undefined}
                  />
                )}
                <MultiFileDropzone
                  files={statsImages} setFiles={setStatsImages}
                  label="평가 이미지" emptyText="이미지 끌어다 놓기"
                  emptyHint="여러 장 가능"
                />
                <MultiFileDropzone
                  files={statsGts} setFiles={setStatsGts}
                  label="정답 마스크 (필수)" emptyText="GT 끌어다 놓기"
                  emptyHint="파일명 매칭. sweep 평가에 필수"
                />
              </>
            ) : mode === "augment" ? (
              <>
                <MultiFileDropzone
                  files={augImages}
                  setFiles={setAugImages}
                  label="증강할 이미지"
                  emptyText="이미지 끌어다 놓거나 클릭"
                  emptyHint="여러 장 가능. 첫 번째 이미지가 미리보기에 사용됨"
                />
                <MultiFileDropzone
                  files={augMasks}
                  setFiles={setAugMasks}
                  label="대응 마스크 (선택)"
                  emptyText="GT 마스크 끌어다 놓기"
                  emptyHint="파일명 매칭. 마스크도 같이 변환되어 ZIP 에 포함"
                />
              </>
            ) : (
              <>
                <MultiFileDropzone
                  files={predictImages}
                  setFiles={setPredictImages}
                  label={mode === "legacy" ? "입력 이미지 (이진 마스크)" : "입력 이미지 (SEM)"}
                  emptyText="이미지 끌어다 놓거나 클릭"
                  emptyHint={mode === "legacy"
                    ? "이미 이진화된 PNG/JPG · 여러 장 가능"
                    : "원본 SEM 이미지 · 여러 장 가능"}
                />
                <MultiFileDropzone
                  files={predictGts}
                  setFiles={setPredictGts}
                  label="정답 마스크 (선택)"
                  emptyText="GT 마스크 끌어다 놓기"
                  emptyHint="파일명 매칭 시 자동으로 지표 계산"
                />
              </>
            )}
          </div>

          {/* GT 매칭 안내 — 평가 지표 산출이 의미 있는 모드에서만 표시 */}
          {(mode === "hybrid" || mode === "legacy"
            || (mode === "yolo" && yoloTab === "predict")
            || (mode === "unet" && unetTab === "predict")) && (
            <div className="rounded-md border border-[#f1f1ef] bg-[#fafafa] px-3 py-2.5 text-[10.5px] text-[#6b7280] leading-relaxed">
              <div className="flex items-start gap-1.5">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="flex-shrink-0 mt-0.5 text-[#9ca3af]">
                  <circle cx="12" cy="12" r="10"/>
                  <line x1="12" y1="16" x2="12" y2="12"/>
                  <line x1="12" y1="8" x2="12.01" y2="8"/>
                </svg>
                <div className="min-w-0">
                  <div className="font-semibold text-[#191919] mb-0.5">평가 지표를 보려면 정답 마스크 (GT) 도 함께 올리세요</div>
                  <div>파일명이 자동 매칭됩니다. 예: <code className="text-[10px]">foo.png</code> ↔ <code className="text-[10px]">foo.png</code>, 또는 <code className="text-[10px]">foo_image.png</code> ↔ <code className="text-[10px]">foo_label.png</code></div>
                </div>
              </div>
            </div>
          )}

          {mode === "augment" ? (
            <RunButton
              onClick={runAugmentDownload}
              disabled={augBusy || augImages.length === 0}
              busy={augBusy}
              label={augBusy ? "ZIP 생성 중..." : `증강 ${augImages.length}장 × ${augNPer}개 → ZIP`}
              icon="download"
            />
          ) : mode === "unet" && unetTab === "train" ? (
            <RunButton
              onClick={runUnetTrain}
              disabled={!canTrainUnet}
              busy={running}
              label={running ? "학습 중..." : "U-Net 학습 시작"}
              icon="train"
            />
          ) : mode === "yolo" && yoloTab === "train" ? (
            <RunButton
              onClick={runYoloTrain}
              disabled={!canTrainYolo}
              busy={running}
              label={running ? "학습 중..." : "YOLO 학습 시작"}
              icon="train"
            />
          ) : (
            <RunButton
              onClick={run}
              disabled={!canRun}
              busy={running}
              label={running ? "실행 중..." : "분석 실행"}
              icon="play"
            />
          )}

          {error && (
            <div className="text-[11.5px] text-[#7a1c1c] bg-[#fff1f1] border border-[#fbcaca] rounded-md p-3 whitespace-pre-wrap leading-relaxed fade-in flex gap-2">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#c92a2a" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="flex-shrink-0 mt-0.5">
                <circle cx="12" cy="12" r="10"/>
                <line x1="12" y1="8" x2="12" y2="12"/>
                <line x1="12" y1="16" x2="12.01" y2="16"/>
              </svg>
              <div className="min-w-0 break-words">{error}</div>
            </div>
          )}
        </section>

        {/* ── 중앙 — 뷰어 / 결과 ── */}
        <section className="space-y-3 min-w-0 max-w-full">
          {mode === "augment" && augPreview && (
            <div className="app-card p-3 min-w-0">
              <div className="flex items-baseline justify-between mb-2 flex-wrap gap-2">
                <h2 className="text-sm font-semibold">미리보기</h2>
                <div className="text-[11px] text-gray-500">
                  {augPreview.shape[1]} × {augPreview.shape[0]} ·
                  brightness=+{augBrightness} ·
                  contrast={(1 + augContrast).toFixed(2)} ·
                  noise σ={augNoise}
                </div>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                <ImageBox label="원본" src={augPreview.original_png} />
                <ImageBox label="증강 후" src={augPreview.augmented_png} />
                {augPreview.mask_original_png && (
                  <ImageBox label="원본 마스크" src={augPreview.mask_original_png} />
                )}
                {augPreview.mask_augmented_png && (
                  <ImageBox label="증강 후 마스크" src={augPreview.mask_augmented_png} />
                )}
              </div>
              <div className="text-[11px] text-gray-500 mt-2 leading-snug">
                <b>마스크는 변형 없음</b> — 원본과 픽셀 동일하게 유지됩니다.
                이미지에만 밝기/대비/노이즈 적용 (image-mask 정합성 보존).
                미리보기는 결정적 (max 값 적용). 실제 ZIP 은 매 인스턴스마다 ±범위에서 랜덤.
              </div>
            </div>
          )}

          {!legacyRes && !yoloRes && !unetRes && !hybridRes && !statsRes && !researchRes && !unetTrainRes && !yoloTrainRes && !augPreview && !running && !augBusy && (
            <div className="app-card p-10 lg:p-16 text-center fade-in">
              <div className="mx-auto w-10 h-10 rounded-full flex items-center justify-center mb-4 bg-[#f5f5f3]">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#6b7280" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                  <circle cx="11" cy="11" r="6.5"/>
                  <path d="m20 20-4.35-4.35"/>
                </svg>
              </div>
              <div className="text-[14px] font-semibold text-[#191919]">아직 표시할 결과가 없어요</div>
              <p className="text-[12px] text-[#6b7280] mt-1.5 max-w-sm mx-auto leading-relaxed">
                왼쪽에서 파일을 올리고 오른쪽에서 파라미터를 조정한 뒤
                <kbd className="mx-1">실행</kbd>을 누르면 결과가 여기에 나타나요.
              </p>
            </div>
          )}

          {running
            && !(mode === "unet" && unetTab === "train" && unetTrainProgress)
            && !(mode === "yolo" && yoloTab === "train" && yoloTrainProgress)
            && !(mode === "stats" && statsJob)
            && !(mode === "stats" && researchJob) && (
            <InferenceProgressCard
              label={inferLabel || "추론"}
              total={inferTotal}
              startMs={inferStartMs}
              tick={inferTick}
              firstCallNote={
                mode === "yolo" && yoloTab === "predict"
                  ? "첫 호출은 모델 로드로 30초~1분 더 걸릴 수 있어요"
                  : mode === "hybrid"
                    ? "U-Net 가중치 + YOLO 가중치 로드 후 이미지별 융합"
                    : mode === "unet" && unetTab === "predict"
                      ? "torch 가중치 로드 후 이미지별 추론"
                      : null
              }
            />
          )}

          {mode === "unet" && unetTab === "train" && unetTrainProgress && running && (
            <UnetTrainProgressCard p={unetTrainProgress} />
          )}
          {mode === "yolo" && yoloTab === "train" && yoloTrainProgress && running && (
            <YoloTrainProgressCard p={yoloTrainProgress} />
          )}
          {mode === "stats" && statsJob && !statsRes && (
            <StatsProgressCard p={statsJob} />
          )}
          {mode === "stats" && researchJob && !researchRes && (
            <ResearchProgressCard p={researchJob} />
          )}
          {yoloTrainRes && !running && (
            <YoloTrainView
              r={yoloTrainRes}
              onUseForPredict={() => {
                setYoloUseTrained(true);
                setYoloTab("predict");
              }}
            />
          )}

          {legacyRes && (
            <LegacyView
              r={legacyRes}
              activeIdx={legacyActiveIdx}
              onPick={setLegacyActiveIdx}
            />
          )}
          {unetRes && (
            <UnetView
              r={unetRes}
              probView={unetProbView}
              activeIdx={unetActiveIdx}
              onPick={setUnetActiveIdx}
            />
          )}
          {unetTrainRes && (
            <UnetTrainView r={unetTrainRes} onUseForPredict={() => {
              setUnetUseTrained(true); setUnetTab("predict");
            }} />
          )}
          {yoloRes && (
            <YoloView
              r={yoloRes}
              activeIdx={yoloActiveIdx}
              onPick={setYoloActiveIdx}
            />
          )}
          {hybridRes && (
            <HybridView
              r={hybridRes}
              activeIdx={hybridActiveIdx}
              onPick={setHybridActiveIdx}
              diagView={hybridDiagView}
              setDiagView={setHybridDiagView}
              originalsMap={hybridOriginals}
              labelScale={hybridLabelScale}
              setLabelScale={setHybridLabelScale}
              numberedSrc={hybridNumberedSrc}
              numberedBusy={hybridNumberedBusy}
            />
          )}
          {statsRes && (
            <StatsView
              r={statsRes}
              activeChart={statsActiveChart}
              setActiveChart={setStatsActiveChart}
            />
          )}
          {researchRes && (
            <ResearchView
              r={researchRes}
              activeHeatmap={researchActiveHeatmap}
              setActiveHeatmap={setResearchActiveHeatmap}
            />
          )}
        </section>

        {/* ── 우측 — 파라미터 패널 (가로 폭 드래그 조절) ── */}
        <aside className="relative space-y-3 min-w-0 lg:max-h-[calc(100vh-90px)] lg:overflow-y-auto lg:sticky lg:top-[68px] pr-1">
          <div
            className="resize-handle hidden lg:flex"
            title="드래그로 너비 조절"
            onMouseDown={startResize}
          />
          {mode === "augment" && (
            <div className="app-card p-4 space-y-3">
              <h2 className="text-sm font-semibold">증강 파라미터</h2>
              <NumberKnob
                label="이미지당 증강 개수 (Augmentations per image)"
                desc="원본 1장당 만들어낼 증강본 수"
                value={augNPer} onChange={setAugNPer}
                min={1} max={50}
              />
              <FloatKnob
                label="밝기 (Brightness ±)"
                desc="±이 값 사이에서 랜덤. 미리보기는 +값 적용"
                value={augBrightness} onChange={setAugBrightness}
                min={0} max={100} step={1}
              />
              <FloatKnob
                label="대비 (Contrast ±)"
                desc="1±이 값 (예: 0.2 → 0.8~1.2 배율)"
                value={augContrast} onChange={setAugContrast}
                min={0} max={1} step={0.05}
              />
              <FloatKnob
                label="가우시안 노이즈 (Noise σ max)"
                desc="0=끄기, 표준편차 0~이 값 사이 랜덤"
                value={augNoise} onChange={setAugNoise}
                min={0} max={50} step={1}
              />
              <label className="block text-xs">
                <div className="mb-1">파일명 접미사 (Suffix template)</div>
                <input
                  type="text"
                  value={augSuffix}
                  onChange={(e) => setAugSuffix(e.target.value)}
                  className="w-full border border-gray-300 rounded px-2 py-1.5 text-xs font-mono"
                />
                <div className="text-[10px] text-gray-500 mt-0.5 leading-snug">
                  {"{n}"} 자리에 1,2,...N 들어감. 예: <code>_aug{"{n}"}</code> → <code>foo_aug1_image.png</code><br/>
                  <code>_batch_p{"{n}"}</code> → <code>foo_batch_p1_image.png</code>
                </div>
              </label>
              <Toggle
                label="ZIP 에 원본도 포함"
                desc="원본 이미지/마스크도 결과 ZIP 에 함께 넣기"
                value={augIncludeOrig} onChange={setAugIncludeOrig}
              />
              <NumberKnob
                label="시드 (Random seed)"
                desc="같은 시드 + 같은 입력이면 같은 결과 재현"
                value={augSeed} onChange={setAugSeed}
                min={0} max={99999}
              />
              <button
                onClick={runAugPreview}
                disabled={augBusy || augImages.length === 0}
                className={
                  "w-full mt-2 py-1.5 rounded text-xs font-semibold " +
                  (augBusy || augImages.length === 0
                    ? "bg-gray-200 text-gray-500"
                    : "bg-amber-500 text-white hover:bg-amber-600")
                }
              >
                {augBusy ? "처리 중..." : "미리보기 (첫 번째 이미지)"}
              </button>
            </div>
          )}

          {mode === "legacy" && (
            <div className="app-card p-4 space-y-3">
              <h2 className="text-sm font-semibold">레거시 파라미터</h2>
              <Toggle label="입자/배경 반전" value={invert} onChange={setInvert} />
              <NumberKnob label="최소 입자 면적 (px)" value={minSize} onChange={setMinSize} min={0} max={2000} />
              <NumberKnob label="모폴로지 커널 (0=끄기)" value={morphKernel} onChange={setMorphKernel} min={0} max={10} />
              <Toggle label="Watershed 분리" value={useWatershed} onChange={setUseWatershed} />
              <NumberKnob label="Watershed 최소거리" value={watershedDist} onChange={setWatershedDist} min={1} max={50} disabled={!useWatershed} />
            </div>
          )}

          {mode === "yolo" && yoloTab === "train" && (
            <div className="app-card p-4 space-y-3">
              <h2 className="text-sm font-semibold">학습 파라미터</h2>
              <label className="block text-xs">
                <div className="mb-1">기본 가중치 (Base architecture)</div>
                <select
                  value={yoloBaseArch}
                  onChange={(e) => setYoloBaseArch(e.target.value)}
                  className="w-full border border-gray-300 rounded px-2 py-1.5 text-xs bg-white"
                >
                  <optgroup label="YOLO11 (2024, 권장)">
                    <option value="yolo11n-seg.pt">yolo11n-seg · ~6 MB · 가장 빠름</option>
                    <option value="yolo11s-seg.pt">yolo11s-seg · ~24 MB · 중간</option>
                    <option value="yolo11m-seg.pt">yolo11m-seg · ~56 MB · 큼</option>
                    <option value="yolo11l-seg.pt">yolo11l-seg · ~78 MB · 매우 큼</option>
                  </optgroup>
                  <optgroup label="YOLOv8 (안정)">
                    <option value="yolov8n-seg.pt">yolov8n-seg · ~7 MB · 가장 빠름</option>
                    <option value="yolov8s-seg.pt">yolov8s-seg · ~25 MB · 중간</option>
                    <option value="yolov8m-seg.pt">yolov8m-seg · ~52 MB · 큼</option>
                    <option value="yolov8l-seg.pt">yolov8l-seg · ~92 MB · 매우 큼</option>
                  </optgroup>
                </select>
                <div className="text-[10px] text-gray-500 mt-0.5 leading-snug">
                  ultralytics 사전학습 가중치. 처음 사용시 자동 다운로드 (인터넷 필요).
                  YOLO11 이 더 새롭고 정확, YOLOv8 은 호환성·안정성 우위.
                </div>
              </label>
              <NumberKnob
                label="에포크 (Epochs)"
                desc="전체 학습 데이터 반복 학습 횟수. 작은 데이터는 50~100 권장"
                value={yoloTrainEpochs} onChange={setYoloTrainEpochs}
                min={1} max={500}
              />
              <NumberKnob
                label="배치 크기 (Batch size)"
                desc="한 번에 처리하는 샘플 수. VRAM 부족하면 줄이기"
                value={yoloTrainBatch} onChange={setYoloTrainBatch}
                min={1} max={32}
              />
              <NumberKnob
                label="입력 크기 (Image size, px)"
                desc="학습/추론 시 리사이즈할 변 길이. 32 의 배수"
                value={yoloTrainImgsz} onChange={setYoloTrainImgsz}
                min={320} max={1600} step={32}
              />
              <FloatKnob
                label="검증 비율 (Validation fraction)"
                desc="전체 중 검증셋 비율. 데이터가 적으면 0.5 도 가능"
                value={yoloTrainValFrac} onChange={setYoloTrainValFrac}
                min={0.1} max={0.5} step={0.05}
              />
              <label className="block text-xs">
                <div className="mb-1">디바이스 (Device)</div>
                <select
                  value={yoloTrainDevice}
                  onChange={(e) => setYoloTrainDevice(e.target.value as any)}
                  className="w-full border border-gray-300 rounded px-2 py-1.5 text-xs bg-white"
                >
                  <option value="auto">auto (mps/cuda 우선)</option>
                  <option value="cpu">cpu (느리지만 안정)</option>
                  <option value="mps">mps (Apple GPU)</option>
                  <option value="cuda">cuda (NVIDIA GPU)</option>
                </select>
                <div className="text-[10px] text-gray-500 mt-0.5 leading-snug">
                  컴퓨터가 무거우면 cpu 또는 작은 batch/imgsz 로 시도
                </div>
              </label>
            </div>
          )}

          {mode === "hybrid" && (
            <HybridParamsPanel
              high={hybridHighThresh} setHigh={setHybridHighThresh}
              low={hybridLowThresh} setLow={setHybridLowThresh}
              boundaryThk={hybridBoundaryThk} setBoundaryThk={setHybridBoundaryThk}
              minArea={hybridMinArea} setMinArea={setHybridMinArea}
              balance={hybridBalance} setBalance={setHybridBalance}
              tauScale={hybridTauScale} setTauScale={setHybridTauScale}
              yoloConf={hybridYoloConf} setYoloConf={setHybridYoloConf}
              yoloIou={hybridYoloIou} setYoloIou={setHybridYoloIou}
              yoloImgsz={hybridYoloImgsz} setYoloImgsz={setHybridYoloImgsz}
              yoloRound={hybridYoloRound} setYoloRound={setHybridYoloRound}
              yoloNoOverlap={hybridYoloNoOverlap} setYoloNoOverlap={setHybridYoloNoOverlap}
              yoloRemoveEnclosed={hybridYoloRemoveEnclosed} setYoloRemoveEnclosed={setHybridYoloRemoveEnclosed}
              enclosedRatio={hybridEnclosedRatio} setEnclosedRatio={setHybridEnclosedRatio}
              device={hybridDevice} setDevice={setHybridDevice}
            />
          )}

          {/* 다운로드 패널 — Hybrid 결과 있을 때만, 파라미터 바로 아래 */}
          {mode === "hybrid" && hybridRes && (() => {
            const it = hybridRes.results[Math.min(hybridActiveIdx, hybridRes.results.length - 1)];
            if (!it) return null;
            const dv = hybridDiagView;
            const diagLabelMap: Record<string, string> = {
              original: "원본", overlay: "오버레이", instances: "Instance",
              numbered: "번호", mask: "Mask", prob: "U-Net prob",
              high: "M_75 (anchor)", close: "M_close",
              boundary: "YOLO 경계", separator: "최종 분리선",
              color: "색 진단", hist: "d₊ 히스토그램",
            };
            const srcByView: Record<string, string> = {
              original: hybridOriginals[it.filename] ?? "",
              overlay: it.overlay_png,
              instances: it.instances_png,
              numbered: hybridNumberedSrc || it.instances_png,
              mask: it.mask_png,
              prob: it.diag_prob_png,
              high: it.diag_high_png,
              close: it.diag_close_png,
              boundary: it.diag_boundary_png,
              separator: it.diag_separator_png,
              color: it.diag_overlap_png,
              hist: it.diag_hist_png ?? "",
            };
            return (
              <HybridDownloadBar
                result={hybridRes}
                currentItem={it}
                currentDiagView={dv}
                currentDiagLabel={diagLabelMap[dv] ?? "image"}
                currentSrc={srcByView[dv] ?? ""}
                originalsMap={hybridOriginals}
                labelScale={hybridLabelScale}
              />
            );
          })()}

          {mode === "stats" && (
            <div className="app-card p-4 space-y-3">
              <h2 className="text-sm font-semibold">Sweep 설정</h2>
              <label className="block text-xs">
                <div className="mb-1 font-semibold">Sweep 모드</div>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-1">
                  {([
                    ["hybrid", "Hybrid", "3-way 비교"],
                    ["yolo", "YOLO만", "단일"],
                    ["unet", "U-Net만", "단일"],
                    ["research", "연구", "민감도 분석"],
                  ] as const).map(([id, label, sub]) => (
                    <button
                      key={id}
                      type="button"
                      onClick={() => {
                        setStatsSweepMode(id);
                        // 모드 바꿀 때 기본 sweep_param 선택
                        if (id === "hybrid") {
                          setStatsSweepParam("high_thresh");
                          setStatsSweepFrom(0.5); setStatsSweepTo(0.95); setStatsSweepStep(0.05);
                        } else if (id === "yolo") {
                          setStatsSweepParam("conf");
                          setStatsSweepFrom(0.1); setStatsSweepTo(0.6); setStatsSweepStep(0.05);
                        } else if (id === "unet") {
                          setStatsSweepParam("threshold");
                          setStatsSweepFrom(0.1); setStatsSweepTo(0.9); setStatsSweepStep(0.05);
                        }
                        // research 는 sweep_param 단일 X — 별도 폼 사용
                      }}
                      className={
                        "px-2 py-1.5 rounded border text-left " +
                        (statsSweepMode === id
                          ? "bg-[#1f5fd6] text-white border-[#1f5fd6]"
                          : "bg-white border-gray-300 hover:border-gray-500")
                      }
                    >
                      <div className="text-[11px] font-semibold">{label}</div>
                      <div className={"text-[9px] " + (statsSweepMode === id ? "opacity-80" : "text-gray-500")}>
                        {sub}
                      </div>
                    </button>
                  ))}
                </div>
              </label>
              <div className="text-[11px] text-gray-500 leading-snug">
                {statsSweepMode === "hybrid" && "Hybrid fusion 파라미터 sweep. U-Net·YOLO 추론은 1회만, fusion 만 반복 — 빠름. 3 방법 metrics 동시 비교."}
                {statsSweepMode === "yolo"   && "YOLO 추론 파라미터 sweep. 각 값마다 YOLO 추론 다시. U-Net 불필요. YOLO metrics 만 산출."}
                {statsSweepMode === "unet"   && "U-Net binary threshold sweep. prob 캐시 후 임계 재적용 — 빠름. YOLO 불필요. U-Net metrics 만 산출."}
                {statsSweepMode === "research" && "논문 3.5 — 민감도 분석 자료 자동 생성. 4개 파라미터 (τ_h, τ_l, β, σ) 동시 sweep + (옵션) τ_h×τ_l 격자. 결과: 1×4 패널 figure + 격자 heatmap + 엑셀. 모든 sweep 결과가 평평할수록 robustness 입증."}
              </div>
              {statsSweepMode !== "research" && (<>
              <label className="block text-xs">
                <div className="mb-1">Sweep 파라미터</div>
                <select
                  value={statsSweepParam}
                  onChange={(e) => setStatsSweepParam(e.target.value)}
                  className="w-full border border-gray-300 rounded px-2 py-1.5 text-xs bg-white"
                >
                  {statsSweepMode === "hybrid" && (<>
                    <option value="high_thresh">high_thresh (U-Net anchor)</option>
                    <option value="low_thresh">low_thresh (U-Net boost)</option>
                    <option value="balance_floor">balance_floor (d₁≈d₂)</option>
                    <option value="tau_scale">tau_scale (Otsu 배수)</option>
                    <option value="boundary_thickness">boundary_thickness (px, 정수)</option>
                    <option value="min_anchor_area">min_anchor_area (px, 정수)</option>
                    <option value="yolo_round_strength">yolo_round_strength (정수)</option>
                  </>)}
                  {statsSweepMode === "yolo" && (<>
                    <option value="conf">conf (confidence threshold)</option>
                    <option value="iou">iou (NMS IoU)</option>
                    <option value="imgsz">imgsz (input size, 정수)</option>
                    <option value="round_strength">round_strength (라운드, 정수)</option>
                    <option value="smooth">smooth (모폴로지, 정수)</option>
                  </>)}
                  {statsSweepMode === "unet" && (<>
                    <option value="threshold">threshold (binary 임계)</option>
                  </>)}
                </select>
              </label>
              <div className="grid grid-cols-3 gap-2 text-xs">
                <label className="block">
                  <div className="mb-1 text-gray-600">from</div>
                  <input
                    type="number" step="any"
                    value={statsSweepFrom}
                    onChange={(e) => setStatsSweepFrom(Number(e.target.value))}
                    className="w-full border border-gray-300 rounded px-2 py-1.5"
                  />
                </label>
                <label className="block">
                  <div className="mb-1 text-gray-600">to</div>
                  <input
                    type="number" step="any"
                    value={statsSweepTo}
                    onChange={(e) => setStatsSweepTo(Number(e.target.value))}
                    className="w-full border border-gray-300 rounded px-2 py-1.5"
                  />
                </label>
                <label className="block">
                  <div className="mb-1 text-gray-600">step</div>
                  <input
                    type="number" step="any"
                    value={statsSweepStep}
                    onChange={(e) => setStatsSweepStep(Number(e.target.value))}
                    className="w-full border border-gray-300 rounded px-2 py-1.5"
                  />
                </label>
              </div>
              <div className="text-[10px] text-gray-500">
                생성될 값 수: ~{Math.max(0, Math.floor((statsSweepTo - statsSweepFrom) / Math.max(statsSweepStep, 1e-9)) + 1)}개 (30 초과 금지)
              </div>
              </>)}

              {statsSweepMode === "research" && (
                <div className="space-y-2 border border-purple-200 bg-purple-50 rounded p-3">
                  <div className="text-[11px] font-semibold text-purple-800">
                    연구 모드 — 4 파라미터 동시 sweep
                  </div>
                  <div className="text-[10px] text-purple-700 leading-snug">
                    각 파라미터 CSV 입력 — <b>비우면 그 sweep 스킵</b> (예: β·σ 만 재실행하려면
                    τ_h, τ_l 칸 비우기). 격자는 별도로 활성/비활성 가능.
                  </div>
                  {([
                    ["resHigh", "τ_h (high_thresh)", resSweepHigh, setResSweepHigh],
                    ["resLow",  "τ_l (low_thresh)",  resSweepLow,  setResSweepLow],
                    ["resBal",  "β (balance_floor)", resSweepBalance, setResSweepBalance],
                    ["resRnd",  "σ (yolo_round)",    resSweepRound, setResSweepRound],
                  ] as const).map(([key, label, val, setter]) => (
                    <label key={key} className="block text-[11px]">
                      <div className="mb-0.5 text-gray-700">{label} — CSV</div>
                      <input
                        type="text" value={val as string}
                        onChange={(e) => (setter as (s: string) => void)(e.target.value)}
                        className="w-full border border-gray-300 rounded px-2 py-1 font-mono text-[11px]"
                      />
                    </label>
                  ))}
                  <Toggle
                    label="격자 heatmap 도 생성"
                    desc="두 파라미터 5×5 grid + heatmap (mAP/CountError/IoU 각 한 장)"
                    value={resEnableGrid}
                    onChange={setResEnableGrid}
                  />
                  {resEnableGrid && (
                    <div className="grid grid-cols-2 gap-2 text-[11px]">
                      <label>
                        <div className="text-gray-600 mb-0.5">x 축</div>
                        <select
                          value={resGridX}
                          onChange={(e) => setResGridX(e.target.value as any)}
                          className="w-full border border-gray-300 rounded px-2 py-1 bg-white"
                        >
                          <option value="low_thresh">τ_l</option>
                          <option value="high_thresh">τ_h</option>
                          <option value="balance_floor">β</option>
                          <option value="yolo_round_strength">σ</option>
                        </select>
                      </label>
                      <label>
                        <div className="text-gray-600 mb-0.5">y 축</div>
                        <select
                          value={resGridY}
                          onChange={(e) => setResGridY(e.target.value as any)}
                          className="w-full border border-gray-300 rounded px-2 py-1 bg-white"
                        >
                          <option value="high_thresh">τ_h</option>
                          <option value="low_thresh">τ_l</option>
                          <option value="balance_floor">β</option>
                          <option value="yolo_round_strength">σ</option>
                        </select>
                      </label>
                    </div>
                  )}
                  <div className="text-[10px] text-purple-700 leading-snug">
                    아래 "고정 파라미터" 의 운용값이 차트의 세로 점선으로 표시됨.
                    가중치 + 이미지 + GT 모두 필요.
                  </div>
                </div>
              )}

              <div className="border-t pt-2 mt-2 space-y-3">
                <div className="text-[11px] text-gray-500 font-semibold">고정 파라미터 (sweep 안 하는 값)</div>
                <FloatKnob
                  label="U-Net high threshold (anchor)"
                  desc="이 확률 이상 = 확실한 입자 (메인). 보수적으로 잡으려면 높게"
                  value={statsHigh} onChange={setStatsHigh}
                  min={0.5} max={0.99} step={0.01}
                />
                <FloatKnob
                  label="U-Net low threshold (boost)"
                  desc="좁은 틈 영역에서만 살리는 보강 임계. high 보다 작아야 함"
                  value={statsLow} onChange={setStatsLow}
                  min={0.05} max={0.7} step={0.01}
                />
                <NumberKnob
                  label="YOLO 분리선 두께 (px)"
                  desc="1=얇음(noise 민감), 3~5=안정. 입자 사이 자르는 두께"
                  value={statsThickness} onChange={setStatsThickness}
                  min={1} max={10}
                />
                <NumberKnob
                  label="최소 입자 크기 (px)"
                  desc="anchor + 최종 결과 모두 이 픽셀 미만 component 제거"
                  value={statsMinArea} onChange={setStatsMinArea}
                  min={1} max={500}
                />
                <FloatKnob
                  label="d₁≈d₂ balance"
                  desc="0=spec 그대로 / 0.25 권장 / 0.5=엄격 (두 입자 정확히 중간 픽셀만)"
                  value={statsBalance} onChange={setStatsBalance}
                  min={0} max={0.7} step={0.05}
                />
                <FloatKnob
                  label="τ* 배수 (좁은 틈 검출 범위)"
                  desc="1.0=Otsu 자동값 그대로. 좁은 틈이 검출 안 되면 1.3~2.0 으로 올리기"
                  value={statsTauScale} onChange={setStatsTauScale}
                  min={0.5} max={3.0} step={0.1}
                />
              </div>
              <div className="border-t pt-2 mt-2 space-y-2">
                <div className="text-[11px] text-gray-500 font-semibold">YOLO 추론 + 후처리 (1 회만)</div>
                <FloatKnob label="conf" value={statsYoloConf} onChange={setStatsYoloConf} min={0} max={1} step={0.01} />
                <FloatKnob label="iou (NMS)" value={statsYoloIou} onChange={setStatsYoloIou} min={0} max={1} step={0.01} />
                <NumberKnob label="imgsz" value={statsYoloImgsz} onChange={setStatsYoloImgsz} min={320} max={1600} step={32} />
                <NumberKnob
                  label="라운드/부드럽게 (Gaussian σ)"
                  desc="0=끄기, 3~5 권장. 클수록 동그래짐 (작은 입자는 줄여)"
                  value={statsRound} onChange={setStatsRound}
                  min={0} max={10}
                />
                <Toggle
                  label="겹친 인스턴스 영역 = 배경"
                  desc="두 YOLO instance 가 겹친 픽셀은 양쪽 모두에서 제거 → 분리선"
                  value={statsYoloNoOverlap}
                  onChange={setStatsYoloNoOverlap}
                />
              </div>
              <label className="block text-xs">
                <div className="mb-1">디바이스</div>
                <select
                  value={statsDevice}
                  onChange={(e) => setStatsDevice(e.target.value)}
                  className="w-full border border-gray-300 rounded px-2 py-1.5 text-xs bg-white"
                >
                  <option value="auto">자동</option>
                  <option value="cpu">CPU</option>
                  <option value="mps">MPS</option>
                  <option value="cuda:0">CUDA</option>
                </select>
              </label>
            </div>
          )}

          {mode === "yolo" && yoloTab === "predict" && (
            <div className="app-card p-4 space-y-3">
              <h2 className="text-sm font-semibold">YOLO 파라미터</h2>
              <FloatKnob label="Confidence" value={yoloConf} onChange={setYoloConf} min={0} max={1} step={0.01} />
              <FloatKnob label="IoU (NMS)" value={yoloIou} onChange={setYoloIou} min={0} max={1} step={0.01} />
              <NumberKnob label="입력 크기 (imgsz)" value={yoloImgsz} onChange={setYoloImgsz} min={320} max={1600} step={32} />
              <NumberKnob label="마스크 스무딩 (모폴로지 커널, 0=끄기)" value={yoloSmooth} onChange={setYoloSmooth} min={0} max={10} />
              <NumberKnob
                label="라운드/부드럽게 (Gaussian σ, 0=끄기)"
                desc="3~5 권장. 클수록 동그래짐. smooth 후에 적용됨"
                value={yoloRound} onChange={setYoloRound}
                min={0} max={10}
              />
              <Toggle
                label="겹친 인스턴스 영역 = 배경"
                desc="두 instance 가 겹친 픽셀은 양쪽 모두에서 제거"
                value={yoloNoOverlap}
                onChange={setYoloNoOverlap}
              />
              <label className="block text-xs">
                <div className="mb-1">디바이스</div>
                <select
                  value={yoloDevice}
                  onChange={(e) => setYoloDevice(e.target.value)}
                  className="w-full border border-gray-300 rounded px-2 py-1.5 text-xs bg-white"
                >
                  <option value="auto">자동 (mps/cuda/cpu)</option>
                  <option value="cpu">CPU</option>
                  <option value="mps">MPS (Apple)</option>
                  <option value="cuda:0">CUDA</option>
                </select>
              </label>
            </div>
          )}

          {mode === "unet" && unetTab === "predict" && (
            <div className="app-card p-4 space-y-3">
              <h2 className="text-sm font-semibold">추론 파라미터</h2>
              <FloatKnob label="이진화 임계값 (mask 산출)"
                value={unetThreshold} onChange={setUnetThreshold}
                min={0} max={1} step={0.01} />
              <FloatKnob label="표시 임계 (이 확률 미만 픽셀 가리기)"
                value={unetMinProb} onChange={setUnetMinProb}
                min={0} max={1} step={0.01} />
              <label className="block text-xs">
                <div className="flex justify-between mb-1">
                  <span>입력 크기 (비우면 모델 메타값)</span>
                  <span className="font-mono text-gray-600">
                    {unetImgsz === "" ? "auto" : unetImgsz}
                  </span>
                </div>
                <input
                  type="number"
                  placeholder="예: 256"
                  value={unetImgsz}
                  onChange={(e) => setUnetImgsz(
                    e.target.value === "" ? "" : Number(e.target.value)
                  )}
                  className="w-full border border-gray-300 rounded px-2 py-1.5 text-xs"
                />
              </label>
              <label className="block text-xs">
                <div className="mb-1">확률맵 표시 방식</div>
                <select
                  value={unetProbView}
                  onChange={(e) => setUnetProbView(e.target.value as any)}
                  className="w-full border border-gray-300 rounded px-2 py-1.5 text-xs bg-white"
                >
                  <option value="color">컬러 (Jet 컬러맵)</option>
                  <option value="gray">회색조</option>
                  <option value="filtered_color">컬러 + 표시임계 적용</option>
                  <option value="filtered_gray">회색조 + 표시임계 적용</option>
                </select>
              </label>
            </div>
          )}

          {mode === "unet" && unetTab === "train" && (
            <div className="app-card p-4 space-y-3">
              <h2 className="text-sm font-semibold">학습 파라미터</h2>

              <label className="block text-xs">
                <div className="mb-1">아키텍처 (Architecture)</div>
                <select
                  value={unetMethod}
                  onChange={(e) => setUnetMethod(e.target.value as any)}
                  className="w-full border border-gray-300 rounded px-2 py-1.5 text-xs bg-white"
                >
                  <option value="U-Net">U-Net</option>
                  <option value="U-Net++">U-Net++</option>
                </select>
                <div className="text-[10px] text-gray-500 mt-0.5 leading-snug">
                  분할 모델 종류. U-Net++ 는 nested skip 으로 보통 더 정밀
                </div>
              </label>

              <NumberKnob
                label="에포크 (Epochs)"
                desc="전체 학습 데이터를 반복 학습하는 횟수. 많을수록 잘 맞지만 과적합 위험"
                value={unetEpochs}
                onChange={setUnetEpochs} min={1} max={300}
              />
              <NumberKnob
                label="배치 크기 (Batch size)"
                desc="한 번에 GPU 에 올리는 샘플 수. VRAM 이 모자라면 줄이기"
                value={unetBatch}
                onChange={setUnetBatch} min={1} max={32}
              />
              <FloatKnob
                label="학습률 (Learning rate, ×1e-3)"
                desc="AdamW 의 보폭. 너무 크면 발산, 작으면 수렴이 느림"
                value={unetLr * 1000}
                onChange={(v) => setUnetLr(v / 1000)}
                min={0.01} max={10} step={0.01}
              />
              <NumberKnob
                label="이미지 크기 (Image size, px)"
                desc="학습 시 입력으로 리사이즈할 변 길이. 16의 배수 권장"
                value={unetTrainImgSize} onChange={setUnetTrainImgSize}
                min={64} max={1024} step={16}
              />
              <FloatKnob
                label="검증 비율 (Validation fraction)"
                desc="전체 데이터 중 검증셋으로 떼어둘 비율. 보통 0.1~0.2"
                value={unetValFrac} onChange={setUnetValFrac}
                min={0.05} max={0.5} step={0.05}
              />
              <label className="block text-xs">
                <div className="mb-1">디바이스 (Device)</div>
                <select
                  value={unetDevice}
                  onChange={(e) => setUnetDevice(e.target.value as any)}
                  className="w-full border border-gray-300 rounded px-2 py-1.5 text-xs bg-white"
                >
                  <option value="auto">auto (mps/cuda 우선)</option>
                  <option value="cpu">cpu (가장 안정적, 느림)</option>
                  <option value="mps">mps (Apple GPU — 발열↑)</option>
                  <option value="cuda">cuda (NVIDIA GPU)</option>
                </select>
                <div className="text-[10px] text-gray-500 mt-0.5 leading-snug">
                  컴퓨터가 멈춘다면 cpu 또는 작은 batch/image_size 로 시도
                </div>
              </label>

              <Toggle
                label="데이터 증강 (Augmentation)"
                desc="회전·뒤집기로 학습 데이터를 무작위 변형해 부풀리기"
                value={unetAugment} onChange={setUnetAugment}
              />
              <Toggle
                label="BCE + Dice 혼합 (Combined loss)"
                desc="가중 BCE 와 Dice 손실의 가중 합 사용 — 균형 잡힌 학습"
                value={unetUseDice} onChange={setUnetUseDice}
              />
              <FloatKnob
                label="Dice 가중치 (Dice weight)"
                desc="혼합 시 Dice 의 비중 (0=BCE만, 1=Dice만)"
                value={unetDiceWeight}
                onChange={setUnetDiceWeight}
                min={0} max={1} step={0.05}
                disabled={!unetUseDice}
              />

              <div className="pt-2 border-t border-gray-200">
                <label className="block text-xs">
                  <div className="mb-1 font-semibold">
                    가중치 손실맵 (Weight loss map)
                  </div>
                  <select
                    value={unetWmapMode}
                    onChange={(e) => setUnetWmapMode(e.target.value as any)}
                    className="w-full border border-gray-300 rounded px-2 py-1.5 text-xs bg-white"
                  >
                    <option value="none">사용 안 함 (None)</option>
                    <option value="distance">distance (SE 모드)</option>
                    <option value="intensity">intensity (STEM 모드)</option>
                  </select>
                  <div className="text-[10px] text-gray-500 mt-0.5 leading-snug">
                    Bals & Epple 식. 인접 입자 사이 좁은 경계 픽셀에 더 큰 손실 가중치
                  </div>
                </label>
                <div className="space-y-2 mt-2">
                  <FloatKnob
                    label="경계 가중치 진폭 (w0)"
                    desc="경계 픽셀에 부여할 최대 가중치. 보통 5~15"
                    value={unetW0}
                    onChange={setUnetW0}
                    min={0} max={50} step={0.5}
                    disabled={unetWmapMode === "none"}
                  />
                  <FloatKnob
                    label="거리 표준편차 (sigma)"
                    desc="경계로부터 가중치가 작아지는 속도. 작을수록 경계만 강조"
                    value={unetSigma}
                    onChange={setUnetSigma}
                    min={0.5} max={30} step={0.5}
                    disabled={unetWmapMode === "none"}
                  />
                  <FloatKnob
                    label="강도 가중치 (gamma)"
                    desc="픽셀 강도 기반 추가 가중. intensity 모드에서만 활성"
                    value={unetGamma} onChange={setUnetGamma}
                    min={0} max={10} step={0.1}
                    disabled={unetWmapMode !== "intensity"}
                  />
                </div>
              </div>
            </div>
          )}
        </aside>
      </div>
    </div>
  );
}

// ─────────────────── 다파일 드롭존 ───────────────────

function MultiFileDropzone({
  files, setFiles, label, emptyText, emptyHint,
}: {
  files: File[];
  setFiles: (fs: File[]) => void;
  label: string;
  emptyText: string;
  emptyHint?: string;
}) {
  const [hover, setHover] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  function add(newFiles: FileList | File[] | null) {
    if (!newFiles) return;
    const arr = Array.from(newFiles).filter((f) => f.type.startsWith("image/"));
    if (arr.length === 0) return;
    const merged = [...files];
    for (const f of arr) {
      if (!merged.some((g) => g.name === f.name && g.size === f.size)) {
        merged.push(f);
      }
    }
    setFiles(merged);
  }

  function remove(i: number) {
    setFiles(files.filter((_, idx) => idx !== i));
  }

  return (
    <div>
      <div className="text-[11.5px] font-medium mb-1.5 flex items-center justify-between">
        <span className="text-[#14161b]">{label}</span>
        {files.length > 0 && (
          <span className="text-[10px] text-[#5a606e] bg-[#f0f0e8] px-1.5 py-0.5 rounded-full font-mono">
            {files.length}장
          </span>
        )}
      </div>
      <div
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setHover(true); }}
        onDragLeave={() => setHover(false)}
        onDrop={(e) => {
          e.preventDefault();
          setHover(false);
          add(e.dataTransfer.files);
        }}
        className={
          "dropzone cursor-pointer text-center text-xs px-3 py-4 " +
          (hover ? "dropzone-active " : "") +
          (files.length > 0 ? "dropzone-filled " : "")
        }
      >
        {files.length === 0 ? (
          <div className="flex flex-col items-center justify-center gap-1.5">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" className="opacity-70">
              <rect x="3" y="3" width="18" height="18" rx="2"/>
              <circle cx="8.5" cy="8.5" r="1.5"/>
              <polyline points="21 15 16 10 5 21"/>
            </svg>
            <div className="font-semibold">{emptyText}</div>
            {emptyHint && <div className="text-[10.5px] opacity-75">{emptyHint}</div>}
          </div>
        ) : (
          <div className="flex items-center justify-center gap-1.5 text-[11px] text-[#128a4f] font-semibold">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
              <line x1="12" y1="5" x2="12" y2="19"/>
              <line x1="5" y1="12" x2="19" y2="12"/>
            </svg>
            더 추가 — 클릭 / 드래그
          </div>
        )}
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          multiple
          className="hidden"
          onChange={(e) => add(e.target.files)}
        />
      </div>

      {files.length > 0 && (
        <div className="mt-2 space-y-1 max-h-32 overflow-y-auto pr-0.5">
          {files.map((f, i) => (
            <div
              key={`${f.name}-${i}`}
              className="flex items-center justify-between gap-2 text-[11px]
                         px-2 py-1 bg-white border border-[#eeeee7] rounded-md
                         hover:border-[#d3d3cb] transition"
            >
              <span className="truncate text-[#14161b]" title={f.name}>{f.name}</span>
              <span className="text-[#8d92a0] flex-shrink-0 font-mono text-[10px]">
                {(f.size / 1024).toFixed(0)} KB
              </span>
              <button
                onClick={(e) => { e.stopPropagation(); remove(i); }}
                className="text-[#8d92a0] hover:text-[#c92a2a] flex-shrink-0 w-4 h-4 flex items-center justify-center rounded hover:bg-[#fff1f1] text-base leading-none"
                aria-label="제거"
              >
                ×
              </button>
            </div>
          ))}
          {files.length > 1 && (
            <button
              onClick={() => setFiles([])}
              className="text-[10px] text-[#8d92a0] hover:text-[#c92a2a] underline decoration-dotted"
            >
              모두 비우기 ({files.length}장)
            </button>
          )}
        </div>
      )}
    </div>
  );
}

// ─────────────────── 결과 뷰들 ───────────────────

function LegacyView({
  r, activeIdx, onPick,
}: {
  r: LegacyResult;
  activeIdx: number;
  onPick: (i: number) => void;
}) {
  const cur = r.results[Math.min(activeIdx, r.results.length - 1)];
  if (!cur) return null;
  return (
    <>
      <InfoCard
        rows={[
          ["이미지 수", `${r.n_images}장`],
          ...(r.n_with_gt > 0 ? [["GT 매칭", `${r.n_with_gt}/${r.n_images}장`] as [string, string]] : []),
          ["파라미터", r.params],
        ]}
      />
      {r.summary && (
        <MetricsCard
          m={r.summary}
          title={`평균 지표 (GT 매칭된 ${r.summary.n_with_gt}장 평균)`}
        />
      )}

      <div className="app-card p-3 min-w-0">
        <div className="flex items-baseline justify-between mb-2 gap-2 flex-wrap">
          <h2 className="text-sm font-semibold flex-shrink-0">
            결과 이미지 ·{" "}
            <span className="text-gray-500 font-normal">
              {activeIdx + 1} / {r.results.length}
            </span>
          </h2>
          <div className="text-[11px] text-gray-500 flex flex-wrap gap-x-2 min-w-0">
            <code className="truncate max-w-[200px]">{cur.filename}</code>
            <span>{cur.shape[1]} × {cur.shape[0]}</span>
            <span>입자 px <b>{cur.pred_pixels.toLocaleString()}</b></span>
            {cur.gt_filename && (
              <span className="text-green-700 truncate max-w-[200px]">GT: {cur.gt_filename}</span>
            )}
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
          <ImageBox label="후처리 결과" src={cur.pred_png} />
          {cur.gt_png && <ImageBox label="정답 마스크" src={cur.gt_png} />}
        </div>

        {r.results.length > 1 && (
          <ResultThumbStrip
            items={r.results.map(it => ({
              filename: it.filename,
              thumb: it.pred_png,
              info: `${it.pred_pixels.toLocaleString()} px`,
              hasGt: !!it.metrics,
            }))}
            activeIdx={activeIdx}
            onPick={onPick}
          />
        )}
      </div>

      {cur.metrics && (
        <MetricsCard
          m={cur.metrics}
          title={`이 이미지의 지표 — ${cur.filename}`}
        />
      )}
    </>
  );
}

function UnetView({
  r, probView, activeIdx, onPick,
}: {
  r: UnetResult;
  probView: "color" | "gray" | "filtered_color" | "filtered_gray";
  activeIdx: number;
  onPick: (i: number) => void;
}) {
  const cur = r.results[Math.min(activeIdx, r.results.length - 1)];
  if (!cur) return null;
  const probSrc =
    probView === "color" ? cur.prob_color_png :
    probView === "gray" ? cur.prob_png :
    probView === "filtered_color" ? cur.prob_filtered_png :
    cur.prob_gray_filtered_png;
  const probLabel = {
    color: "확률맵 (Jet)",
    gray: "확률맵 (회색조)",
    filtered_color: `확률맵 (≥${(r.min_prob * 100).toFixed(0)}% 표시)`,
    filtered_gray: `확률맵 회색조 (≥${(r.min_prob * 100).toFixed(0)}% 표시)`,
  }[probView];

  return (
    <>
      <InfoCard
        rows={[
          ["이미지 수", `${r.n_images}장`],
          ...(r.n_with_gt > 0 ? [["GT 매칭", `${r.n_with_gt}/${r.n_images}장`] as [string, string]] : []),
          ["모델", r.method],
          ["출처", r.weights_source],
          ["파라미터", r.params],
        ]}
      />
      {r.excel_data_url && (
        <div className="text-[11px] flex flex-wrap items-center gap-2">
          <a
            href={r.excel_data_url}
            download={`unet_results_${r.n_images}imgs.xlsx`}
            className="px-3 py-1.5 bg-emerald-50 border border-emerald-300 rounded
                       hover:bg-emerald-100 font-semibold text-emerald-800"
          >
            📥 엑셀 (.xlsx)
            {r.excel_size_bytes ? ` — ${(r.excel_size_bytes / 1024).toFixed(0)} KB` : ""}
          </a>
          <span className="text-gray-500">
            시트: Parameters / Metrics_per_image / Summary / Image_diagnostics
          </span>
        </div>
      )}
      {r.summary && (
        <MetricsCard
          m={r.summary}
          title={`평균 지표 (GT 매칭된 ${r.summary.n_with_gt}장 평균)`}
        />
      )}

      <div className="app-card p-3 min-w-0">
        <div className="flex items-baseline justify-between mb-2 gap-2 flex-wrap">
          <h2 className="text-sm font-semibold flex-shrink-0">
            결과 이미지 ·{" "}
            <span className="text-gray-500 font-normal">
              {activeIdx + 1} / {r.results.length}
            </span>
          </h2>
          <div className="text-[11px] text-gray-500 flex flex-wrap gap-x-2 min-w-0">
            <code className="truncate max-w-[200px]">{cur.filename}</code>
            <span>{cur.shape[1]} × {cur.shape[0]}</span>
            <span>입자 px <b>{cur.pred_pixels.toLocaleString()}</b></span>
            <span>표시 px {cur.filtered_pixels.toLocaleString()}</span>
            {cur.gt_filename && (
              <span className="text-green-700 truncate max-w-[200px]">GT: {cur.gt_filename}</span>
            )}
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
          <ImageBox label={probLabel} src={probSrc} />
          <ImageBox
            label={`컬러 인스턴스 (≥${(r.threshold * 100).toFixed(0)}%${
              cur.metrics ? ` · ${cur.metrics.n_pred_inst}개` : ""
            })`}
            src={cur.mask_color_png}
          />
          {cur.gt_png && <ImageBox label="정답 마스크" src={cur.gt_png} />}
        </div>

        {r.results.length > 1 && (
          <ResultThumbStrip
            items={r.results.map(it => ({
              filename: it.filename,
              thumb: it.mask_color_png,
              info: it.metrics
                ? `${it.metrics.n_pred_inst}개`
                : `${it.pred_pixels.toLocaleString()} px`,
              hasGt: !!it.metrics,
            }))}
            activeIdx={activeIdx}
            onPick={onPick}
          />
        )}
      </div>

      {cur.metrics && (
        <MetricsCard
          m={cur.metrics}
          title={`이 이미지의 지표 — ${cur.filename}`}
        />
      )}
    </>
  );
}

function UnetTrainProgressCard({ p }: { p: UnetTrainProgress }) {
  const pct = Math.round(p.progress * 100);
  const fmtTime = (s: number) => {
    if (!Number.isFinite(s) || s < 0) return "—";
    if (s < 60) return `${s.toFixed(0)}초`;
    const m = Math.floor(s / 60);
    const sec = Math.round(s % 60);
    return `${m}분 ${sec}초`;
  };
  return (
    <div className="app-card p-4 space-y-3 min-w-0">
      <div className="flex items-baseline justify-between gap-3 flex-wrap">
        <h2 className="text-sm font-semibold">학습 진행 중</h2>
        <span className="text-[11px] text-gray-500">
          상태: <code className={
            p.status === "running" ? "text-blue-600 font-semibold" :
            p.status === "pending" ? "text-amber-600 font-semibold" :
            p.status === "error" ? "text-red-600 font-semibold" :
            "text-green-600 font-semibold"
          }>
            {p.status === "pending" ? "준비 중" :
             p.status === "running" ? "학습 중" :
             p.status === "done" ? "완료" : "오류"}
          </code>
        </span>
      </div>

      {/* 진행 바 */}
      <div>
        <div className="flex justify-between text-[11px] mb-1">
          <span>에포크 <b>{p.epoch}</b> / {p.total_epochs}</span>
          <span className="font-mono font-semibold">{pct}%</span>
        </div>
        <div className="h-3 bg-gray-100 rounded-full overflow-hidden">
          <div
            className="h-full bg-[#1f5fd6] transition-all"
            style={{ width: `${pct}%` }}
          />
        </div>
      </div>

      {/* 상세 셀들 */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-1.5">
        {[
          ["경과", fmtTime(p.elapsed_sec)],
          ["남은 시간", p.eta_sec == null ? "—" : `약 ${fmtTime(p.eta_sec)}`],
          ["학습 손실",
            p.train_loss == null ? "—" : p.train_loss.toFixed(4)],
          ["검증 손실",
            p.val_loss == null ? "—" : p.val_loss.toFixed(4)],
          ["학습쌍", `${p.n_pairs}쌍`],
          ["미매칭", `${p.n_unmatched}장`],
          ["디바이스", p.device_request],
          ["job_id", p.job_id.slice(0, 8)],
        ].map(([k, v]) => (
          <div key={k} className="border border-gray-200 rounded px-2 py-1.5 min-w-0">
            <div className="text-[9px] uppercase tracking-wider text-gray-500 truncate">{k}</div>
            <div className="text-sm font-mono font-semibold truncate">{v}</div>
          </div>
        ))}
      </div>

      <div className="text-[10px] text-gray-400">
        진행률은 1.5초마다 자동 갱신됩니다. 창을 닫지 않아도 됩니다.
      </div>
    </div>
  );
}

function StatsProgressCard({ p }: { p: StatsJob }) {
  const pct = Math.round(p.progress * 100);
  const fmtTime = (s: number | null) => {
    if (s == null || !Number.isFinite(s) || s < 0) return "—";
    if (s < 60) return `${s.toFixed(0)}초`;
    const m = Math.floor(s / 60);
    const sec = Math.round(s % 60);
    return `${m}분 ${sec}초`;
  };
  const stageLabel: Record<string, string> = {
    queued: "대기 중",
    loading_weights: "가중치 로드",
    preparing: "추론 1회 + 캐시",
    sweeping: "Sweep 진행",
    rendering: "차트/엑셀 생성",
    done: "완료",
  };
  return (
    <div className="app-card p-4 space-y-3 min-w-0">
      <div className="flex items-baseline justify-between gap-3 flex-wrap">
        <h2 className="text-sm font-semibold">
          통계 Sweep 진행 중 — <span className="text-[#1f5fd6]">{p.sweep_mode.toUpperCase()}</span>
          <span className="text-gray-500"> · {p.sweep_param}</span>
        </h2>
        <span className="text-[11px] text-gray-500">
          {stageLabel[p.stage] || p.stage}
        </span>
      </div>
      <div>
        <div className="flex justify-between text-[11px] mb-1">
          <span><b>Step {p.current_step}</b> / {p.total_steps}</span>
          <span className="font-mono">{pct}%</span>
        </div>
        <div className="w-full bg-gray-200 rounded h-2.5 overflow-hidden">
          <div
            className="bg-[#1f5fd6] h-full transition-all"
            style={{ width: `${pct}%` }}
          />
        </div>
      </div>
      <div className="grid grid-cols-2 gap-2 text-[11px]">
        <div className="border border-gray-200 rounded px-2 py-1.5">
          <div className="text-[9px] uppercase tracking-wider text-gray-500">
            sweep value
          </div>
          <div className="font-mono">
            {p.current_value !== null ? p.current_value : "—"}
          </div>
        </div>
        <div className="border border-gray-200 rounded px-2 py-1.5">
          <div className="text-[9px] uppercase tracking-wider text-gray-500">
            current image
          </div>
          <div className="font-mono truncate" title={p.current_image ?? ""}>
            {p.current_image ?? "—"}
          </div>
        </div>
        <div className="border border-gray-200 rounded px-2 py-1.5">
          <div className="text-[9px] uppercase tracking-wider text-gray-500">elapsed</div>
          <div className="font-mono">{fmtTime(p.elapsed_sec)}</div>
        </div>
        <div className="border border-gray-200 rounded px-2 py-1.5">
          <div className="text-[9px] uppercase tracking-wider text-gray-500">eta</div>
          <div className="font-mono">{fmtTime(p.eta_sec)}</div>
        </div>
      </div>
      <div className="text-[10px] text-gray-500 space-y-0.5">
        <div>{p.n_sweep_values} sweep values × {p.n_with_gt} images (GT 매칭됨)</div>
        {(p.unet_method || p.yolo_model_name) && (
          <div>
            <span className="text-gray-400">모델: </span>
            {p.unet_method && (
              <code className="mr-2">
                {p.unet_method}
                {p.unet_image_size ? ` @ ${p.unet_image_size}px` : ""}
              </code>
            )}
            {p.yolo_model_name && (
              <code>{p.yolo_model_name}</code>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function ResearchProgressCard({ p }: { p: ResearchJob }) {
  const pct = Math.round(p.progress * 100);
  const fmtTime = (s: number | null) => {
    if (s == null || !Number.isFinite(s) || s < 0) return "—";
    if (s < 60) return `${s.toFixed(0)}초`;
    const m = Math.floor(s / 60);
    const sec = Math.round(s % 60);
    return `${m}분 ${sec}초`;
  };
  return (
    <div className="bg-white border border-purple-200 rounded-md p-4 space-y-3 min-w-0">
      <div className="flex items-baseline justify-between gap-3 flex-wrap">
        <h2 className="text-sm font-semibold">
          연구 (민감도 분석) 진행 중 — <span className="text-purple-700">{p.n_sweep_params} 파라미터</span>
          {p.has_grid && <span className="text-gray-500"> + 격자</span>}
        </h2>
        <span className="text-[11px] text-gray-500">{p.stage}</span>
      </div>
      <div>
        <div className="flex justify-between text-[11px] mb-1">
          <span><b>Step {p.current_step}</b> / {p.total_steps}</span>
          <span className="font-mono">{pct}%</span>
        </div>
        <div className="w-full bg-gray-200 rounded h-2.5 overflow-hidden">
          <div className="bg-purple-600 h-full transition-all" style={{ width: `${pct}%` }} />
        </div>
      </div>
      <div className="grid grid-cols-2 gap-2 text-[11px]">
        <div className="border border-gray-200 rounded px-2 py-1.5">
          <div className="text-[9px] uppercase tracking-wider text-gray-500">현재 파라미터</div>
          <div className="font-mono truncate" title={p.current_param ?? ""}>
            {p.current_param ?? "—"}
          </div>
        </div>
        <div className="border border-gray-200 rounded px-2 py-1.5">
          <div className="text-[9px] uppercase tracking-wider text-gray-500">현재 값</div>
          <div className="font-mono truncate" title={String(p.current_value ?? "")}>
            {p.current_value !== null ? String(p.current_value) : "—"}
          </div>
        </div>
        <div className="border border-gray-200 rounded px-2 py-1.5">
          <div className="text-[9px] uppercase tracking-wider text-gray-500">elapsed</div>
          <div className="font-mono">{fmtTime(p.elapsed_sec)}</div>
        </div>
        <div className="border border-gray-200 rounded px-2 py-1.5">
          <div className="text-[9px] uppercase tracking-wider text-gray-500">eta</div>
          <div className="font-mono">{fmtTime(p.eta_sec)}</div>
        </div>
      </div>
      {(p.unet_method || p.yolo_model_name) && (
        <div className="text-[10px] text-gray-500">
          모델:
          {p.unet_method && <code className="ml-1 mr-2">{p.unet_method}{p.unet_image_size ? ` @ ${p.unet_image_size}px` : ""}</code>}
          {p.yolo_model_name && <code>{p.yolo_model_name}</code>}
        </div>
      )}
    </div>
  );
}

function ResearchView({
  r, activeHeatmap, setActiveHeatmap,
}: {
  r: ResearchResult;
  activeHeatmap: keyof Metrics;
  setActiveHeatmap: (k: keyof Metrics) => void;
}) {
  const paramOrder: { key: string; label: string }[] = [
    { key: "high_thresh",         label: "τ_h (high_thresh)" },
    { key: "low_thresh",          label: "τ_l (low_thresh)" },
    { key: "balance_floor",       label: "β (balance)" },
    { key: "yolo_round_strength", label: "σ (yolo_round)" },
  ];
  const heatmapKeys: { key: keyof Metrics; label: string; note: string }[] = [
    { key: "ParticleCountError", label: "ParticleCountError", note: "낮을수록 좋음" },
    { key: "mAP@0.5",            label: "mAP@0.5",            note: "높을수록 좋음" },
    { key: "IoU",                label: "IoU",                note: "보조" },
  ];

  // 각 파라미터의 metric range (변동폭) — 평평할수록 robust
  function rangeOf(p: string, metric: keyof Metrics): number | null {
    const rows = r.sweeps[p] || [];
    const vals = rows.map(x => x.metrics?.[metric]).filter(v => v != null && isFinite(v as number)) as number[];
    if (vals.length < 2) return null;
    return Math.max(...vals) - Math.min(...vals);
  }

  return (
    <div className="space-y-3 min-w-0">
      <div className="bg-white border border-purple-200 rounded-md p-3 min-w-0">
        <div className="flex items-baseline justify-between mb-2 gap-2 flex-wrap">
          <div>
            <h2 className="text-sm font-semibold">
              연구 — 하이퍼파라미터 민감도 분석 (논문 3.5)
            </h2>
            <div className="text-[11px] text-gray-600 mt-0.5">
              GT {r.n_with_gt}/{r.n_images}장
              {(r.unet_method || r.yolo_model_name) && (
                <>  ·  <span className="text-gray-400">모델:</span>
                  {r.unet_method && <code className="ml-1 mr-2">{r.unet_method}{r.unet_image_size ? ` @ ${r.unet_image_size}px` : ""}</code>}
                  {r.yolo_model_name && <code>{r.yolo_model_name}</code>}
                </>
              )}
            </div>
          </div>
          {r.excel_data_url && (
            <a
              href={r.excel_data_url}
              download="research_sensitivity.xlsx"
              className="px-2.5 py-1.5 rounded text-[11px] font-semibold
                         bg-emerald-50 border border-emerald-300 text-emerald-800
                         hover:bg-emerald-100"
            >
              📥 엑셀 {r.excel_size_bytes ? `${(r.excel_size_bytes / 1024).toFixed(0)}KB` : ""}
            </a>
          )}
        </div>
      </div>

      {/* 1x4 패널 figure */}
      {r.panel_figure_png && (
        <div className="app-card p-3">
          <div className="text-xs font-semibold mb-2">
            1×4 패널 — 각 파라미터별 mAP@0.5 (파랑) + ParticleCountError (빨강 점선) · 검은 점선 = 운용값
          </div>
          <img
            src={r.panel_figure_png}
            alt="sensitivity panels"
            className="w-full max-w-full"
          />
          <div className="text-[10px] text-gray-500 mt-2 leading-snug">
            평평한 고원(plateau)이 보이면 robust — "특정 값에서만 좋다" 가 아니라
            "광범위한 값에서 비슷한 성능" 을 의미함. cherry-pick 의혹 차단.
          </div>
        </div>
      )}

      {/* 변동폭 요약 */}
      <div className="app-card p-3 overflow-x-auto">
        <h3 className="text-xs font-semibold mb-2">
          파라미터별 변동폭 (max − min, 작을수록 robust)
        </h3>
        <table className="text-[11px] border-collapse">
          <thead>
            <tr className="bg-gray-100">
              <th className="border border-gray-300 px-2 py-1">파라미터</th>
              <th className="border border-gray-300 px-2 py-1">sweep 값</th>
              {heatmapKeys.map(h => (
                <th key={h.key} className="border border-gray-300 px-2 py-1">
                  Δ {h.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {paramOrder.map(p => (
              <tr key={p.key}>
                <td className="border border-gray-300 px-2 py-1 font-semibold">{p.label}</td>
                <td className="border border-gray-300 px-2 py-1 font-mono text-[10px]">
                  [{(r.sweep_specs[p.key] || []).join(", ")}]
                </td>
                {heatmapKeys.map(h => {
                  const d = rangeOf(p.key, h.key);
                  return (
                    <td key={h.key}
                        className="border border-gray-300 px-2 py-1 text-right font-mono">
                      {d == null ? "—" : d.toFixed(4)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* 각 sweep 의 상세 표 */}
      {paramOrder.map(p => {
        const rows = r.sweeps[p.key] || [];
        if (!rows.length) return null;
        return (
          <div key={p.key} className="app-card p-3 overflow-x-auto">
            <h3 className="text-xs font-semibold mb-2">{p.label} sweep</h3>
            <table className="text-[11px] border-collapse">
              <thead>
                <tr className="bg-gray-100">
                  <th className="border border-gray-300 px-2 py-1">value</th>
                  {(["Precision","Recall","Accuracy","IoU","F1","mAP@0.5","ParticleCountError"] as const).map(mk => (
                    <th key={mk} className="border border-gray-300 px-2 py-1">{mk === "ParticleCountError" ? "CountErr" : mk}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.map((row, i) => (
                  <tr key={i}>
                    <td className="border border-gray-300 px-2 py-1 font-mono font-semibold">{row.value}</td>
                    {(["Precision","Recall","Accuracy","IoU","F1","mAP@0.5","ParticleCountError"] as const).map(mk => {
                      const v = row.metrics?.[mk];
                      return (
                        <td key={mk} className="border border-gray-300 px-2 py-1 text-right font-mono">
                          {v != null && isFinite(v as number) ? (v as number).toFixed(3) : "—"}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        );
      })}

      {/* 격자 heatmap */}
      {r.grid && Object.keys(r.heatmap_pngs).length > 0 && (
        <div className="app-card p-3 min-w-0">
          <div className="flex items-baseline justify-between mb-2">
            <h3 className="text-xs font-semibold">
              격자 heatmap — {r.grid.y_param} × {r.grid.x_param}
            </h3>
            <div className="flex gap-1">
              {heatmapKeys.map(h => (
                <button
                  key={h.key}
                  onClick={() => setActiveHeatmap(h.key)}
                  className={
                    "text-[11px] px-2 py-1 rounded border " +
                    (activeHeatmap === h.key
                      ? "bg-[#1f5fd6] text-white border-[#1f5fd6]"
                      : "bg-white border-gray-300 hover:border-gray-500")
                  }
                >
                  {h.label}
                </button>
              ))}
            </div>
          </div>
          {r.heatmap_pngs[activeHeatmap as string] && (
            <div className="flex justify-center">
              <img
                src={r.heatmap_pngs[activeHeatmap as string]}
                alt={activeHeatmap as string}
                className="max-w-full max-h-[60vh]"
              />
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function YoloTrainProgressCard({ p }: { p: YoloTrainProgress }) {
  const pct = Math.round(p.progress * 100);
  const fmtTime = (s: number) => {
    if (!Number.isFinite(s) || s < 0) return "—";
    if (s < 60) return `${s.toFixed(0)}초`;
    const m = Math.floor(s / 60);
    const sec = Math.round(s % 60);
    return `${m}분 ${sec}초`;
  };
  return (
    <div className="app-card p-4 space-y-3 min-w-0">
      <div className="flex items-baseline justify-between gap-3 flex-wrap">
        <h2 className="text-sm font-semibold">YOLO 학습 진행 중</h2>
        <span className="text-[11px] text-gray-500">
          상태: <code className={
            p.status === "running" ? "text-blue-600 font-semibold" :
            p.status === "pending" ? "text-amber-600 font-semibold" :
            p.status === "error" ? "text-red-600 font-semibold" :
            "text-green-600 font-semibold"
          }>
            {p.status === "pending" ? "준비 중" :
             p.status === "running" ? "학습 중" :
             p.status === "done" ? "완료" : "오류"}
          </code>
        </span>
      </div>

      <div>
        <div className="flex justify-between text-[11px] mb-1">
          <span>
            에포크 <b>{p.epoch}</b> / {p.total_epochs}
            {p.total_batches ? (
              <span className="text-gray-500 ml-2">
                · 배치 <b>{p.batch ?? 0}</b> / {p.total_batches}
              </span>
            ) : null}
          </span>
          <span className="font-mono font-semibold">{pct}%</span>
        </div>
        <div className="h-3 bg-gray-100 rounded-full overflow-hidden">
          <div
            className="h-full bg-[#1f5fd6] transition-all"
            style={{ width: `${pct}%` }}
          />
        </div>
        {p.total_batches && p.batch != null && p.total_batches > 0 && (
          <>
            <div className="text-[10px] text-gray-500 mt-1">
              현재 에포크 진행
            </div>
            <div className="h-1.5 bg-gray-100 rounded-full overflow-hidden mt-0.5">
              <div
                className="h-full bg-[#1f5fd6]/50 transition-all"
                style={{ width: `${(p.batch / p.total_batches) * 100}%` }}
              />
            </div>
          </>
        )}
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-1.5">
        {[
          ["경과", fmtTime(p.elapsed_sec)],
          ["남은 시간", p.eta_sec == null ? "—" : `약 ${fmtTime(p.eta_sec)}`],
          ["학습 손실", p.train_loss == null ? "—" : p.train_loss.toFixed(3)],
          ["검증 손실", p.val_loss == null ? "—" : p.val_loss.toFixed(3)],
          ["학습쌍", `${p.n_pairs}쌍`],
          ["미매칭", `${p.n_unmatched}장`],
          ["base", p.base_arch],
          ["job_id", p.job_id.slice(0, 8)],
        ].map(([k, v]) => (
          <div key={k} className="border border-gray-200 rounded px-2 py-1.5 min-w-0">
            <div className="text-[9px] uppercase tracking-wider text-gray-500 truncate">{k}</div>
            <div className="text-sm font-mono font-semibold truncate">{v}</div>
          </div>
        ))}
      </div>

      <div className="text-[10px] text-gray-400">
        배치 단위 실시간 갱신 (1.5초 간격). 첫 호출은 base 가중치 다운로드로 늦을 수 있습니다.
      </div>
    </div>
  );
}

function YoloTrainView({
  r, onUseForPredict,
}: {
  r: YoloTrainResult;
  onUseForPredict: () => void;
}) {
  const dlUrl = `${API_BASE}/api/export/yolo-model/${r.model_id}`;
  const lastEpoch = r.history[r.history.length - 1];
  const w = 320, h = 80, pad = 4;
  const tl = r.history.map(h => h.train_loss);
  const vl = r.history.map(h => h.val_loss);
  const all = [...tl, ...vl];
  const lo = Math.min(...all), hi = Math.max(...all);
  const span = hi - lo || 1;
  const xs = (i: number) =>
    pad + (i / Math.max(r.history.length - 1, 1)) * (w - 2 * pad);
  const ys = (v: number) => pad + (1 - (v - lo) / span) * (h - 2 * pad);
  const path = (arr: number[]) =>
    arr.map((v, i) => `${i === 0 ? "M" : "L"}${xs(i).toFixed(1)},${ys(v).toFixed(1)}`).join(" ");

  return (
    <div className="app-card p-3 space-y-3 min-w-0">
      <div className="flex items-baseline justify-between gap-3 flex-wrap">
        <h2 className="text-sm font-semibold">YOLO 학습 결과</h2>
        <div className="flex gap-2 flex-wrap">
          <button
            onClick={onUseForPredict}
            className="px-2.5 py-1.5 rounded text-[11px] font-semibold
                       bg-green-600 text-white hover:bg-green-700"
          >
            추론에 사용
          </button>
          <a
            href={dlUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="px-2.5 py-1.5 rounded text-[11px] font-semibold
                       bg-[#1f5fd6] text-white hover:bg-[#1849a8]"
          >
            모델 다운로드 (.pt)
          </a>
        </div>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-1.5">
        {[
          ["에포크", String(r.history.length)],
          ["base", r.base_arch],
          ["imgsz", `${r.imgsz}px`],
          ["소요시간", `${r.elapsed_sec.toFixed(1)}초`],
          ["디바이스", r.device],
          ["학습쌍", `${r.n_pairs}쌍`],
          ["미매칭", `${r.n_unmatched}장`],
          ["모델크기", `${(r.model_bytes / 1024 / 1024).toFixed(1)}MB`],
        ].map(([k, v]) => (
          <div key={k} className="border border-gray-200 rounded px-2 py-1.5">
            <div className="text-[9px] uppercase tracking-wider text-gray-500 truncate">{k}</div>
            <div className="text-sm font-mono font-semibold truncate">{v}</div>
          </div>
        ))}
      </div>

      {r.history.length > 1 && (
        <div>
          <div className="text-[10px] text-gray-500 mb-1 uppercase tracking-wide">손실 추이</div>
          <svg width={w} height={h} className="border border-gray-200 rounded bg-gray-50">
            <path d={path(tl)} fill="none" stroke="#1f5fd6" strokeWidth={1.5} />
            <path d={path(vl)} fill="none" stroke="#d6336c" strokeWidth={1.5} strokeDasharray="3 3" />
          </svg>
          <div className="text-[10px] text-gray-500 mt-1 flex gap-3">
            <span><span className="inline-block w-2 h-2 bg-[#1f5fd6] mr-1" />train</span>
            <span><span className="inline-block w-2 h-2 bg-[#d6336c] mr-1" />val</span>
            {lastEpoch && (
              <span className="ml-auto text-gray-700">
                마지막 — train {lastEpoch.train_loss.toFixed(3)} / val {lastEpoch.val_loss.toFixed(3)}
              </span>
            )}
          </div>
        </div>
      )}

      <div className="text-[11px] text-gray-500 break-all">
        <code>{r.params}</code>
      </div>
    </div>
  );
}

function UnetTrainView({
  r, onUseForPredict,
}: {
  r: UnetTrainResult;
  onUseForPredict: () => void;
}) {
  const dlUrl = `${API_BASE}/api/export/unet-model/${r.model_id}`;
  const lastEpoch = r.history[r.history.length - 1];
  // 손실 추이 — 작은 SVG 라인차트
  const w = 320, h = 80, pad = 4;
  const tl = r.history.map(h => h.train_loss);
  const vl = r.history.map(h => h.val_loss);
  const all = [...tl, ...vl];
  const lo = Math.min(...all), hi = Math.max(...all);
  const span = hi - lo || 1;
  const xs = (i: number) => pad + (i / Math.max(r.history.length - 1, 1)) * (w - 2 * pad);
  const ys = (v: number) => pad + (1 - (v - lo) / span) * (h - 2 * pad);
  const path = (arr: number[]) =>
    arr.map((v, i) => `${i === 0 ? "M" : "L"}${xs(i).toFixed(1)},${ys(v).toFixed(1)}`).join(" ");

  return (
    <div className="app-card p-3 space-y-3 min-w-0">
      <div className="flex items-baseline justify-between gap-3 flex-wrap">
        <h2 className="text-sm font-semibold">학습 결과</h2>
        <div className="flex gap-2 flex-wrap">
          <button
            onClick={onUseForPredict}
            className="px-2.5 py-1.5 rounded text-[11px] font-semibold
                       bg-green-600 text-white hover:bg-green-700"
          >
            추론에 사용
          </button>
          <a
            href={dlUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="px-2.5 py-1.5 rounded text-[11px] font-semibold
                       bg-[#1f5fd6] text-white hover:bg-[#1849a8]"
          >
            모델 다운로드 (.pt)
          </a>
        </div>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-1.5">
        {[
          ["에포크", String(r.history.length)],
          ["best_val", r.best_val_loss.toFixed(4)],
          ["소요시간", `${r.elapsed_sec.toFixed(1)}초`],
          ["디바이스", r.device],
          ["학습쌍", `${r.n_pairs}쌍`],
          ["미매칭", `${r.n_unmatched}장`],
          ["모델크기", `${(r.model_bytes / 1024 / 1024).toFixed(1)}MB`],
          ["model_id", r.model_id.slice(0, 8)],
        ].map(([k, v]) => (
          <div key={k} className="border border-gray-200 rounded px-2 py-1.5">
            <div className="text-[9px] uppercase tracking-wider text-gray-500 truncate">{k}</div>
            <div className="text-sm font-mono font-semibold truncate">{v}</div>
          </div>
        ))}
      </div>

      <div>
        <div className="text-[10px] text-gray-500 mb-1 uppercase tracking-wide">손실 추이</div>
        <svg width={w} height={h} className="border border-gray-200 rounded bg-gray-50">
          <path d={path(tl)} fill="none" stroke="#1f5fd6" strokeWidth={1.5} />
          <path d={path(vl)} fill="none" stroke="#d6336c" strokeWidth={1.5} strokeDasharray="3 3" />
        </svg>
        <div className="text-[10px] text-gray-500 mt-1 flex gap-3">
          <span><span className="inline-block w-2 h-2 bg-[#1f5fd6] mr-1" />train</span>
          <span><span className="inline-block w-2 h-2 bg-[#d6336c] mr-1" />val</span>
          {lastEpoch && (
            <span className="ml-auto text-gray-700">
              마지막 epoch — train {lastEpoch.train_loss.toFixed(4)} / val {lastEpoch.val_loss.toFixed(4)}
            </span>
          )}
        </div>
      </div>

      <div className="text-[11px] text-gray-500 break-all">
        <code>{r.params}</code>
      </div>
    </div>
  );
}

function YoloView({
  r, activeIdx, onPick,
}: {
  r: YoloResult;
  activeIdx: number;
  onPick: (i: number) => void;
}) {
  const cur = r.results[Math.min(activeIdx, r.results.length - 1)];
  if (!cur) return null;
  const pptUrl = r.result_id
    ? `${API_BASE}/api/export/yolo-pptx/${r.result_id}`
    : null;
  return (
    <>
      <div className="app-card p-3 flex items-start justify-between gap-2 min-w-0">
        <div className="text-xs text-gray-700 space-y-0.5 min-w-0 flex-1">
          <div className="flex flex-wrap gap-x-3 gap-y-0.5">
            <span><span className="text-gray-500">이미지 </span><code>{r.n_images}장</code></span>
            <span><span className="text-gray-500">검출 </span><code>{r.total_detections.toLocaleString()}</code></span>
            {r.n_with_gt > 0 && (
              <span><span className="text-gray-500">GT </span><code>{r.n_with_gt}/{r.n_images}</code></span>
            )}
          </div>
          <div className="text-[11px] break-all line-clamp-2">
            <span className="text-gray-500">파라미터 </span><code>{r.params}</code>
          </div>
        </div>
        <div className="flex flex-col gap-1 flex-shrink-0">
          {r.excel_data_url && (
            <a
              href={r.excel_data_url}
              download={`yolo_results_${r.n_images}imgs.xlsx`}
              className="px-2.5 py-1.5 rounded text-[11px] font-semibold
                         bg-emerald-50 border border-emerald-300 text-emerald-800
                         hover:bg-emerald-100 whitespace-nowrap"
            >
              📥 엑셀 (.xlsx)
              {r.excel_size_bytes ? ` ${(r.excel_size_bytes / 1024).toFixed(0)}KB` : ""}
            </a>
          )}
          {pptUrl && (
            <a
              href={pptUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="px-2.5 py-1.5 rounded text-[11px] font-semibold
                         bg-[#1f5fd6] text-white hover:bg-[#1849a8]
                         whitespace-nowrap"
            >
              PPT 다운로드
            </a>
          )}
        </div>
      </div>

      {r.summary && (
        <MetricsCard
          m={r.summary}
          title={`평균 지표 (GT 매칭된 ${r.summary.n_with_gt}장 평균)`}
        />
      )}

      <div className="app-card p-3 min-w-0">
        <div className="flex items-baseline justify-between mb-2 gap-2 flex-wrap min-w-0">
          <h2 className="text-sm font-semibold flex-shrink-0">
            결과 이미지 ·{" "}
            <span className="text-gray-500 font-normal">
              {activeIdx + 1} / {r.results.length}
            </span>
          </h2>
          <div className="text-[11px] text-gray-500 min-w-0 flex flex-wrap gap-x-2">
            <code className="truncate max-w-[200px]">{cur.filename}</code>
            <span>입자 <b>{cur.n_detections}</b></span>
            {cur.mean_conf > 0 && (
              <span>conf {cur.mean_conf.toFixed(3)}</span>
            )}
            {cur.gt_filename && (
              <span className="text-green-700 truncate max-w-[200px]">GT: {cur.gt_filename}</span>
            )}
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
          <ImageBox label="컬러 인스턴스" src={cur.overlay_png} />
          <ImageBox label="합쳐진 마스크" src={cur.mask_png} />
          {cur.gt_png && <ImageBox label="정답 마스크" src={cur.gt_png} />}
        </div>

        {r.results.length > 1 && (
          <YoloThumbStrip
            items={r.results}
            activeIdx={activeIdx}
            onPick={onPick}
          />
        )}
      </div>

      {cur.metrics && (
        <MetricsCard
          m={cur.metrics}
          title={`이 이미지의 지표 — ${cur.filename}`}
        />
      )}
    </>
  );
}

function ResultThumbStrip({
  items, activeIdx, onPick,
}: {
  items: { filename: string; thumb: string; info: string; hasGt: boolean }[];
  activeIdx: number;
  onPick: (i: number) => void;
}) {
  return (
    <div className="mt-4 pt-3 border-t border-gray-200 min-w-0">
      <div className="text-[11px] text-gray-500 mb-2">
        이미지 클릭으로 전환 ({items.length}장) · 좌우로 스크롤
      </div>
      <div className="flex gap-2 overflow-x-auto pb-2 scroll-smooth"
           style={{ scrollbarWidth: "thin" }}>
        {items.map((it, i) => (
          <button
            key={i}
            onClick={() => onPick(i)}
            title={`${it.filename} · ${it.info}` + (it.hasGt ? " · GT매칭" : "")}
            className={
              "flex-shrink-0 w-24 rounded overflow-hidden border-2 transition-all relative " +
              (i === activeIdx
                ? "border-[#1f5fd6] ring-2 ring-[#1f5fd6]/30"
                : "border-transparent hover:border-gray-400")
            }
          >
            <img
              src={it.thumb}
              alt={it.filename}
              className="w-full h-16 object-cover bg-black"
            />
            <div className="text-[9px] px-1 py-0.5 bg-gray-50 text-gray-600 truncate">
              #{i + 1} · {it.info}
            </div>
            {it.hasGt && (
              <span className="absolute top-0.5 right-0.5 px-1 text-[8px] rounded
                               bg-green-600 text-white font-bold">GT</span>
            )}
          </button>
        ))}
      </div>
    </div>
  );
}

function YoloThumbStrip({
  items, activeIdx, onPick,
}: {
  items: YoloItem[];
  activeIdx: number;
  onPick: (i: number) => void;
}) {
  return (
    <div className="mt-4 pt-3 border-t border-gray-200 min-w-0">
      <div className="text-[11px] text-gray-500 mb-2">
        이미지 클릭으로 전환 ({items.length}장) · 좌우로 스크롤
      </div>
      <div className="flex gap-2 overflow-x-auto pb-2 scroll-smooth"
           style={{ scrollbarWidth: "thin" }}>
        {items.map((it, i) => (
          <button
            key={i}
            onClick={() => onPick(i)}
            title={`${it.filename} · 입자 ${it.n_detections}` + (it.metrics ? " · GT매칭" : "")}
            className={
              "flex-shrink-0 w-24 rounded overflow-hidden border-2 transition-all relative " +
              (i === activeIdx
                ? "border-[#1f5fd6] ring-2 ring-[#1f5fd6]/30"
                : "border-transparent hover:border-gray-400")
            }
          >
            <img
              src={it.overlay_png}
              alt={it.filename}
              className="w-full h-16 object-cover bg-black"
            />
            <div className="text-[9px] px-1 py-0.5 bg-gray-50 text-gray-600 truncate">
              #{i + 1} · {it.n_detections}
            </div>
            {it.metrics && (
              <span className="absolute top-0.5 right-0.5 px-1 text-[8px] rounded
                               bg-green-600 text-white font-bold">GT</span>
            )}
          </button>
        ))}
      </div>
    </div>
  );
}

// ─────────────────── 공통 UI ───────────────────

function InfoCard({ rows }: { rows: [string, string | number][] }) {
  return (
    <div className="app-card p-3 min-w-0 fade-in">
      <div className="flex flex-wrap gap-x-4 gap-y-1 text-[12px] text-[#5a606e]">
        {rows.map(([k, v]) => (
          <div key={k} className="min-w-0">
            <span className="text-[#8d92a0] mr-1">{k}</span>
            <code className="text-[#14161b] font-mono text-[11.5px] break-all">{v}</code>
          </div>
        ))}
      </div>
    </div>
  );
}

function MetricsCard({ m, title }: { m: Metrics; title: string }) {
  // 논문 2.3절 기준 7개 평가 지표
  // higher is better except ParticleCountError
  const ratings: { en: string; ko?: string; v: number; display: string; better: "high" | "low" }[] = [
    { en: "Precision", ko: "정밀도",   v: m.Precision, display: m.Precision.toFixed(4), better: "high" },
    { en: "Recall",    ko: "재현율",   v: m.Recall,    display: m.Recall.toFixed(4),    better: "high" },
    { en: "Accuracy",  ko: "정확도",   v: m.Accuracy,  display: m.Accuracy.toFixed(4),  better: "high" },
    { en: "IoU",       ko: undefined,  v: m.IoU,       display: m.IoU.toFixed(4),       better: "high" },
    { en: "F1 Score",  ko: undefined,  v: m.F1,        display: m.F1.toFixed(4),        better: "high" },
    { en: "mAP@0.5",   ko: undefined,  v: m["mAP@0.5"], display: m["mAP@0.5"].toFixed(4), better: "high" },
    { en: "Count Error", ko: "입자수오차", v: m.ParticleCountError,
      display: `${(m.ParticleCountError * 100).toFixed(2)}%`, better: "low" },
  ];
  // minimal coloring: only highlight extremes (good/bad), neutral otherwise
  const tier = (v: number, better: "high" | "low") => {
    if (!isFinite(v)) return { c: "#9ca3af", bg: "transparent" };
    if (better === "high") {
      if (v >= 0.85) return { c: "#009f50", bg: "transparent" };
      if (v >= 0.40) return { c: "#191919", bg: "transparent" };
      return { c: "#e8483c", bg: "transparent" };
    } else {
      if (v <= 0.05) return { c: "#009f50", bg: "transparent" };
      if (v <= 0.30) return { c: "#191919", bg: "transparent" };
      return { c: "#e8483c", bg: "transparent" };
    }
  };
  return (
    <div className="app-card p-3.5 space-y-2.5 min-w-0 fade-in">
      <h2 className="text-[13px] font-semibold truncate text-[#14161b]">{title}</h2>
      <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-1.5">
        {ratings.map((cell) => {
          const t = tier(cell.v, cell.better);
          return (
            <div
              key={cell.en}
              className="rounded-md px-2 py-1.5 min-w-0"
              style={{
                border: "1px solid var(--c-border)",
                background: t.bg,
              }}
            >
              <div
                className="text-[9px] uppercase tracking-wider truncate font-semibold"
                style={{ color: "#8d92a0" }}
                title={`${cell.en}${cell.ko ? " · " + cell.ko : ""}`}
              >
                {cell.en}
              </div>
              {cell.ko && (
                <div className="text-[9px] text-[#a8aebb] truncate">{cell.ko}</div>
              )}
              <div className="text-[14px] font-mono font-semibold truncate" style={{ color: t.c }}>
                {cell.display}
              </div>
            </div>
          );
        })}
      </div>
      <div className="text-[10.5px] text-[#5a606e] pt-1 flex flex-wrap items-center gap-x-3 gap-y-0.5">
        <span>GT 입자수 <b className="text-[#14161b]">{m.n_gt_inst}</b></span>
        <span>Pred 입자수 <b className="text-[#14161b]">{m.n_pred_inst}</b></span>
        <span className="text-[#a8aebb]">connected components 기준</span>
      </div>
    </div>
  );
}

function SingleFileDropzone({
  label, hint, file, onChange, accept, isDefault, onRevertToDefault,
}: {
  label: string;
  hint?: string;
  file: File | null;
  onChange: (f: File | null) => void;
  accept?: string;
  /** 현재 file 이 자동 로드된 기본 모델일 때 true → "기본" 배지 표시 */
  isDefault?: boolean;
  /** 사용자가 교체한 상태에서 기본 모델로 되돌릴 수 있을 때 콜백 */
  onRevertToDefault?: () => void;
}) {
  const [hover, setHover] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  function pick(fs: FileList | File[] | null) {
    if (!fs) return;
    const arr = Array.from(fs);
    if (arr.length === 0) return;
    onChange(arr[0]);
  }

  return (
    <div>
      <div className="text-[11.5px] font-medium mb-1.5 text-[#14161b]">{label}</div>
      <div
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setHover(true); }}
        onDragLeave={() => setHover(false)}
        onDrop={(e) => {
          e.preventDefault();
          setHover(false);
          pick(e.dataTransfer.files);
        }}
        className={
          "dropzone cursor-pointer text-center px-3 py-3 min-w-0 " +
          (hover ? "dropzone-active " : "") +
          (file ? "dropzone-filled " : "")
        }
      >
        {file ? (
          <div className="min-w-0 flex items-center gap-2 text-left">
            <div className="w-7 h-7 rounded-md flex items-center justify-center flex-shrink-0"
                 style={{ background: isDefault ? "rgba(25,25,25,0.08)" : "rgba(18,138,79,0.12)" }}>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke={isDefault ? "#191919" : "#128a4f"} strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="20 6 9 17 4 12"/>
              </svg>
            </div>
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-1.5 min-w-0">
                <div className={"text-[11.5px] font-semibold truncate " + (isDefault ? "text-[#191919]" : "text-[#0d6e3f]")} title={file.name}>
                  {file.name}
                </div>
                {isDefault && (
                  <span className="text-[8.5px] font-bold px-1.5 py-[1px] rounded-full bg-[#191919] text-white tracking-wider flex-shrink-0">
                    기본
                  </span>
                )}
              </div>
              <div className="text-[10px] text-[#5a606e] mt-0.5 flex items-center gap-2 flex-wrap">
                <span className="font-mono">{(file.size / 1024).toFixed(1)} KB</span>
                <button
                  onClick={(e) => { e.stopPropagation(); onChange(null); }}
                  className="text-[#8d92a0] hover:text-[#c92a2a] underline decoration-dotted"
                >
                  교체
                </button>
                {onRevertToDefault && (
                  <button
                    onClick={(e) => { e.stopPropagation(); onRevertToDefault(); }}
                    className="text-[#8d92a0] hover:text-[#191919] underline decoration-dotted"
                  >
                    기본값으로
                  </button>
                )}
              </div>
            </div>
          </div>
        ) : (
          <div className="flex items-center justify-center gap-2 text-[11.5px] text-[#5a606e]">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" className="opacity-70 flex-shrink-0">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
              <polyline points="17 8 12 3 7 8"/>
              <line x1="12" y1="3" x2="12" y2="15"/>
            </svg>
            <div className="text-left min-w-0">
              <div className="font-semibold">파일을 끌어다 놓거나 클릭</div>
              {hint && <div className="text-[10px] opacity-75 truncate">{hint}</div>}
              {onRevertToDefault && (
                <button
                  onClick={(e) => { e.stopPropagation(); onRevertToDefault(); }}
                  className="text-[10px] text-[#191919] underline decoration-dotted mt-0.5"
                >
                  기본 모델 다시 불러오기
                </button>
              )}
            </div>
          </div>
        )}
        <input
          ref={inputRef}
          type="file"
          accept={accept}
          className="hidden"
          onChange={(e) => pick(e.target.files)}
        />
      </div>
    </div>
  );
}

function KnobValueInput({
  value, min, max, step, disabled, format, onCommit,
}: {
  value: number;
  min: number;
  max: number;
  step: number;
  disabled?: boolean;
  /** non-focused display (e.g. integer or fixed decimals) */
  format: (v: number) => string;
  onCommit: (v: number) => void;
}) {
  const [text, setText] = useState<string>(format(value));
  const [focused, setFocused] = useState(false);
  useEffect(() => {
    if (!focused) setText(format(value));
  }, [value, focused, format]);

  const commit = () => {
    const parsed = parseFloat(text.replace(/,/g, ""));
    let v = Number.isFinite(parsed) ? parsed : value;
    v = Math.max(min, Math.min(max, v));
    // step 정수배로 스냅 — 슬라이더와 일관되게.
    if (step > 0) v = Math.round(v / step) * step;
    // 부동소수점 잡음 정리
    const digits = step < 1 ? Math.min(6, Math.max(0, -Math.floor(Math.log10(step)))) : 0;
    v = Number(v.toFixed(digits));
    onCommit(v);
    setText(format(v));
  };

  return (
    <input
      type="text"
      inputMode="decimal"
      value={text}
      disabled={disabled}
      onChange={(e) => setText(e.target.value)}
      onFocus={(e) => {
        setFocused(true);
        // 비포커스 표시값(예: 0.25) 대신 raw value 로 — 정밀 편집 용이
        setText(String(value));
        requestAnimationFrame(() => e.target.select());
      }}
      onBlur={() => { commit(); setFocused(false); }}
      onKeyDown={(e) => {
        if (e.key === "Enter") { (e.target as HTMLInputElement).blur(); }
        else if (e.key === "Escape") {
          setText(format(value));
          (e.target as HTMLInputElement).blur();
        }
      }}
      onClick={(e) => e.stopPropagation()}
      className="knob-num font-mono text-[12px] text-right bg-transparent w-16 px-1.5 py-0.5 rounded-md border border-transparent text-[#6b7280] hover:border-[#e6e6e2] focus:border-[#191919] focus:text-[#191919] focus:bg-white"
    />
  );
}

function NumberKnob({
  label, value, onChange, min, max, step = 1, disabled, desc,
}: {
  label: string;
  value: number;
  onChange: (n: number) => void;
  min: number;
  max: number;
  step?: number;
  disabled?: boolean;
  desc?: string;
}) {
  return (
    <div className={"block text-xs " + (disabled ? "opacity-50" : "")}>
      <div className="flex justify-between items-center mb-1 gap-2">
        <span className="min-w-0 truncate">{label}</span>
        <KnobValueInput
          value={value} min={min} max={max} step={step}
          disabled={disabled}
          format={(v) => String(Math.round(v))}
          onCommit={onChange}
        />
      </div>
      <input
        type="range"
        min={min} max={max} step={step}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full"
      />
      {desc && <div className="text-[10px] text-gray-500 mt-0.5 leading-snug">{desc}</div>}
    </div>
  );
}

function FloatKnob(props: {
  label: string;
  value: number;
  onChange: (n: number) => void;
  min: number; max: number; step: number;
  disabled?: boolean;
  desc?: string;
}) {
  // 표시 자리수 — step 으로부터 자동 결정 (0.01 → 2자리, 0.1 → 1자리)
  const digits = props.step < 1
    ? Math.min(6, Math.max(0, -Math.floor(Math.log10(props.step))))
    : 2;
  return (
    <div className={"block text-xs " + (props.disabled ? "opacity-50" : "")}>
      <div className="flex justify-between items-center mb-1 gap-2">
        <span className="min-w-0 truncate">{props.label}</span>
        <KnobValueInput
          value={props.value} min={props.min} max={props.max} step={props.step}
          disabled={props.disabled}
          format={(v) => v.toFixed(digits)}
          onCommit={props.onChange}
        />
      </div>
      <input
        type="range"
        min={props.min} max={props.max} step={props.step}
        value={props.value}
        disabled={props.disabled}
        onChange={(e) => props.onChange(Number(e.target.value))}
        className="w-full"
      />
      {props.desc && <div className="text-[10px] text-gray-500 mt-0.5 leading-snug">{props.desc}</div>}
    </div>
  );
}

function Toggle({
  label, value, onChange, desc,
}: {
  label: string;
  value: boolean;
  onChange: (b: boolean) => void;
  desc?: string;
}) {
  return (
    <label className="block text-xs">
      <div className="flex items-center justify-between">
        <span>{label}</span>
        <input
          type="checkbox"
          checked={value}
          onChange={(e) => onChange(e.target.checked)}
        />
      </div>
      {desc && <div className="text-[10px] text-gray-500 mt-0.5 leading-snug">{desc}</div>}
    </label>
  );
}

function StatsView({
  r, activeChart, setActiveChart,
}: {
  r: StatsResult;
  activeChart: keyof Metrics;
  setActiveChart: (k: keyof Metrics) => void;
}) {
  const METRIC_KEYS: (keyof Metrics)[] = [
    "Precision", "Recall", "Accuracy", "IoU", "F1",
    "mAP@0.5", "ParticleCountError",
  ];
  const allMethods: { key: "yolo" | "unet_low" | "hybrid"; label: string; color: string }[] = [
    { key: "yolo",     label: "YOLO only",   color: "#dc2626" },
    { key: "unet_low", label: "U-Net (low)", color: "#16a34a" },
    { key: "hybrid",   label: "Hybrid",      color: "#1f5fd6" },
  ];
  const mode = r.sweep_mode || "hybrid";
  const methodOrder = mode === "yolo"
    ? allMethods.filter(m => m.key === "yolo")
    : mode === "unet"
      ? allMethods.filter(m => m.key === "unet_low")
      : allMethods;

  // 각 지표별 최고값을 가진 (value, method) 찾기
  function bestFor(metric: keyof Metrics): { value: number; method: string; v: number } | null {
    let best: { value: number; method: string; v: number } | null = null;
    for (const row of r.rows) {
      for (const m of methodOrder) {
        const val = row[m.key]?.[metric];
        if (val === undefined || val === null || !isFinite(val as number)) continue;
        // ParticleCountError 는 작을수록 좋음
        const better = metric === "ParticleCountError"
          ? best == null || (val as number) < best.v
          : best == null || (val as number) > best.v;
        if (better) best = { value: row.value, method: m.label, v: val as number };
      }
    }
    return best;
  }

  return (
    <div className="space-y-3 min-w-0">
      <div className="app-card p-3 min-w-0">
        <div className="flex items-baseline justify-between mb-2 gap-2 flex-wrap">
          <div>
            <h2 className="text-sm font-semibold">
              통계 sweep · <b className="text-[#1f5fd6]">{(r.sweep_mode || "hybrid").toUpperCase()}</b>
              <span className="text-gray-500"> — {r.sweep_param}  ·  {r.sweep_values.length} values  ·  GT {r.n_with_gt}/{r.n_images}장</span>
            </h2>
            {(r.unet_method || r.yolo_model_name) && (
              <div className="text-[11px] text-gray-600 mt-0.5">
                <span className="text-gray-400">모델: </span>
                {r.unet_method && (
                  <code className="mr-2">
                    {r.unet_method}{r.unet_image_size ? ` @ ${r.unet_image_size}px` : ""}
                  </code>
                )}
                {r.yolo_model_name && (
                  <code>{r.yolo_model_name}</code>
                )}
              </div>
            )}
          </div>
          <div className="flex items-center gap-2">
            {r.excel_data_url && (
              <a
                href={r.excel_data_url}
                download={`stats_${r.sweep_param}.xlsx`}
                className="px-2.5 py-1.5 rounded text-[11px] font-semibold
                           bg-emerald-50 border border-emerald-300 text-emerald-800
                           hover:bg-emerald-100"
              >
                📥 엑셀
                {r.excel_size_bytes ? ` ${(r.excel_size_bytes / 1024).toFixed(0)}KB` : ""}
              </a>
            )}
          </div>
        </div>
        <code className="text-[10px] text-gray-500 truncate block">{r.params}</code>
      </div>

      {/* 지표별 차트 selector */}
      <div className="app-card p-3 min-w-0">
        <div className="flex flex-wrap gap-1 mb-2">
          {METRIC_KEYS.map((mk) => (
            <button
              key={mk}
              onClick={() => setActiveChart(mk)}
              className={
                "text-[11px] px-2 py-1 rounded border " +
                (activeChart === mk
                  ? "bg-[#1f5fd6] text-white border-[#1f5fd6]"
                  : "bg-white border-gray-300 hover:border-gray-500")
              }
            >
              {mk}
            </button>
          ))}
        </div>
        <div className="bg-white border border-gray-100 rounded flex justify-center">
          {r.chart_pngs[activeChart] && (
            <img
              src={r.chart_pngs[activeChart]}
              alt={activeChart}
              className="max-w-full max-h-[55vh] object-contain"
            />
          )}
        </div>
        {(() => {
          const b = bestFor(activeChart);
          if (!b) return null;
          return (
            <div className="text-[11px] text-gray-700 mt-2">
              최고값 — <b>{b.method}</b> @ {r.sweep_param}={b.value} :
              <span className="font-mono ml-1">{b.v.toFixed(4)}</span>
              {activeChart === "ParticleCountError" && (
                <span className="text-gray-500"> (낮을수록 좋음)</span>
              )}
            </div>
          );
        })()}
      </div>

      {/* sweep 표 */}
      <div className="app-card p-3 min-w-0 overflow-x-auto">
        <h3 className="text-xs font-semibold mb-2">Sweep 표 — 모든 지표 ({methodOrder.length} methods)</h3>
        <table className="text-[11px] border-collapse min-w-full">
          <thead>
            <tr className="bg-gray-100">
              <th className="border border-gray-300 px-2 py-1 sticky left-0 bg-gray-100">{r.sweep_param}</th>
              {methodOrder.map((m) => (
                <th key={m.key} colSpan={METRIC_KEYS.length}
                    className="border border-gray-300 px-2 py-1 text-center"
                    style={{ color: m.color }}>
                  {m.label}
                </th>
              ))}
            </tr>
            <tr className="bg-gray-50 text-[10px]">
              <th className="border border-gray-300 px-2 py-1 sticky left-0 bg-gray-50"></th>
              {methodOrder.flatMap((m) =>
                METRIC_KEYS.map((mk) => (
                  <th key={`${m.key}-${mk}`} className="border border-gray-300 px-1 py-0.5 font-normal">
                    {mk === "ParticleCountError" ? "CE" : mk === "mAP@0.5" ? "mAP" : mk}
                  </th>
                ))
              )}
            </tr>
          </thead>
          <tbody>
            {r.rows.map((row) => (
              <tr key={row.value}>
                <td className="border border-gray-300 px-2 py-1 font-mono font-semibold sticky left-0 bg-white">
                  {row.value}
                </td>
                {methodOrder.flatMap((m) =>
                  METRIC_KEYS.map((mk) => {
                    const v = row[m.key]?.[mk];
                    return (
                      <td key={`${row.value}-${m.key}-${mk}`}
                          className="border border-gray-300 px-1 py-0.5 text-right font-mono">
                        {v !== undefined && v !== null && isFinite(v as number)
                          ? (v as number).toFixed(3) : "—"}
                      </td>
                    );
                  })
                )}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function HybridView({
  r, activeIdx, onPick, diagView, setDiagView, originalsMap,
  labelScale, setLabelScale, numberedSrc, numberedBusy,
}: {
  r: HybridResult;
  activeIdx: number;
  onPick: (i: number) => void;
  diagView: "original" | "overlay" | "instances" | "numbered" | "mask" | "prob" | "high" | "close" | "boundary" | "separator" | "color" | "hist";
  setDiagView: (v: "original" | "overlay" | "instances" | "numbered" | "mask" | "prob" | "high" | "close" | "boundary" | "separator" | "color" | "hist") => void;
  originalsMap: Record<string, string>;
  labelScale: number;
  setLabelScale: (n: number) => void;
  numberedSrc: string;
  numberedBusy: boolean;
}) {
  const it = r.results[activeIdx];
  if (!it) return null;
  const originalSrc = originalsMap[it.filename] ?? "";

  const diagOptions: [typeof diagView, string, string][] = [
    ["original", "원본", "입력 이미지"],
    ["overlay", "오버레이", "원본 + instance"],
    ["instances", "Instance", "입자별 색"],
    ["numbered", "번호", "1, 2 ... 라벨"],
    ["mask", "Mask", "semantic 이진"],
    ["prob", "U-Net prob", "확률맵"],
    ["high", "M_75 (anchor)", "확실한 입자"],
    ["close", "M_close (좁은 틈)", "거리 + Otsu"],
    ["boundary", "YOLO 경계", "분리선 후보"],
    ["separator", "최종 분리선", "M_close ∩ B"],
    ["color", "색 진단", "빨강=M_close, 파랑=B, 흰=둘 다(분리선)"],
    ["hist", "d₊ 히스토그램", "Otsu 임계 시각화 (논문 figure)"],
  ];
  const srcMap: Record<typeof diagView, string> = {
    original: originalSrc,
    overlay: it.overlay_png,
    instances: it.instances_png,
    numbered: numberedSrc || it.instances_png,
    mask: it.mask_png,
    prob: it.diag_prob_png,
    high: it.diag_high_png,
    close: it.diag_close_png,
    boundary: it.diag_boundary_png,
    separator: it.diag_separator_png,
    color: it.diag_overlap_png,
    hist: it.diag_hist_png ?? "",
  };

  return (
    <div className="space-y-3 min-w-0">
      <div className="app-card p-4 min-w-0">
        {/* 간결한 헤더 — 큰 숫자 위주 */}
        <div className="flex items-end justify-between gap-4 flex-wrap mb-3">
          <div className="flex items-end gap-5 flex-wrap">
            <div>
              <div className="text-[10px] uppercase tracking-wider text-[#9ca3af] font-semibold">이미지</div>
              <div className="text-[20px] font-bold text-[#191919] leading-none mt-0.5">
                {r.n_images}<span className="text-[12px] text-[#9ca3af] font-medium ml-0.5">장</span>
              </div>
            </div>
            <div>
              <div className="text-[10px] uppercase tracking-wider text-[#9ca3af] font-semibold">총 입자</div>
              <div className="text-[20px] font-bold text-[#191919] leading-none mt-0.5">
                {r.total_instances.toLocaleString()}<span className="text-[12px] text-[#9ca3af] font-medium ml-0.5">개</span>
              </div>
            </div>
            <div>
              <div className="text-[10px] uppercase tracking-wider text-[#9ca3af] font-semibold">GT 매칭</div>
              <div className={"text-[20px] font-bold leading-none mt-0.5 " + (r.n_with_gt === 0 ? "text-[#9ca3af]" : "text-[#191919]")}>
                {r.n_with_gt}<span className="text-[12px] text-[#9ca3af] font-medium ml-0.5">/{r.n_images}</span>
              </div>
            </div>
          </div>
        </div>

        {r.n_with_gt > 0 && r.n_with_gt < r.n_images && (
          <div className="text-[11px] text-[#b7791f] bg-[#fffaeb] border border-[#fde68a] rounded-md px-3 py-2 mb-3">
            {r.n_images - r.n_with_gt}장이 GT 와 파일명 매칭 실패. 매칭된 {r.n_with_gt}장만 평가됨.
          </div>
        )}

        {/* 파일 selector */}
        {r.results.length > 1 && (
          <div className="flex flex-wrap gap-1 mb-2">
            {r.results.map((x, i) => (
              <button
                key={i}
                onClick={() => onPick(i)}
                className={
                  "text-[11px] px-2.5 py-1 rounded-md border truncate max-w-[200px] " +
                  (i === activeIdx
                    ? "bg-[#191919] text-white border-[#191919]"
                    : "bg-white border-[#e6e6e2] text-[#6b7280] hover:border-[#191919] hover:text-[#191919]")
                }
                title={x.filename}
              >
                {x.filename}
              </button>
            ))}
          </div>
        )}

        {/* 뷰 selector */}
        <div className="flex flex-wrap gap-1 mb-3">
          {diagOptions.map(([id, label, sub]) => (
            <button
              key={id}
              onClick={() => setDiagView(id)}
              className={
                "text-[11px] px-2.5 py-1 rounded-md border " +
                (diagView === id
                  ? "bg-[#191919] text-white border-[#191919]"
                  : "bg-white border-[#e6e6e2] text-[#6b7280] hover:border-[#191919] hover:text-[#191919]")
              }
              title={sub}
            >
              {label}
            </button>
          ))}
        </div>

        <div className="text-[11px] text-[#6b7280] mb-2 flex flex-wrap gap-x-4 gap-y-0.5">
          <span>입자 수 <b className="text-[#191919]">{it.n_instances}</b></span>
          <span>YOLO instances <b className="text-[#191919]">{it.n_yolo}</b></span>
          <span>U-Net anchor <b className="text-[#191919]">{it.K_unet_anchor}</b></span>
          <span>τ (Otsu) <b className="text-[#191919]">{it.tau.toFixed(2)}</b></span>
        </div>

        {diagView === "numbered" && (it.particles?.length ?? 0) > 0 && (
          <div className="flex items-center gap-3 mb-2 px-3 py-2 rounded-md border border-[#f1f1ef] bg-[#fafafa] text-[11.5px]">
            <span className="text-[#6b7280] font-medium flex-shrink-0">번호 크기</span>
            <input
              type="range"
              min={0.3} max={3.5} step={0.05}
              value={labelScale}
              onChange={(e) => setLabelScale(Number(e.target.value))}
              className="flex-1"
            />
            <span className="font-mono text-[#191919] tabular-nums w-12 text-right text-[11px]">
              {labelScale.toFixed(2)}×
            </span>
            <button
              type="button"
              onClick={() => setLabelScale(1.0)}
              className="text-[10.5px] text-[#9ca3af] hover:text-[#191919] px-1.5 py-0.5 rounded hover:bg-white"
              title="기본값"
            >
              초기화
            </button>
            <span className={"text-[10px] text-[#9ca3af] tabular-nums flex-shrink-0 " + (numberedBusy ? "" : "opacity-0")}>
              그리는 중...
            </span>
          </div>
        )}

        <ImageBox label={diagOptions.find(o => o[0] === diagView)?.[1] ?? ""} src={srcMap[diagView]} />

        {/* 다운로드는 우측 패널 (파라미터 아래) 로 이동 */}

        {/* 실행 파라미터 — 접혀있고 펼쳐서 볼 수 있음 */}
        <details className="mt-3 text-[10.5px] text-[#9ca3af] group">
          <summary className="cursor-pointer select-none hover:text-[#6b7280] inline-flex items-center gap-1">
            <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" className="group-open:rotate-90 transition-transform">
              <polyline points="9 18 15 12 9 6"/>
            </svg>
            실행 파라미터 보기
          </summary>
          <code className="block mt-1.5 break-all bg-[#fafafa] border border-[#f1f1ef] rounded px-2 py-1.5 font-mono text-[10px] leading-relaxed">
            {r.params}
          </code>
        </details>
        {r.excel_error && (
          <div className="mt-2 text-[11px] text-[#a4262c] bg-[#fff5f5] border border-[#fed7d7] rounded-md px-3 py-2">
            엑셀 생성 실패: {r.excel_error}
          </div>
        )}

        {/* d_+ 히스토그램 + CSV 다운로드 */}
        {(it.hist_csv || it.pixel_csv) && (
          <div className="mt-2 flex flex-wrap items-center gap-2 text-[11px]">
            <span className="text-gray-500">데이터 다운로드:</span>
            {it.hist_csv && (
              <a
                href={it.hist_csv}
                download={`${it.filename.replace(/\.[^.]+$/, "")}_d_histogram.csv`}
                className="px-2 py-1 bg-blue-50 border border-blue-200 rounded hover:bg-blue-100"
              >
                📊 히스토그램 CSV (bin width=1, 엑셀 차트용)
              </a>
            )}
            {it.pixel_csv && (
              <a
                href={it.pixel_csv}
                download={`${it.filename.replace(/\.[^.]+$/, "")}_d_pixels.csv`}
                className="px-2 py-1 bg-blue-50 border border-blue-200 rounded hover:bg-blue-100"
              >
                📋 픽셀별 CSV (y, x, d₁, d₂, d₊, balance, in_M_close)
                {it.pixel_csv_rows ? ` — ${it.pixel_csv_rows.toLocaleString()} 행` : ""}
              </a>
            )}
          </div>
        )}

        {it.gt_png && (
          <div className="mt-2">
            <ImageBox label={`정답 (${it.gt_filename ?? ""})`} src={it.gt_png} />
          </div>
        )}
      </div>

      {/* 평가 지표 — 새 백엔드는 세 방법별, 옛 백엔드는 hybrid 단일 metrics 만 */}
      {(() => {
        const lowPct = Math.round(((r.low_thresh ?? 0.30) * 100));
        const unetLabel = `② U-Net ≥ ${lowPct}%`;
        return (
          <>
            {(it.metrics_yolo || it.metrics_unet_low || it.metrics_hybrid || it.metrics) ? (
              <div className="space-y-2">
                <div className="text-xs font-semibold text-gray-700">
                  이미지별 지표 — {it.filename}
                </div>
                {it.metrics_yolo && (
                  <MetricsCard m={it.metrics_yolo} title="① YOLO 만 (라운드/overlap 후처리 적용)" />
                )}
                {it.metrics_unet_low && (
                  <MetricsCard m={it.metrics_unet_low} title={unetLabel} />
                )}
                {(it.metrics_hybrid || it.metrics) && (
                  <MetricsCard
                    m={(it.metrics_hybrid ?? it.metrics)!}
                    title={it.metrics_yolo
                      ? "③ Hybrid (최종)"
                      : "Hybrid (최종) — 백엔드 재시작 시 ①②③ 비교 표시"}
                  />
                )}
              </div>
            ) : it.gt_filename ? (
              <div className="text-[11px] text-amber-700 bg-amber-50 border border-amber-200 rounded p-2">
                ⚠️ GT 매칭 완료 ({it.gt_filename}) 인데 평가 지표 누락 — 백엔드 재시작 필요할 수도.
              </div>
            ) : null}
            {(r.summary_yolo || r.summary_unet_low || r.summary_hybrid || r.summary) && (
              <div className="space-y-2">
                <div className="text-xs font-semibold text-gray-700">
                  전체 평균 ({(r.summary_hybrid ?? r.summary ?? r.summary_yolo ?? r.summary_unet_low)?.n_with_gt ?? 0}장)
                </div>
                {r.summary_yolo && (
                  <MetricsCard m={r.summary_yolo as any} title="① YOLO 만 — 평균" />
                )}
                {r.summary_unet_low && (
                  <MetricsCard m={r.summary_unet_low as any} title={`${unetLabel} — 평균`} />
                )}
                {(r.summary_hybrid || r.summary) && (
                  <MetricsCard
                    m={(r.summary_hybrid ?? r.summary) as any}
                    title={r.summary_yolo ? "③ Hybrid — 평균" : "Hybrid — 평균"}
                  />
                )}
              </div>
            )}
          </>
        );
      })()}
    </div>
  );
}

function InferenceProgressCard({
  label, total, startMs, tick, firstCallNote,
}: {
  label: string;
  total: number;
  startMs: number | null;
  tick: number;
  firstCallNote: string | null;
}) {
  void tick; // tick prop triggers re-render every 300ms
  const start = startMs ?? Date.now();
  const elapsed = Math.max(0, (Date.now() - start) / 1000);
  // 휴리스틱: 첫 호출 모델 로드 ~6s + 이미지당 ~3.5s
  const perImage = 3.5;
  const warmup = 6.0;
  const totalEst = Math.max(warmup + (total || 1) * perImage, 8);
  // 점근식 진행 — 응답 도착 전까지 95% 에서 멈춤
  const raw = elapsed / totalEst;
  const progress = Math.min(0.96, 1 - Math.exp(-raw * 0.9));
  const etaSec = Math.max(0, totalEst - elapsed);
  const overTime = elapsed > totalEst * 1.05;

  const fmt = (s: number) => {
    if (!Number.isFinite(s)) return "—";
    if (s < 60) return `${s.toFixed(0)}초`;
    const m = Math.floor(s / 60);
    const sec = Math.round(s % 60);
    return `${m}분 ${sec.toString().padStart(2,"0")}초`;
  };
  const doneEst = Math.min(total, Math.round(progress * (total || 1)));

  return (
    <div className="app-card p-5 fade-in">
      <div className="flex items-baseline justify-between gap-3 flex-wrap">
        <div>
          <div className="text-[13px] font-semibold text-[#191919]">{label} 진행 중</div>
          <div className="text-[11px] text-[#9ca3af] mt-0.5">
            {total > 0
              ? <>약 <b className="text-[#191919]">{doneEst}</b> / {total}장 처리 추정</>
              : <>서버 응답 대기 중</>
            }
          </div>
        </div>
        <div className="text-right">
          <div className="text-[22px] font-bold text-[#191919] leading-none tabular-nums">
            {(progress * 100).toFixed(0)}<span className="text-[12px] text-[#9ca3af] font-medium ml-0.5">%</span>
          </div>
          <div className="text-[10px] text-[#9ca3af] mt-0.5">예상 진행률</div>
        </div>
      </div>

      <div className="mt-3 h-1.5 bg-[#f1f1ef] rounded-full overflow-hidden relative">
        <div
          className="h-full bg-[#191919] transition-all duration-200"
          style={{ width: `${progress * 100}%` }}
        />
      </div>

      <div className="mt-3 grid grid-cols-3 gap-2">
        <div>
          <div className="text-[9.5px] uppercase tracking-wider text-[#9ca3af] font-semibold">경과</div>
          <div className="text-[14px] font-semibold text-[#191919] font-mono mt-0.5 tabular-nums">{fmt(elapsed)}</div>
        </div>
        <div>
          <div className="text-[9.5px] uppercase tracking-wider text-[#9ca3af] font-semibold">남은 시간 (예상)</div>
          <div className={"text-[14px] font-semibold font-mono mt-0.5 tabular-nums " + (overTime ? "text-[#b7791f]" : "text-[#191919]")}>
            {overTime ? "곧 완료..." : `약 ${fmt(etaSec)}`}
          </div>
        </div>
        <div>
          <div className="text-[9.5px] uppercase tracking-wider text-[#9ca3af] font-semibold">전체 예상</div>
          <div className="text-[14px] font-semibold text-[#191919] font-mono mt-0.5 tabular-nums">{fmt(totalEst)}</div>
        </div>
      </div>

      {firstCallNote && (
        <div className="text-[10.5px] text-[#9ca3af] mt-3 leading-snug">
          {firstCallNote}
        </div>
      )}
      <div className="text-[10px] text-[#cbd0d6] mt-2">
        ※ 서버 단일 응답 방식 — 진행률·남은 시간은 이미지 수와 경과시간 기반 추정값입니다.
      </div>
    </div>
  );
}

function HybridDownloadBar({
  result, currentItem, currentDiagView, currentDiagLabel, currentSrc, originalsMap,
  labelScale,
}: {
  result: HybridResult;
  currentItem: HybridItem;
  currentDiagView: string;
  currentDiagLabel: string;
  currentSrc: string;
  originalsMap: Record<string, string>;
  labelScale: number;
}) {
  const [busy, setBusy] = useState<null | "zip" | "ppt" | "pdf" | "particles">(null);

  // ── PDF 옵션 모달 ──
  type PdfViewKey =
    | "original" | "overlay" | "instances" | "numbered" | "mask" | "prob"
    | "high" | "close" | "boundary" | "separator" | "color" | "hist";
  const PDF_VIEW_DEFS: { key: PdfViewKey; label: string; desc?: string }[] = [
    { key: "original",  label: "원본",            desc: "입력 SEM 이미지" },
    { key: "overlay",   label: "오버레이",         desc: "원본 + instance 채색" },
    { key: "instances", label: "Instance",       desc: "입자별 색" },
    { key: "numbered",  label: "번호",            desc: "p1, p2 ... 라벨 오버레이" },
    { key: "mask",      label: "Mask",           desc: "이진 마스크" },
    { key: "prob",      label: "U-Net prob",     desc: "확률맵" },
    { key: "high",      label: "M_75 (anchor)",  desc: "확실한 입자" },
    { key: "close",     label: "M_close",        desc: "거리 + Otsu (좁은 틈)" },
    { key: "boundary",  label: "YOLO 경계",       desc: "분리선 후보" },
    { key: "separator", label: "최종 분리선",      desc: "M_close ∩ B" },
    { key: "color",     label: "색 진단",          desc: "M_close / B / 겹침" },
    { key: "hist",      label: "d₊ 히스토그램",   desc: "Otsu 임계 시각화" },
  ];
  const hasGTGlobal = result.n_with_gt > 0;
  const hasHistAny = result.results.some(it => !!it.diag_hist_png);
  const hasParticlesAny = result.results.some(it => !!it.particles && it.particles.length > 0);
  const [pdfModalOpen, setPdfModalOpen] = useState(false);
  const [pdfOpts, setPdfOpts] = useState<Record<PdfViewKey, boolean> & { metrics: boolean }>({
    original: true,
    overlay: true,
    instances: true,
    numbered: false,
    mask: false,
    prob: false,
    high: false,
    close: false,
    boundary: false,
    separator: false,
    color: false,
    hist: false,
    metrics: true,
  });

  useEffect(() => {
    if (!pdfModalOpen) return;
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setPdfModalOpen(false); };
    window.addEventListener("keydown", onKey);
    return () => {
      document.body.style.overflow = prev;
      window.removeEventListener("keydown", onKey);
    };
  }, [pdfModalOpen]);

  // 원본 (blob URL) 을 data URL 로 변환 — PDF/PPT 임베드용
  async function blobUrlToDataUrl(url: string): Promise<string | null> {
    if (!url) return null;
    if (url.startsWith("data:")) return url;
    try {
      const r = await fetch(url);
      const b = await r.blob();
      return await new Promise<string>((res, rej) => {
        const fr = new FileReader();
        fr.onload = () => res(fr.result as string);
        fr.onerror = rej;
        fr.readAsDataURL(b);
      });
    } catch {
      return null;
    }
  }

  async function getImageSize(dataUrl: string): Promise<{ w: number; h: number } | null> {
    return new Promise((resolve) => {
      const im = new Image();
      im.onload = () => resolve({ w: im.naturalWidth, h: im.naturalHeight });
      im.onerror = () => resolve(null);
      im.src = dataUrl;
    });
  }
  const safeName = (s: string) => s.replace(/[^\w가-힣.-]+/g, "_").slice(0, 80) || "image";

  // 현재 이미지 다운로드
  const currentImgName = `${safeName(currentItem.filename.replace(/\.[^.]+$/, ""))}_${currentDiagView}.png`;

  async function downloadAllImagesZip() {
    setBusy("zip");
    try {
      const JSZipMod = (await import("jszip")).default;
      const zip = new JSZipMod();
      // overlay + mask + instances 한 폴더씩
      for (let i = 0; i < result.results.length; i++) {
        const it = result.results[i];
        const stem = safeName(it.filename.replace(/\.[^.]+$/, ""));
        const idx = String(i + 1).padStart(3, "0");
        const pairs: [string, string | undefined][] = [
          ["overlay", it.overlay_png],
          ["instances", it.instances_png],
          ["mask", it.mask_png],
          ["unet_prob", it.diag_prob_png],
          ["separator", it.diag_separator_png],
        ];
        for (const [name, dataUrl] of pairs) {
          if (!dataUrl) continue;
          const base64 = dataUrl.split(",")[1];
          if (!base64) continue;
          zip.file(`${idx}_${stem}/${name}.png`, base64, { base64: true });
        }
        if (it.gt_png) {
          const b64 = it.gt_png.split(",")[1];
          if (b64) zip.file(`${idx}_${stem}/gt.png`, b64, { base64: true });
        }
      }
      const blob = await zip.generateAsync({ type: "blob" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `hybrid_images_${result.n_images}imgs.zip`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    } catch (e) {
      alert(`ZIP 생성 실패: ${(e as Error).message}`);
    } finally {
      setBusy(null);
    }
  }

  async function downloadPPT() {
    setBusy("ppt");
    try {
      const PptxGenJS = (await import("pptxgenjs")).default;
      const pptx = new PptxGenJS();
      pptx.layout = "LAYOUT_WIDE"; // 13.33 x 7.5 in
      pptx.title = "Particle Lab — Hybrid 결과";
      pptx.author = "Particle Lab v2";

      // 표지
      const cover = pptx.addSlide();
      cover.background = { color: "FFFFFF" };
      cover.addText("Particle Lab — Hybrid 결과", {
        x: 0.6, y: 0.7, w: 12, h: 0.8,
        fontSize: 32, bold: true, color: "191919", fontFace: "Arial",
      });
      cover.addText(`이미지 ${result.n_images}장 · 총 입자 ${result.total_instances.toLocaleString()}개 · GT 매칭 ${result.n_with_gt}/${result.n_images}장`, {
        x: 0.6, y: 1.55, w: 12, h: 0.4,
        fontSize: 14, color: "6B7280", fontFace: "Arial",
      });
      cover.addText(result.params, {
        x: 0.6, y: 2.1, w: 12, h: 1.0,
        fontSize: 10, color: "9CA3AF", fontFace: "Courier New",
      });
      cover.addText(new Date().toLocaleString("ko-KR"), {
        x: 0.6, y: 6.8, w: 12, h: 0.3,
        fontSize: 10, color: "9CA3AF", fontFace: "Arial",
      });

      // 각 이미지 슬라이드
      for (let i = 0; i < result.results.length; i++) {
        const it = result.results[i];
        const slide = pptx.addSlide();
        slide.background = { color: "FFFFFF" };
        slide.addText(it.filename, {
          x: 0.4, y: 0.25, w: 12.5, h: 0.45,
          fontSize: 16, bold: true, color: "191919", fontFace: "Arial",
        });
        slide.addText(
          `입자 ${it.n_instances}개 · YOLO ${it.n_yolo} · U-Net anchor ${it.K_unet_anchor} · τ=${it.tau.toFixed(2)}` +
          (it.gt_filename ? `   ·   GT: ${it.gt_filename}` : ""),
          {
            x: 0.4, y: 0.7, w: 12.5, h: 0.3,
            fontSize: 10, color: "6B7280", fontFace: "Arial",
          }
        );
        // 좌: overlay, 우: instances
        if (it.overlay_png) {
          slide.addImage({ data: it.overlay_png, x: 0.4, y: 1.15, w: 6.2, h: 5.5, sizing: { type: "contain", w: 6.2, h: 5.5 } });
          slide.addText("Overlay", { x: 0.4, y: 6.65, w: 6.2, h: 0.3, fontSize: 10, color: "9CA3AF", align: "center", fontFace: "Arial" });
        }
        if (it.instances_png) {
          slide.addImage({ data: it.instances_png, x: 6.75, y: 1.15, w: 6.2, h: 5.5, sizing: { type: "contain", w: 6.2, h: 5.5 } });
          slide.addText("Instances", { x: 6.75, y: 6.65, w: 6.2, h: 0.3, fontSize: 10, color: "9CA3AF", align: "center", fontFace: "Arial" });
        }

        if (it.metrics_hybrid || it.metrics) {
          const m = (it.metrics_hybrid ?? it.metrics)!;
          slide.addText(
            `Precision ${m.Precision.toFixed(3)}  ·  Recall ${m.Recall.toFixed(3)}  ·  F1 ${m.F1.toFixed(3)}  ·  IoU ${m.IoU.toFixed(3)}  ·  CountErr ${(m.ParticleCountError*100).toFixed(1)}%`,
            { x: 0.4, y: 7.05, w: 12.5, h: 0.35, fontSize: 11, color: "191919", align: "center", fontFace: "Arial" }
          );
        }
      }

      await pptx.writeFile({ fileName: `hybrid_results_${result.n_images}imgs.pptx` });
    } catch (e) {
      alert(`PPT 생성 실패: ${(e as Error).message}`);
    } finally {
      setBusy(null);
    }
  }

  async function downloadPDF(optsParam?: typeof pdfOpts) {
    const opts = optsParam ?? pdfOpts;
    // 선택된 이미지 뷰 — VIEW_DEFS 순서 보존
    const selectedViews = PDF_VIEW_DEFS.filter(v => opts[v.key]);
    if (selectedViews.length === 0 && !opts.metrics) {
      alert("최소 1개 항목 이상 선택해주세요.");
      return;
    }
    setBusy("pdf");
    try {
      const { jsPDF } = await import("jspdf");

      // ── 페이지 좌표계 (mm, A4 landscape) ──
      const PAGE_W = 297, PAGE_H = 210;
      const doc = new jsPDF({ orientation: "landscape", unit: "mm", format: "a4" });

      // ── 색상 팔레트 (Toss 풍 무채색) ──
      const C_INK    = [25, 25, 25] as const;
      const C_TEXT   = [55, 55, 60] as const;
      const C_MUTE   = [120, 122, 130] as const;
      const C_FAINT  = [170, 174, 184] as const;
      const C_LINE   = [232, 232, 228] as const;
      const C_LINE_SOFT = [241, 241, 239] as const;
      const C_BG     = [250, 250, 247] as const;
      const C_GOOD   = [0, 159, 80] as const;
      const C_BAD    = [232, 72, 60] as const;

      // ── 폰트 시스템 사전 로드 (GmarketSans 캔버스 렌더링 보장) ──
      if (typeof document !== "undefined" && (document as any).fonts) {
        try {
          await Promise.all([
            (document as any).fonts.load("400 24px GmarketSans"),
            (document as any).fonts.load("500 14px GmarketSans"),
            (document as any).fonts.load("700 14px GmarketSans"),
            (document as any).fonts.ready,
          ]);
        } catch { /* ignore */ }
      }

      // ── 한글 친화 텍스트 렌더러 (Canvas → PNG) ──
      // jsPDF 의 doc.text 는 라틴 글리프만 지원하므로,
      // 한글이 섞인 모든 텍스트는 캔버스로 그려서 이미지로 PDF 에 박는다.
      // 캔버스 해상도 = ptSize × SCALE (px). mm = pt × 25.4/72.
      const SCALE = 4; // 4x 해상도 (인쇄 품질)
      const wrapByChar = (ctx: CanvasRenderingContext2D, text: string, maxW: number): string[] => {
        if (!text) return [""];
        const out: string[] = [];
        // 줄바꿈 문자 우선
        for (const part of text.split("\n")) {
          if (!part) { out.push(""); continue; }
          let line = "";
          for (const ch of part) {
            const t = line + ch;
            if (ctx.measureText(t).width > maxW && line) {
              out.push(line);
              line = ch;
            } else {
              line = t;
            }
          }
          if (line) out.push(line);
        }
        return out;
      };
      const renderText = async (
        text: string,
        ptSize: number,
        opts: {
          bold?: boolean;
          color?: readonly [number, number, number];
          maxWidthMm?: number;
          align?: "left" | "center" | "right";
          letterSpacing?: number;
        } = {},
      ): Promise<{ dataUrl: string; widthMm: number; heightMm: number; widthPx: number; heightPx: number }> => {
        const { bold = false, color = C_INK, maxWidthMm = 280, letterSpacing = 0 } = opts;
        const fontSizePx = ptSize * SCALE;
        const maxWidthPx = (maxWidthMm / 25.4 * 72) * SCALE;
        const weight = bold ? "700" : "500";
        const fontStr = `${weight} ${fontSizePx}px "GmarketSans","Inter","Apple SD Gothic Neo","Pretendard",sans-serif`;
        // 가측정
        const m = document.createElement("canvas");
        const mc = m.getContext("2d")!;
        mc.font = fontStr;
        // letter-spacing 시뮬레이션 — Canvas 는 직접 지원이 약하므로 글자 단위로 그림
        const measureLine = (s: string) => {
          if (letterSpacing === 0) return mc.measureText(s).width;
          let w = 0;
          for (const ch of s) w += mc.measureText(ch).width + letterSpacing * SCALE;
          return Math.max(0, w - letterSpacing * SCALE);
        };
        const lines = wrapByChar(mc, text, maxWidthPx);
        const lineHeight = fontSizePx * 1.34;
        const widthPx = Math.ceil(Math.max(1, ...lines.map(measureLine))) + 4;
        const heightPx = Math.ceil(lineHeight * lines.length + fontSizePx * 0.35);
        const canvas = document.createElement("canvas");
        canvas.width = widthPx;
        canvas.height = heightPx;
        const ctx = canvas.getContext("2d")!;
        ctx.font = fontStr;
        ctx.textBaseline = "alphabetic";
        ctx.fillStyle = `rgb(${color[0]},${color[1]},${color[2]})`;
        // 안티앨리어싱 친화
        (ctx as any).textRendering = "geometricPrecision";
        lines.forEach((l, i) => {
          const y = i * lineHeight + fontSizePx;
          if (letterSpacing === 0) {
            ctx.fillText(l, 2, y);
          } else {
            let x = 2;
            for (const ch of l) {
              ctx.fillText(ch, x, y);
              x += ctx.measureText(ch).width + letterSpacing * SCALE;
            }
          }
        });
        return {
          dataUrl: canvas.toDataURL("image/png"),
          widthMm: (widthPx / SCALE) * 25.4 / 72,
          heightMm: (heightPx / SCALE) * 25.4 / 72,
          widthPx, heightPx,
        };
      };
      const drawText = async (text: string, x: number, y: number, ptSize: number, opts: {
        bold?: boolean;
        color?: readonly [number, number, number];
        maxWidthMm?: number;
        align?: "left" | "center" | "right";
        letterSpacing?: number;
      } = {}) => {
        if (!text) return { widthMm: 0, heightMm: 0 };
        const img = await renderText(text, ptSize, opts);
        const align = opts.align ?? "left";
        let dx = x;
        if (align === "center") dx = x - img.widthMm / 2;
        else if (align === "right") dx = x - img.widthMm;
        doc.addImage(img.dataUrl, "PNG", dx, y, img.widthMm, img.heightMm, undefined, "FAST");
        return { widthMm: img.widthMm, heightMm: img.heightMm };
      };

      const fillRect = (x: number, y: number, w: number, h: number, c: readonly [number, number, number]) => {
        doc.setFillColor(c[0], c[1], c[2]);
        doc.rect(x, y, w, h, "F");
      };
      const strokeRect = (x: number, y: number, w: number, h: number, c: readonly [number, number, number], lw = 0.15) => {
        doc.setDrawColor(c[0], c[1], c[2]);
        doc.setLineWidth(lw);
        doc.rect(x, y, w, h, "S");
      };
      const hLine = (x1: number, y: number, x2: number, c: readonly [number, number, number] = C_LINE, lw = 0.15) => {
        doc.setDrawColor(c[0], c[1], c[2]);
        doc.setLineWidth(lw);
        doc.line(x1, y, x2, y);
      };

      // ─────────────────── 표지 ───────────────────
      // 상단 좌측 brand block
      const M = 18;
      // 로고 마크
      fillRect(M, M, 9, 9, C_INK);
      await drawText("P", M + 4.5, M + 1.6, 11, { bold: true, color: [255, 255, 255], align: "center" });
      // 브랜드 이름 + 작은 라벨
      await drawText("Particle Lab", M + 12, M - 0.5, 9.5, { bold: true, color: C_INK, letterSpacing: 0.2 });
      await drawText("SEM Particle Segmentation Report", M + 12, M + 4.8, 7, { color: C_MUTE });

      // 우측 상단: 메타 라벨
      await drawText("REPORT", PAGE_W - M, M + 0.5, 7, { color: C_FAINT, align: "right", letterSpacing: 1.2 });
      await drawText(new Date().toLocaleDateString("ko-KR"), PAGE_W - M, M + 4.6, 8, { color: C_MUTE, align: "right" });

      // 본문 큰 타이틀
      const titleY = 56;
      await drawText("Hybrid 분석 리포트", M, titleY, 32, { bold: true, color: C_INK, letterSpacing: -0.6 });
      await drawText("U-Net 확률맵 × YOLO 인스턴스 경계 융합", M, titleY + 16, 11, { color: C_MUTE });

      // 요약 통계 — 3 큰 숫자
      const summaryY = 100;
      hLine(M, summaryY - 6, PAGE_W - M, C_LINE);
      const colW = (PAGE_W - 2 * M) / 3;
      const stats: { label: string; value: string; sub?: string }[] = [
        { label: "이미지", value: String(result.n_images), sub: "장 분석" },
        { label: "총 입자", value: result.total_instances.toLocaleString(), sub: "개 검출" },
        { label: "GT 매칭", value: `${result.n_with_gt} / ${result.n_images}`, sub: result.n_with_gt > 0 ? "지표 산출됨" : "지표 없음" },
      ];
      for (let s = 0; s < stats.length; s++) {
        const cx = M + s * colW;
        await drawText(stats[s].label, cx, summaryY + 2, 8, { color: C_FAINT, letterSpacing: 1.0 });
        await drawText(stats[s].value, cx, summaryY + 7, 28, { bold: true, color: C_INK, letterSpacing: -0.4 });
        if (stats[s].sub) await drawText(stats[s].sub!, cx, summaryY + 24, 8.5, { color: C_MUTE });
      }
      hLine(M, summaryY + 36, PAGE_W - M, C_LINE);

      // 실행 파라미터 블록
      const paramY = summaryY + 46;
      await drawText("실행 파라미터", M, paramY, 9, { bold: true, color: C_INK, letterSpacing: 0.6 });
      // params chips — key=value 들을 칩으로 나눠 배치
      const paramChips = result.params.split(/\s+·\s+|\s+\/\s+|\s+,\s+/).map(s => s.trim()).filter(Boolean);
      let cx2 = M, cy2 = paramY + 6;
      const CHIP_GAP = 1.5;
      const CHIP_PAD_X = 2.4;
      const CHIP_PAD_Y = 1.4;
      const MAX_X = PAGE_W - M;
      for (const chip of paramChips) {
        const img = await renderText(chip, 7.5, { color: C_TEXT });
        const w = img.widthMm + CHIP_PAD_X * 2;
        const h = img.heightMm + CHIP_PAD_Y * 2;
        if (cx2 + w > MAX_X) { cx2 = M; cy2 += h + CHIP_GAP; }
        if (cy2 + h > PAGE_H - M - 8) break;
        fillRect(cx2, cy2, w, h, C_BG);
        strokeRect(cx2, cy2, w, h, C_LINE_SOFT, 0.1);
        doc.addImage(img.dataUrl, "PNG", cx2 + CHIP_PAD_X, cy2 + CHIP_PAD_Y, img.widthMm, img.heightMm);
        cx2 += w + CHIP_GAP;
      }

      // 푸터
      hLine(M, PAGE_H - 12, PAGE_W - M, C_LINE_SOFT, 0.1);
      await drawText("Particle Lab v2", M, PAGE_H - 9, 7.5, { color: C_FAINT, letterSpacing: 0.8 });
      await drawText(new Date().toLocaleString("ko-KR"), PAGE_W - M, PAGE_H - 9, 7.5, { color: C_FAINT, align: "right" });

      // ─────────────────── 이미지별 페이지 ───────────────────
      for (let i = 0; i < result.results.length; i++) {
        const it = result.results[i];
        doc.addPage();

        const pM = 14;
        const pHeadY = pM;
        // 페이지 헤더 — 페이지 번호 칩 + 파일명
        const pageChip = `${String(i + 1).padStart(2, "0")} / ${String(result.results.length).padStart(2, "0")}`;
        const pageChipImg = await renderText(pageChip, 7.5, { color: C_MUTE, letterSpacing: 1.2 });
        fillRect(pM, pHeadY, pageChipImg.widthMm + 4, pageChipImg.heightMm + 2.8, C_BG);
        strokeRect(pM, pHeadY, pageChipImg.widthMm + 4, pageChipImg.heightMm + 2.8, C_LINE_SOFT, 0.1);
        doc.addImage(pageChipImg.dataUrl, "PNG", pM + 2, pHeadY + 1.4, pageChipImg.widthMm, pageChipImg.heightMm);

        await drawText(it.filename, pM + pageChipImg.widthMm + 8, pHeadY - 0.5, 12, {
          bold: true, color: C_INK, maxWidthMm: PAGE_W - 2 * pM - 60,
        });

        // 상단 우측 — 요약 한 줄
        const headerSub = `입자 ${it.n_instances}   ·   YOLO ${it.n_yolo}   ·   Anchor ${it.K_unet_anchor}   ·   τ ${it.tau.toFixed(2)}`;
        await drawText(headerSub, PAGE_W - pM, pHeadY + 1, 8.5, { color: C_MUTE, align: "right" });
        if (it.gt_filename) {
          await drawText(`GT: ${it.gt_filename}`, PAGE_W - pM, pHeadY + 6, 7.5, { color: C_FAINT, align: "right" });
        }

        hLine(pM, pHeadY + 13, PAGE_W - pM, C_LINE);

        // 본문 영역 — 사용자가 선택한 N개 뷰에 맞춘 적응형 그리드
        const m = it.metrics_hybrid ?? it.metrics;
        const showMetrics = opts.metrics && !!m;
        const FOOT_H = showMetrics ? 38 : 16;
        const bodyY = pHeadY + 18;
        const bodyH = PAGE_H - bodyY - FOOT_H;

        // 선택된 뷰들의 src 매핑
        const origData = await blobUrlToDataUrl(originalsMap[it.filename] ?? "");
        // 번호 모드 — 현재 보기모드의 labelScale 로 클라이언트 렌더링 (viewer 와 동일한 글자 크기)
        const numberedClient = opts.numbered
          ? await renderNumberedImage(it.instances_png ?? "", it.particles ?? [], labelScale)
          : null;
        const SRC: Record<PdfViewKey, string | null> = {
          original:  origData,
          overlay:   it.overlay_png ?? null,
          instances: it.instances_png ?? null,
          numbered:  numberedClient,
          mask:      it.mask_png ?? null,
          prob:      it.diag_prob_png ?? null,
          high:      it.diag_high_png ?? null,
          close:     it.diag_close_png ?? null,
          boundary:  it.diag_boundary_png ?? null,
          separator: it.diag_separator_png ?? null,
          color:     it.diag_overlap_png ?? null,
          hist:      it.diag_hist_png ?? null,
        };
        const cells = selectedViews.map(v => ({
          label: v.label,
          src: SRC[v.key],
        }));

        // 적응형 그리드 — n 개 뷰 → cols × rows
        const gridDims = (n: number): [number, number] => {
          if (n <= 1) return [1, 1];
          if (n === 2) return [2, 1];
          if (n === 3) return [3, 1];
          if (n === 4) return [2, 2];
          if (n <= 6) return [3, 2];
          if (n <= 9) return [3, 3];
          return [4, 3];
        };
        const N = cells.length;
        const [cols, rows] = N === 0 ? [0, 0] : gridDims(N);
        const GAP = N <= 3 ? 4 : 3;

        if (N > 0) {
          const cellW = (PAGE_W - 2 * pM - (cols - 1) * GAP) / cols;
          const cellH = (bodyH - (rows - 1) * GAP) / rows;

          for (let c = 0; c < cells.length; c++) {
            const cell = cells[c];
            const colIdx = c % cols;
            const rowIdx = Math.floor(c / cols);
            const x = pM + colIdx * (cellW + GAP);
            const y = bodyY + rowIdx * (cellH + GAP);

            // 카드 보더
            strokeRect(x, y, cellW, cellH, C_LINE_SOFT, 0.15);

            // 캡션
            const CAP_H = N >= 7 ? 6 : 8;
            fillRect(x, y, cellW, CAP_H, C_BG);
            hLine(x, y + CAP_H, x + cellW, C_LINE_SOFT, 0.1);
            const capFont = N >= 7 ? 6.5 : 7.5;
            await drawText(cell.label, x + 2.2, y + (N >= 7 ? 1.3 : 1.7), capFont, {
              color: C_INK, bold: true, letterSpacing: 0.5,
            });

            const imgArea = {
              x: x + 1, y: y + CAP_H + 1,
              w: cellW - 2, h: cellH - CAP_H - 2,
            };

            if (!cell.src) {
              await drawText(
                cell.label === "원본" ? "원본 미연결" : "없음",
                imgArea.x + imgArea.w / 2, imgArea.y + imgArea.h / 2 - 2,
                Math.max(6.5, 8 - (N - 3) * 0.3),
                { color: C_FAINT, align: "center" },
              );
              continue;
            }

            // 비율 보존 fit
            const sz = await getImageSize(cell.src);
            let drawW = imgArea.w, drawH = imgArea.h;
            if (sz) {
              const ar = sz.w / sz.h;
              const fitH = imgArea.w / ar;
              if (fitH <= imgArea.h) { drawW = imgArea.w; drawH = fitH; }
              else { drawH = imgArea.h; drawW = imgArea.h * ar; }
            }
            const ox = imgArea.x + (imgArea.w - drawW) / 2;
            const oy = imgArea.y + (imgArea.h - drawH) / 2;
            try {
              const fmt = /^data:image\/jpe?g/.test(cell.src) ? "JPEG" : "PNG";
              doc.addImage(cell.src, fmt, ox, oy, drawW, drawH, undefined, "FAST");
            } catch {
              strokeRect(imgArea.x, imgArea.y, imgArea.w, imgArea.h, C_LINE);
            }
          }
        }

        // ── 평가 지표 (옵션 + GT 있을 때) — 화면 동일 7 셀 ──
        if (showMetrics && m) {
          const metricsY = PAGE_H - FOOT_H + 2;
          hLine(pM, metricsY - 2, PAGE_W - pM, C_LINE);
          const metrics: { label: string; value: string; tier: "good" | "bad" | "ink" }[] = (() => {
            const tier = (v: number, better: "high" | "low"): "good" | "bad" | "ink" => {
              if (!isFinite(v)) return "ink";
              if (better === "high") {
                if (v >= 0.85) return "good";
                if (v < 0.40) return "bad";
                return "ink";
              } else {
                if (v <= 0.05) return "good";
                if (v > 0.30) return "bad";
                return "ink";
              }
            };
            return [
              { label: "Precision", value: m.Precision.toFixed(4), tier: tier(m.Precision, "high") },
              { label: "Recall",    value: m.Recall.toFixed(4),    tier: tier(m.Recall, "high") },
              { label: "Accuracy",  value: m.Accuracy.toFixed(4),  tier: tier(m.Accuracy, "high") },
              { label: "IoU",       value: m.IoU.toFixed(4),       tier: tier(m.IoU, "high") },
              { label: "F1",        value: m.F1.toFixed(4),        tier: tier(m.F1, "high") },
              { label: "mAP@0.5",   value: m["mAP@0.5"].toFixed(4),tier: tier(m["mAP@0.5"], "high") },
              { label: "CountErr",  value: `${(m.ParticleCountError * 100).toFixed(2)}%`, tier: tier(m.ParticleCountError, "low") },
            ];
          })();
          const mCellW = (PAGE_W - 2 * pM) / metrics.length;
          for (let k = 0; k < metrics.length; k++) {
            const mx = pM + k * mCellW;
            const valColor =
              metrics[k].tier === "good" ? C_GOOD :
              metrics[k].tier === "bad"  ? C_BAD  : C_INK;
            await drawText(metrics[k].label, mx + 1.5, metricsY + 2, 7, { color: C_FAINT, letterSpacing: 0.8 });
            await drawText(metrics[k].value, mx + 1.5, metricsY + 7, 14, { bold: true, color: valColor, letterSpacing: -0.2 });
          }
          await drawText(
            `n_gt ${m.n_gt_inst}   ·   n_pred ${m.n_pred_inst}`,
            pM, metricsY + 26, 7.5, { color: C_FAINT, letterSpacing: 0.4 },
          );
        } else if (opts.metrics && !m) {
          // 사용자가 평가지표를 켰지만 이 이미지엔 GT 없음
          await drawText(
            `${it.filename} — 정답 마스크 미매칭, 지표 생략됨`,
            pM, PAGE_H - 12, 7.5, { color: C_FAINT },
          );
        }

        // 페이지 푸터
        hLine(pM, PAGE_H - 8, PAGE_W - pM, C_LINE_SOFT, 0.1);
        await drawText("Particle Lab v2  ·  Hybrid Report", pM, PAGE_H - 5.5, 7, { color: C_FAINT, letterSpacing: 0.5 });
        await drawText(`p. ${i + 2} / ${result.results.length + 1}`, PAGE_W - pM, PAGE_H - 5.5, 7, { color: C_FAINT, align: "right" });
      }

      const pdfName = `hybrid_report_${result.n_images}imgs.pdf`;
      // 데스크탑 런처(pywebview)에서는 doc.save() 의 anchor/iframe trigger 가
      // macOS WebKit 에서 메인 창을 PDF 로 navigate 시키는 문제가 있어,
      // launcher 가 노출하는 __plabSaveBlob 으로 직접 bridge 한다.
      const plabSave = (window as any).__plabSaveBlob;
      if (typeof plabSave === "function") {
        const blob = doc.output("blob");
        await plabSave(pdfName, blob);
      } else {
        doc.save(pdfName);
      }
    } catch (e) {
      alert(`PDF 생성 실패: ${(e as Error).message}`);
    } finally {
      setBusy(null);
    }
  }

  // ── 입자별 크기 엑셀 (.xlsx) ──
  // 각 이미지마다 시트 1개 + "Summary" 시트 1개.
  // 시트 컬럼: id (1..N), label (p1, p2, ...), area_px, centroid_x, centroid_y, area_um2(추후)
  async function downloadParticlesXlsx() {
    setBusy("particles");
    try {
      const XLSX = await import("xlsx");
      const wb = XLSX.utils.book_new();

      // 시트명 안전 처리 — Excel 제한 (31자, /\?*[] 금지)
      const safeSheetName = (s: string, used: Set<string>) => {
        let base = s.replace(/\.[^.]+$/, "").replace(/[\/\\?*[\]:]/g, "_").slice(0, 28);
        if (!base) base = "sheet";
        let name = base, k = 2;
        while (used.has(name)) { name = `${base}_${k}`.slice(0, 31); k++; }
        used.add(name);
        return name;
      };
      const used = new Set<string>();

      // Summary 시트 — 이미지별 총 입자 수 + 평균/중위 크기
      const summaryRows: (string | number)[][] = [
        ["filename", "n_particles", "total_area_px", "mean_area_px", "median_area_px", "min_area_px", "max_area_px", "image_w", "image_h"],
      ];

      for (const it of result.results) {
        const parts = it.particles ?? [];
        const sheetName = safeSheetName(it.filename || "image", used);
        // 정렬: 면적 큰 순 → 그대로 (backend 가 이미 정렬해서 보냄)
        const rows: (string | number)[][] = [
          ["id", "label", "area_px", "centroid_x", "centroid_y"],
          ...parts.map(p => [p.id, p.label, p.area_px, p.cx, p.cy]),
        ];
        const ws = XLSX.utils.aoa_to_sheet(rows);
        // 컬럼 너비
        (ws as any)["!cols"] = [
          { wch: 6 }, { wch: 8 }, { wch: 12 }, { wch: 12 }, { wch: 12 },
        ];
        XLSX.utils.book_append_sheet(wb, ws, sheetName);

        // 요약 통계
        const areas = parts.map(p => p.area_px);
        const total = areas.reduce((a, b) => a + b, 0);
        const mean = areas.length ? total / areas.length : 0;
        const sorted = [...areas].sort((a, b) => a - b);
        const median = sorted.length
          ? (sorted.length % 2
            ? sorted[(sorted.length - 1) / 2]
            : (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2)
          : 0;
        summaryRows.push([
          it.filename,
          parts.length,
          total,
          Number(mean.toFixed(2)),
          Number(median.toFixed(2)),
          areas.length ? sorted[0] : 0,
          areas.length ? sorted[sorted.length - 1] : 0,
          it.shape[1], it.shape[0],
        ]);
      }

      const wsSummary = XLSX.utils.aoa_to_sheet(summaryRows);
      (wsSummary as any)["!cols"] = [
        { wch: 32 }, { wch: 12 }, { wch: 14 }, { wch: 13 }, { wch: 14 },
        { wch: 12 }, { wch: 12 }, { wch: 10 }, { wch: 10 },
      ];
      // Summary 를 첫 시트로
      XLSX.utils.book_append_sheet(wb, wsSummary, "Summary");
      // wb.SheetNames 의 끝에 추가되었으니 첫번째로 이동
      const idx = wb.SheetNames.indexOf("Summary");
      if (idx > 0) {
        wb.SheetNames.splice(idx, 1);
        wb.SheetNames.unshift("Summary");
      }

      XLSX.writeFile(wb, `hybrid_particles_${result.n_images}imgs.xlsx`);
    } catch (e) {
      alert(`엑셀 생성 실패: ${(e as Error).message}`);
    } finally {
      setBusy(null);
    }
  }

  const Row = ({
    icon, label, sub, onClick, href, downloadName, disabled, kind,
  }: {
    icon: React.ReactNode;
    label: string;
    sub?: string;
    onClick?: () => void;
    href?: string;
    downloadName?: string;
    disabled?: boolean;
    kind?: "primary" | "default";
  }) => {
    const cls =
      "w-full flex items-center gap-2.5 px-3 py-2.5 rounded-md border text-[12px] font-medium text-left transition " +
      (kind === "primary"
        ? "bg-[#191919] text-white border-[#191919] hover:bg-[#2c2c2c] "
        : "bg-white text-[#191919] border-[#f1f1ef] hover:border-[#191919] hover:bg-[#fafafa] ") +
      (disabled ? "opacity-50 cursor-not-allowed " : "");
    const inner = (
      <>
        <span className={"w-7 h-7 rounded-md flex items-center justify-center flex-shrink-0 " + (kind === "primary" ? "bg-white/10" : "bg-[#fafafa]")}>
          {icon}
        </span>
        <span className="flex-1 min-w-0">
          <span className="block truncate">{label}</span>
          {sub && <span className={"block text-[10px] mt-0.5 truncate font-normal " + (kind === "primary" ? "text-white/60" : "text-[#9ca3af]")}>{sub}</span>}
        </span>
        <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="opacity-50 flex-shrink-0">
          <polyline points="9 18 15 12 9 6"/>
        </svg>
      </>
    );
    if (href) {
      return (
        <a href={href} download={downloadName} className={cls} onClick={(e) => e.stopPropagation()}>{inner}</a>
      );
    }
    return (
      <button type="button" onClick={onClick} disabled={disabled} className={cls}>{inner}</button>
    );
  };

  return (
    <div className="app-card p-4">
      <div className="flex items-center justify-between mb-3">
        <div>
          <h2 className="text-[13px] font-semibold text-[#191919]">다운로드</h2>
          <p className="text-[10.5px] text-[#9ca3af] mt-0.5">결과를 파일로 저장</p>
        </div>
        <span className="text-[10px] text-[#9ca3af] font-mono">{result.n_images}장</span>
      </div>

      <div className="space-y-1.5">
        <Row
          icon={
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <rect x="3" y="3" width="18" height="18" rx="2"/>
              <circle cx="8.5" cy="8.5" r="1.5"/>
              <polyline points="21 15 16 10 5 21"/>
            </svg>
          }
          label="현재 이미지 저장"
          sub={currentDiagLabel + " · PNG"}
          href={currentSrc}
          downloadName={currentImgName}
          kind="primary"
        />
        <Row
          icon={
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
              <polyline points="14 2 14 8 20 8"/>
              <path d="M9 13h6"/>
              <path d="M9 17h4"/>
            </svg>
          }
          label={busy === "pdf" ? "PDF 생성 중..." : "PDF 리포트"}
          sub="포함할 항목 선택 후 생성"
          onClick={() => setPdfModalOpen(true)}
          disabled={busy !== null}
        />
        <Row
          icon={
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <rect x="3" y="3" width="18" height="18" rx="2"/>
              <path d="M8 8h6a2 2 0 0 1 0 4H8z"/>
              <path d="M8 12v5"/>
            </svg>
          }
          label={busy === "ppt" ? "PPT 생성 중..." : "PPT 슬라이드"}
          sub="이미지당 1장 슬라이드"
          onClick={downloadPPT}
          disabled={busy !== null}
        />
        <Row
          icon={
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 8v13a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8"/>
              <polyline points="3 8 12 13 21 8"/>
              <path d="M12 2v11"/>
            </svg>
          }
          label={busy === "zip" ? "ZIP 생성 중..." : "전체 이미지 ZIP"}
          sub={`${result.n_images}장 · overlay/instances/mask 폴더별`}
          onClick={downloadAllImagesZip}
          disabled={busy !== null}
        />
        {hasParticlesAny && (
          <Row
            icon={
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="7" cy="8" r="2.5"/>
                <circle cx="14" cy="15" r="2.5"/>
                <circle cx="17" cy="7" r="1.7"/>
                <circle cx="6" cy="17" r="1.7"/>
              </svg>
            }
            label={busy === "particles" ? "엑셀 생성 중..." : "입자별 픽셀 엑셀"}
            sub={`p1, p2... · 이미지당 시트 1개 + Summary`}
            onClick={downloadParticlesXlsx}
            disabled={busy !== null}
          />
        )}
        {result.excel_data_url && (
          <Row
            icon={
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                <polyline points="14 2 14 8 20 8"/>
                <line x1="16" y1="13" x2="8" y2="13"/>
                <line x1="16" y1="17" x2="8" y2="17"/>
                <polyline points="10 9 9 9 8 9"/>
              </svg>
            }
            label="지표 엑셀 (.xlsx)"
            sub={`평가지표 시트 4개${result.excel_size_bytes ? ` · ${(result.excel_size_bytes / 1024).toFixed(0)}KB` : ""}`}
            href={result.excel_data_url}
            downloadName={`hybrid_results_${result.n_images}imgs.xlsx`}
          />
        )}
      </div>

      {pdfModalOpen && typeof document !== "undefined" && createPortal(
        <div
          className="fixed inset-0 z-50 bg-black/40 backdrop-blur-[2px] flex items-center justify-center p-4 fade-in"
          onClick={() => setPdfModalOpen(false)}
        >
          <div
            className="bg-white rounded-xl shadow-lg w-full max-w-[480px] max-h-[90vh] flex flex-col"
            onClick={(e) => e.stopPropagation()}
          >
            {/* 헤더 */}
            <div className="px-5 pt-5 pb-3">
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <h3 className="text-[15px] font-semibold text-[#191919]">PDF 리포트 옵션</h3>
                  <p className="text-[11.5px] text-[#9ca3af] mt-0.5">
                    각 이미지마다 포함할 항목을 선택하세요
                  </p>
                </div>
                <button
                  type="button"
                  onClick={() => setPdfModalOpen(false)}
                  className="w-7 h-7 rounded-md hover:bg-[#f5f5f3] text-[#6b7280] hover:text-[#191919] flex items-center justify-center text-[18px] leading-none"
                  aria-label="닫기"
                >
                  ×
                </button>
              </div>
            </div>

            {/* 옵션 본문 */}
            <div className="px-5 pb-1 overflow-y-auto">
              {/* 일괄 선택 */}
              <div className="flex items-center gap-2 text-[11px] mb-3">
                <button
                  type="button"
                  className="px-2 py-1 rounded-md border border-[#e6e6e2] text-[#6b7280] hover:border-[#191919] hover:text-[#191919]"
                  onClick={() => {
                    const next: any = { ...pdfOpts };
                    PDF_VIEW_DEFS.forEach(v => {
                      if (v.key === "hist") next[v.key] = hasHistAny;
                      else if (v.key === "numbered") next[v.key] = hasParticlesAny;
                      else next[v.key] = true;
                    });
                    next.metrics = hasGTGlobal;
                    setPdfOpts(next);
                  }}
                >
                  모두 선택
                </button>
                <button
                  type="button"
                  className="px-2 py-1 rounded-md border border-[#e6e6e2] text-[#6b7280] hover:border-[#191919] hover:text-[#191919]"
                  onClick={() => {
                    const next: any = { ...pdfOpts };
                    PDF_VIEW_DEFS.forEach(v => { next[v.key] = false; });
                    next.metrics = false;
                    setPdfOpts(next);
                  }}
                >
                  모두 해제
                </button>
                <button
                  type="button"
                  className="px-2 py-1 rounded-md border border-[#e6e6e2] text-[#6b7280] hover:border-[#191919] hover:text-[#191919] ml-auto"
                  onClick={() => {
                    setPdfOpts({
                      original: true, overlay: true, instances: true,
                      numbered: false, mask: false, prob: false, high: false,
                      close: false, boundary: false, separator: false,
                      color: false, hist: false,
                      metrics: hasGTGlobal,
                    });
                  }}
                >
                  기본값
                </button>
              </div>

              {/* 이미지 항목 그리드 */}
              <div className="text-[10px] uppercase tracking-wider font-semibold text-[#9ca3af] mb-2">이미지 항목</div>
              <div className="grid grid-cols-2 gap-1.5 mb-4">
                {PDF_VIEW_DEFS.map(v => {
                  const disabled =
                    (v.key === "hist" && !hasHistAny) ||
                    (v.key === "numbered" && !hasParticlesAny);
                  const checked = !!pdfOpts[v.key];
                  return (
                    <label
                      key={v.key}
                      className={
                        "flex items-start gap-2 px-2.5 py-2 rounded-md border cursor-pointer transition " +
                        (disabled
                          ? "border-[#f1f1ef] bg-[#fafafa] opacity-50 cursor-not-allowed "
                          : checked
                            ? "border-[#191919] bg-[#fafafa] "
                            : "border-[#f1f1ef] bg-white hover:border-[#d3d3cb] ")
                      }
                    >
                      <input
                        type="checkbox"
                        checked={checked}
                        disabled={disabled}
                        onChange={(e) => setPdfOpts({ ...pdfOpts, [v.key]: e.target.checked })}
                        className="mod-check mt-0.5 flex-shrink-0"
                      />
                      <div className="min-w-0">
                        <div className="text-[12px] font-semibold text-[#191919] truncate">{v.label}</div>
                        {v.desc && <div className="text-[10px] text-[#9ca3af] truncate">{v.desc}</div>}
                      </div>
                    </label>
                  );
                })}
              </div>

              {/* 추가 */}
              <div className="text-[10px] uppercase tracking-wider font-semibold text-[#9ca3af] mb-2">추가</div>
              <label
                className={
                  "flex items-start gap-2 px-2.5 py-2 rounded-md border cursor-pointer transition mb-2 " +
                  (!hasGTGlobal
                    ? "border-[#f1f1ef] bg-[#fafafa] opacity-50 cursor-not-allowed "
                    : pdfOpts.metrics
                      ? "border-[#191919] bg-[#fafafa] "
                      : "border-[#f1f1ef] bg-white hover:border-[#d3d3cb] ")
                }
              >
                <input
                  type="checkbox"
                  checked={pdfOpts.metrics && hasGTGlobal}
                  disabled={!hasGTGlobal}
                  onChange={(e) => setPdfOpts({ ...pdfOpts, metrics: e.target.checked })}
                  className="mod-check mt-0.5 flex-shrink-0"
                />
                <div className="min-w-0">
                  <div className="text-[12px] font-semibold text-[#191919]">평가 지표</div>
                  <div className="text-[10px] text-[#9ca3af]">
                    {hasGTGlobal
                      ? `Precision · Recall · IoU · F1 · mAP · CountErr (GT 매칭 ${result.n_with_gt}/${result.n_images}장)`
                      : "정답 마스크 (GT) 가 매칭되지 않아 사용 불가"}
                  </div>
                </div>
              </label>
            </div>

            {/* 푸터 */}
            <div className="px-5 py-4 border-t border-[#f1f1ef] flex items-center justify-between gap-3">
              <div className="text-[11px] text-[#9ca3af] min-w-0 truncate">
                {(() => {
                  const n = PDF_VIEW_DEFS.filter(v => pdfOpts[v.key]).length;
                  const includeMetrics = pdfOpts.metrics && hasGTGlobal;
                  return `이미지 ${n}항목${includeMetrics ? " + 지표" : ""} · ${result.n_images}페이지`;
                })()}
              </div>
              <div className="flex gap-2 flex-shrink-0">
                <button
                  type="button"
                  onClick={() => setPdfModalOpen(false)}
                  className="px-3 py-1.5 rounded-md text-[12px] font-medium text-[#6b7280] hover:bg-[#f5f5f3]"
                >
                  취소
                </button>
                <button
                  type="button"
                  onClick={() => { setPdfModalOpen(false); downloadPDF(pdfOpts); }}
                  disabled={busy !== null}
                  className="px-3.5 py-1.5 rounded-md text-[12px] font-semibold bg-[#191919] text-white hover:bg-[#2c2c2c] disabled:opacity-50"
                >
                  PDF 생성
                </button>
              </div>
            </div>
          </div>
        </div>,
        document.body,
      )}
    </div>
  );
}

function ImageBox({ label, src }: { label: string; src: string }) {
  // Zoom/pan + download interactive viewer
  const [scale, setScale] = useState(1);
  const [tx, setTx] = useState(0);
  const [ty, setTy] = useState(0);
  const [dragging, setDragging] = useState(false);
  const dragStart = useRef<{ x: number; y: number; tx: number; ty: number } | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [lightbox, setLightbox] = useState(false);
  // 이미지 자연 비율을 추적 — 컨테이너가 비율에 맞춰 줄어들면서 사진이 채움
  const [aspect, setAspect] = useState<number | null>(null);
  useEffect(() => { setAspect(null); setScale(1); setTx(0); setTy(0); }, [src]);

  const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));
  const MIN = 1, MAX = 8;

  const reset = () => { setScale(1); setTx(0); setTy(0); };

  function onWheel(e: React.WheelEvent<HTMLDivElement>) {
    e.preventDefault();
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect) return;
    const cx = e.clientX - rect.left - rect.width / 2;
    const cy = e.clientY - rect.top - rect.height / 2;
    const factor = e.deltaY < 0 ? 1.18 : 1 / 1.18;
    const nextScale = clamp(scale * factor, MIN, MAX);
    if (nextScale === scale) return;
    const ratio = nextScale / scale;
    setTx((cx - (cx - tx) * ratio));
    setTy((cy - (cy - ty) * ratio));
    setScale(nextScale);
  }

  function onMouseDown(e: React.MouseEvent<HTMLDivElement>) {
    if (scale <= 1) return;
    e.preventDefault();
    setDragging(true);
    dragStart.current = { x: e.clientX, y: e.clientY, tx, ty };
  }
  function onMouseMove(e: React.MouseEvent<HTMLDivElement>) {
    if (!dragging || !dragStart.current) return;
    setTx(dragStart.current.tx + (e.clientX - dragStart.current.x));
    setTy(dragStart.current.ty + (e.clientY - dragStart.current.y));
  }
  function endDrag() { setDragging(false); dragStart.current = null; }

  function onDoubleClick(e: React.MouseEvent<HTMLDivElement>) {
    if (scale > 1) { reset(); return; }
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect) return;
    const cx = e.clientX - rect.left - rect.width / 2;
    const cy = e.clientY - rect.top - rect.height / 2;
    const nextScale = 2.5;
    setScale(nextScale);
    setTx(-cx * (nextScale - 1));
    setTy(-cy * (nextScale - 1));
  }

  const downloadName = (() => {
    const safe = label.replace(/[^\w가-힣.-]+/g, "_").slice(0, 60) || "image";
    return `${safe}.png`;
  })();

  // ESC to close lightbox, lock body scroll while open
  useEffect(() => {
    if (!lightbox) return;
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setLightbox(false);
    };
    window.addEventListener("keydown", onKey);
    return () => {
      document.body.style.overflow = prev;
      window.removeEventListener("keydown", onKey);
    };
  }, [lightbox]);

  const ViewerBody = ({ inLightbox = false }: { inLightbox?: boolean }) => (
    <div
      ref={inLightbox ? undefined : containerRef}
      className={
        "rounded-lg overflow-hidden flex items-center justify-center relative select-none w-full " +
        (inLightbox ? "h-full" : "")
      }
      style={{
        background:
          "repeating-conic-gradient(#161616 0% 25%, #1f1f1f 0% 50%) 50% / 16px 16px, #0e0e0e",
        border: inLightbox ? "none" : "1px solid #e6e6df",
        boxShadow: inLightbox ? "none" : "inset 0 0 0 1px rgba(255,255,255,0.02), 0 1px 0 rgba(20,22,27,0.03)",
        cursor: scale > 1 ? (dragging ? "grabbing" : "grab") : "zoom-in",
        // 컨테이너가 이미지 비율을 따라가도록 → 첫 화면부터 사진이 미리보기 영역을 채움.
        // 비율 정해지기 전엔 안전한 최소 높이 확보.
        aspectRatio: !inLightbox && aspect ? String(aspect) : undefined,
        maxHeight: inLightbox ? undefined : "78vh",
        minHeight: inLightbox ? undefined : (aspect ? undefined : 420),
      }}
      onWheel={inLightbox ? undefined : onWheel}
      onMouseDown={inLightbox ? undefined : onMouseDown}
      onMouseMove={inLightbox ? undefined : onMouseMove}
      onMouseUp={inLightbox ? undefined : endDrag}
      onMouseLeave={inLightbox ? undefined : endDrag}
      onDoubleClick={inLightbox ? undefined : onDoubleClick}
    >
      <img
        src={src}
        alt={label}
        draggable={false}
        onLoad={(e) => {
          const t = e.currentTarget;
          if (t.naturalWidth > 0 && t.naturalHeight > 0) {
            setAspect(t.naturalWidth / t.naturalHeight);
          }
        }}
        className="w-full h-full object-contain transition-transform"
        style={{
          transform: inLightbox ? "none" : `translate(${tx}px, ${ty}px) scale(${scale})`,
          transformOrigin: "center center",
          transition: dragging ? "none" : "transform 0.08s ease-out",
        }}
      />
    </div>
  );

  return (
    <>
      <div className="min-w-0 group">
        <div className="text-[11px] text-[#5a606e] mb-1 flex items-center justify-between gap-2">
          <span className="truncate font-medium">{label}</span>
          <div className="flex items-center gap-1 flex-shrink-0">
            <span className="text-[10px] font-mono text-[#8d92a0] tabular-nums w-10 text-right">
              {(scale * 100).toFixed(0)}%
            </span>
            <ToolbarBtn title="축소" onClick={() => setScale(s => clamp(s / 1.3, MIN, MAX))}>
              <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round"><line x1="5" y1="12" x2="19" y2="12"/></svg>
            </ToolbarBtn>
            <ToolbarBtn title="확대" onClick={() => setScale(s => clamp(s * 1.3, MIN, MAX))}>
              <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
            </ToolbarBtn>
            <ToolbarBtn title="원본 크기로 되돌리기" onClick={reset}>
              <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="1 4 1 10 7 10"/><path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10"/></svg>
            </ToolbarBtn>
            <ToolbarBtn title="크게 보기" onClick={() => setLightbox(true)}>
              <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="15 3 21 3 21 9"/><polyline points="9 21 3 21 3 15"/><line x1="21" y1="3" x2="14" y2="10"/><line x1="3" y1="21" x2="10" y2="14"/></svg>
            </ToolbarBtn>
            <a
              href={src}
              download={downloadName}
              title="이미지 저장"
              onClick={(e) => e.stopPropagation()}
              className="w-6 h-6 inline-flex items-center justify-center rounded-md border border-[#e6e6df] bg-white hover:border-[#2057d4] hover:text-[#2057d4] text-[#5a606e]"
            >
              <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                <polyline points="7 10 12 15 17 10"/>
                <line x1="12" y1="15" x2="12" y2="3"/>
              </svg>
            </a>
          </div>
        </div>
        <ViewerBody />
        {scale > 1 && (
          <div className="text-[9px] text-[#8d92a0] mt-0.5 text-right">
            드래그로 이동 · 휠로 줌 · 더블클릭으로 리셋
          </div>
        )}
      </div>

      {lightbox && (
        <div
          className="fixed inset-0 z-50 bg-black/85 backdrop-blur-sm flex flex-col fade-in"
          onClick={() => setLightbox(false)}
        >
          <div className="flex items-center justify-between px-4 py-2 text-white">
            <div className="text-sm font-semibold truncate">{label}</div>
            <div className="flex items-center gap-2">
              <a
                href={src}
                download={downloadName}
                onClick={(e) => e.stopPropagation()}
                className="px-3 py-1.5 rounded-md bg-white/10 hover:bg-white/20 text-xs font-semibold border border-white/20"
              >
                📥 저장
              </a>
              <button
                onClick={(e) => { e.stopPropagation(); setLightbox(false); }}
                className="w-8 h-8 rounded-md bg-white/10 hover:bg-white/20 border border-white/20 text-base"
                aria-label="닫기"
              >
                ✕
              </button>
            </div>
          </div>
          <div
            className="flex-1 min-h-0 flex items-center justify-center p-4"
            onClick={(e) => e.stopPropagation()}
          >
            <ViewerBody inLightbox />
          </div>
          <div className="text-center text-[11px] text-white/60 py-2">
            배경 클릭 / ESC 로 닫기
          </div>
        </div>
      )}
    </>
  );
}

function ToolbarBtn({
  onClick, title, children,
}: {
  onClick: () => void;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      title={title}
      className="w-6 h-6 inline-flex items-center justify-center rounded-md border border-[#e6e6df] bg-white text-[#5a606e] hover:border-[#2057d4] hover:text-[#2057d4]"
    >
      {children}
    </button>
  );
}

function HybridParamsPanel(props: {
  high: number; setHigh: (n: number) => void;
  low: number; setLow: (n: number) => void;
  boundaryThk: number; setBoundaryThk: (n: number) => void;
  minArea: number; setMinArea: (n: number) => void;
  balance: number; setBalance: (n: number) => void;
  tauScale: number; setTauScale: (n: number) => void;
  yoloConf: number; setYoloConf: (n: number) => void;
  yoloIou: number; setYoloIou: (n: number) => void;
  yoloImgsz: number; setYoloImgsz: (n: number) => void;
  yoloRound: number; setYoloRound: (n: number) => void;
  yoloNoOverlap: boolean; setYoloNoOverlap: (b: boolean) => void;
  yoloRemoveEnclosed: boolean; setYoloRemoveEnclosed: (b: boolean) => void;
  enclosedRatio: number; setEnclosedRatio: (n: number) => void;
  device: string; setDevice: (s: string) => void;
}) {
  const [open, setOpen] = useState(true);
  return (
    <div className="app-card p-4">
      <div className="flex items-center justify-between gap-2 mb-1">
        <div className="min-w-0">
          <h2 className="text-[13px] font-semibold text-[#191919]">Hybrid 파라미터</h2>
          <p className="text-[10.5px] text-[#9ca3af] leading-snug mt-0.5">
            U-Net 확률맵 + 거리변환·Otsu + YOLO 경계 융합
          </p>
        </div>
        <button
          type="button"
          onClick={() => setOpen(o => !o)}
          className="text-[11px] px-2 py-1 rounded-md text-[#6b7280] hover:text-[#191919] hover:bg-[#f5f5f3] font-medium flex items-center gap-1 flex-shrink-0"
        >
          {open ? "접기" : "펼치기"}
          <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"
               style={{ transform: open ? "rotate(180deg)" : "none", transition: "transform 0.18s" }}>
            <polyline points="6 9 12 15 18 9"/>
          </svg>
        </button>
      </div>

      {open && (
        <div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-x-5 gap-y-3 fade-in">
          <div className="space-y-3">
            <SectionHead>U-Net 임계</SectionHead>
            <FloatKnob
              label="high threshold (anchor)"
              desc="이 확률 이상 = 확실한 입자 (메인)"
              value={props.high} onChange={props.setHigh}
              min={0.5} max={0.99} step={0.01}
            />
            <FloatKnob
              label="low threshold (boost)"
              desc="좁은 틈 영역에서만 살리는 보강 임계"
              value={props.low} onChange={props.setLow}
              min={0.05} max={0.7} step={0.01}
            />
            <SectionHead>분리 / 후처리</SectionHead>
            <NumberKnob
              label="YOLO 분리선 두께 (px)"
              desc="1=얇음(noise 민감), 3~5=안정"
              value={props.boundaryThk} onChange={props.setBoundaryThk}
              min={1} max={10}
            />
            <NumberKnob
              label="최소 입자 크기 (px)"
              desc="이 픽셀 미만 component 제거"
              value={props.minArea} onChange={props.setMinArea}
              min={1} max={500}
            />
            <FloatKnob
              label="d₁≈d₂ balance"
              desc="0=spec / 0.25 권장 / 0.5=엄격"
              value={props.balance} onChange={props.setBalance}
              min={0} max={0.7} step={0.05}
            />
            <FloatKnob
              label="τ* 배수 (좁은 틈 검출 범위)"
              desc="1.0=Otsu 자동값. 안 잡히면 1.3~2.0"
              value={props.tauScale} onChange={props.setTauScale}
              min={0.5} max={3.0} step={0.1}
            />
          </div>

          <div className="space-y-3">
            <SectionHead>YOLO 추론</SectionHead>
            <FloatKnob label="confidence" value={props.yoloConf} onChange={props.setYoloConf} min={0} max={1} step={0.01} />
            <FloatKnob label="IoU (NMS)" value={props.yoloIou} onChange={props.setYoloIou} min={0} max={1} step={0.01} />
            <NumberKnob label="imgsz" value={props.yoloImgsz} onChange={props.setYoloImgsz} min={320} max={1600} step={32} />
            <NumberKnob
              label="라운드 / 부드럽게 (Gaussian σ)"
              desc="0=끄기, 3~5 권장. 클수록 동그래짐"
              value={props.yoloRound} onChange={props.setYoloRound}
              min={0} max={10}
            />
            <Toggle
              label="겹친 인스턴스 영역 = 배경"
              desc="겹친 픽셀을 양쪽 모두에서 제거 → 분리선"
              value={props.yoloNoOverlap}
              onChange={props.setYoloNoOverlap}
            />
            <Toggle
              label="포위된 인스턴스 제거"
              desc="큰 입자 외곽선 안에 갇힌 작은 detection 삭제"
              value={props.yoloRemoveEnclosed}
              onChange={props.setYoloRemoveEnclosed}
            />
            {props.yoloRemoveEnclosed && (
              <FloatKnob
                label="포위 판정 비율"
                desc="작은 입자의 이 비율 이상이 큰 것 안에 들면 제거 (0.9 권장)"
                value={props.enclosedRatio} onChange={props.setEnclosedRatio}
                min={0.5} max={1.0} step={0.05}
              />
            )}
            <SectionHead>실행 환경</SectionHead>
            <label className="block text-xs">
              <div className="mb-1 text-[#5a606e]">디바이스</div>
              <select
                value={props.device}
                onChange={(e) => props.setDevice(e.target.value)}
                className="w-full border border-[#e6e6df] rounded-md px-2 py-1.5 text-xs bg-white"
              >
                <option value="auto">자동</option>
                <option value="cpu">CPU</option>
                <option value="mps">MPS (Apple)</option>
                <option value="cuda:0">CUDA</option>
              </select>
            </label>
          </div>
        </div>
      )}
    </div>
  );
}

function SectionHead({ children }: { children: React.ReactNode }) {
  return (
    <div className="text-[10px] uppercase tracking-wider font-semibold text-[#8d92a0] flex items-center gap-2 mt-0.5">
      <span>{children}</span>
      <span className="flex-1 h-px bg-[#eeeee7]" />
    </div>
  );
}

function RunButton({
  onClick, disabled, busy, label, icon,
}: {
  onClick: () => void;
  disabled: boolean;
  busy: boolean;
  label: string;
  icon: "play" | "train" | "download";
}) {
  const Icon = () => {
    if (icon === "play") {
      return (
        <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
          <path d="M8 5v14l11-7z" />
        </svg>
      );
    }
    if (icon === "train") {
      return (
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M12 2v6m0 8v6M4.93 4.93l4.24 4.24m5.66 5.66 4.24 4.24M2 12h6m8 0h6M4.93 19.07l4.24-4.24m5.66-5.66 4.24-4.24"/>
        </svg>
      );
    }
    return (
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
        <polyline points="7 10 12 15 17 10"/>
        <line x1="12" y1="15" x2="12" y2="3"/>
      </svg>
    );
  };
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={
        "w-full py-3 rounded-md font-semibold text-[13px] flex items-center justify-center gap-2 relative overflow-hidden " +
        (disabled ? "btn-primary" : "btn-primary")
      }
    >
      {busy ? (
        <span
          aria-hidden
          className="w-3.5 h-3.5 rounded-full border-2 border-white/30 border-t-white"
          style={{ animation: "spinBtn 0.7s linear infinite" }}
        />
      ) : (
        <Icon />
      )}
      <span>{label}</span>
      <style jsx>{`@keyframes spinBtn { to { transform: rotate(360deg); } }`}</style>
    </button>
  );
}
