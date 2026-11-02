"use client";

// AnalysisParams 기본값 + localStorage 영속 훅 + JSON 내보내기/불러오기 (레거시 데스크톱 포맷 매핑 포함).

import { useCallback, useEffect, useRef, useState } from "react";
import type {
  AnalysisParams,
  EdgeMethod,
  EngineId,
  FilterMode,
  MarkerMethod,
  MorphologyParams,
  ScaleMode,
} from "./types";

export const PARAMS_STORAGE_KEY = "plab.params";

// DESIGN.md §1 기본값과 동일.
export const DEFAULT_PARAMS: AnalysisParams = {
  engine: "watershed",
  scale: {
    mode: "auto",
    pixel_to_micro: 0.198,
    auto_default: 0.7,
  },
  sem: {
    bottom_crop_ratio: 0.074,
    brightness: 0,
    contrast: 1.0,
    gamma: 1.0,
    use_clahe: false,
    clahe_clip: 2.0,
    clahe_tile: 8,
    use_usm: false,
    usm_amount: 0.6,
    usm_radius: 1.5,
  },
  watershed: {
    blur_kernel: [3, 3],
    enhance_edges: false,
    edge: { weight: 0.5, kernel: 3, method: "laplacian" },
    morphology: {
      open: { kernel: 3, iterations: 1 },
      close: { kernel: 3, iterations: 1 },
      erode: { kernel: 3, iterations: 1 },
      dilate: { kernel: 3, iterations: 1 },
    },
    marker_method: "peak_local_max",
    marker: {
      peak_min_distance: 5,
      distance_threshold: 0.3,
      dilate_kernel: 3,
      dilate_iter: 1,
      erosion_kernel: 5,
      erosion_iter: 2,
      blob_min_sigma: 3.0,
      blob_max_sigma: 30.0,
      blob_num_sigma: 10,
      blob_threshold: 0.1,
    },
    min_size: 30,
  },
  hybrid: {
    high_thresh: 0.75,
    low_thresh: 0.4,
    boundary_thickness: 3,
    min_anchor_area: 10,
    balance_floor: 0.25,
    tau_scale: 1.0,
    yolo_round_strength: 3,
    yolo_remove_overlaps: true,
    yolo_remove_enclosed: true,
    enclosed_ratio: 0.9,
    yolo_conf: 0.15,
    yolo_iou: 0.5,
    yolo_imgsz: 640,
    unet_imgsz: null,
    device: "auto",
    unet_weights: "basic",
    yolo_weights: "basic",
  },
  measure: {
    filter_border_particles: true,
    min_contour_points: 30,
    min_diameter: 0.0,
    exclude_linear_contour: false,
    linear_contour_percentile: 10.0,
    exclude_dark_particle: false,
    dark_particle_percentile: 10.0,
    curvature_percentile: 0.0,
  },
  after_filter: {
    diameter_lower_percentile: 0.0,
    diameter_upper_percentile: 0.0,
    enable_circularity: true,
    circularity_mode: "value",
    circularity_value: 0.7,
    circularity_percentile: 5.0,
    enable_aspect_ratio: true,
    aspect_ratio_mode: "value",
    aspect_ratio_value: 2.0,
    aspect_ratio_percentile: 5.0,
  },
  spatial: {
    enable_boundary: false,
    boundary_contact_threshold: 2.0,
    enable_nnd: false,
    enable_dbscan: false,
    dbscan_eps_auto: true,
    dbscan_eps_factor: 2.5,
    dbscan_eps_value: 5.0,
    dbscan_min_samples_auto: true,
    dbscan_min_samples_factor: 0.01,
    dbscan_min_samples_value: 2,
  },
  output: {
    make_ppt: true,
    make_contour_only: true,
    make_normal_dist: true,
    jpeg_quality: 95,
  },
};

/** 기본 모폴로지 블록 (토글 ON 시 복원용) */
export const DEFAULT_MORPHOLOGY: MorphologyParams = {
  open: { kernel: 3, iterations: 1 },
  close: { kernel: 3, iterations: 1 },
  erode: { kernel: 3, iterations: 1 },
  dilate: { kernel: 3, iterations: 1 },
};

// ───────────────────────── 유틸 ─────────────────────────

type Dict = Record<string, unknown>;

export function isDict(v: unknown): v is Dict {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

export function deepClone<T>(v: T): T {
  return JSON.parse(JSON.stringify(v)) as T;
}

/**
 * 기본값 위에 부분 객체를 덮어쓴다. 기본값 쪽에 없는 키는 버리고, 타입이 맞지 않으면 기본값 유지.
 * 배열(blur_kernel)은 통째로 교체. `morphology: null` 은 허용(명시적 null).
 */
function mergeWithDefaults<T>(def: T, src: unknown): T {
  if (src === null) return def;
  if (Array.isArray(def)) {
    return (Array.isArray(src) ? (src as unknown as T) : def) as T;
  }
  if (isDict(def)) {
    if (!isDict(src)) return def;
    const out: Dict = {};
    for (const key of Object.keys(def)) {
      const dv = (def as Dict)[key];
      const sv = src[key];
      if (sv === undefined) {
        out[key] = dv;
      } else if (dv === null) {
        // 기본값이 null 인 필드 (hybrid.unet_imgsz): 숫자/null 만 허용
        out[key] = typeof sv === "number" || sv === null ? sv : dv;
      } else if (isDict(dv)) {
        // 기본값이 객체인데 소스가 null 이면 (morphology) null 허용
        out[key] = sv === null ? null : mergeWithDefaults(dv, sv);
      } else if (Array.isArray(dv)) {
        out[key] = Array.isArray(sv) ? sv : dv;
      } else if (typeof dv === typeof sv) {
        out[key] = sv;
      } else if (typeof dv === "number" && typeof sv === "string" && sv.trim() !== "" && !Number.isNaN(Number(sv))) {
        out[key] = Number(sv);
      } else {
        out[key] = dv;
      }
    }
    return out as T;
  }
  return (typeof def === typeof src ? (src as T) : def) as T;
}

/** 임의 입력을 AnalysisParams 로 정규화 (기본값과 병합). */
export function normalizeParams(src: unknown): AnalysisParams {
  const p = mergeWithDefaults(DEFAULT_PARAMS, src);
  // blur_kernel 홀수 보정
  p.watershed.blur_kernel = [oddify(p.watershed.blur_kernel?.[0]), oddify(p.watershed.blur_kernel?.[1])];
  // morphology 가 null 이 아니면 각 op 를 기본값과 병합
  if (p.watershed.morphology !== null) {
    p.watershed.morphology = mergeWithDefaults(DEFAULT_MORPHOLOGY, p.watershed.morphology);
  }
  return p;
}

export function oddify(v: unknown, fallback = 3): number {
  const n = Math.round(Number(v));
  if (!Number.isFinite(n) || n < 1) return fallback;
  return n % 2 === 0 ? n + 1 : n;
}

// ───────────────────────── localStorage 훅 ─────────────────────────

function loadStored(): AnalysisParams {
  if (typeof window === "undefined") return deepClone(DEFAULT_PARAMS);
  try {
    const raw = window.localStorage.getItem(PARAMS_STORAGE_KEY);
    if (!raw) return deepClone(DEFAULT_PARAMS);
    return normalizeParams(JSON.parse(raw));
  } catch {
    return deepClone(DEFAULT_PARAMS);
  }
}

export function saveParams(p: AnalysisParams) {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(PARAMS_STORAGE_KEY, JSON.stringify(p));
  } catch {
    /* ignore quota / private mode */
  }
}

export type ParamsUpdater = AnalysisParams | ((prev: AnalysisParams) => AnalysisParams);

/**
 * localStorage("plab.params") 와 동기화되는 AnalysisParams 훅.
 * 첫 렌더는 기본값 → 마운트 후 저장값 로드(`loaded` 플래그). 변경 시 자동 저장.
 */
export function useAnalysisParams() {
  const [params, setParamsState] = useState<AnalysisParams>(() => deepClone(DEFAULT_PARAMS));
  const [loaded, setLoaded] = useState(false);
  const skipSave = useRef(true);

  useEffect(() => {
    setParamsState(loadStored());
    setLoaded(true);
    // 다른 탭에서 변경된 경우 반영
    const onStorage = (e: StorageEvent) => {
      if (e.key === PARAMS_STORAGE_KEY && e.newValue) {
        try {
          skipSave.current = true;
          setParamsState(normalizeParams(JSON.parse(e.newValue)));
        } catch {
          /* ignore */
        }
      }
    };
    window.addEventListener("storage", onStorage);
    return () => window.removeEventListener("storage", onStorage);
  }, []);

  useEffect(() => {
    if (!loaded) return;
    if (skipSave.current) {
      skipSave.current = false;
      return;
    }
    saveParams(params);
  }, [params, loaded]);

  const setParams = useCallback((u: ParamsUpdater) => {
    setParamsState((prev) => (typeof u === "function" ? u(prev) : u));
  }, []);

  const reset = useCallback(() => setParamsState(deepClone(DEFAULT_PARAMS)), []);

  return { params, setParams, reset, loaded } as const;
}

/** DESIGN §4 의 이름(useParams)도 제공 — Next 의 useParams 와 혼동되지 않게 별칭만. */
export const useParams = useAnalysisParams;

// ───────────────────────── 내보내기 ─────────────────────────

export interface ParamsFile {
  format: "particlelab.params";
  version: 1;
  exported_at: string;
  params: AnalysisParams;
}

export function exportParamsJson(p: AnalysisParams): string {
  const file: ParamsFile = {
    format: "particlelab.params",
    version: 1,
    exported_at: new Date().toISOString(),
    params: p,
  };
  return JSON.stringify(file, null, 2);
}

export function downloadParamsJson(p: AnalysisParams, filename = "particlelab_params.json") {
  const blob = new Blob([exportParamsJson(p)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

// ───────────────────────── 불러오기 ─────────────────────────

const LEGACY_FLAT_KEYS = [
  "bottom_crop_ratio",
  "blur_kernel_size",
  "enhance_edges",
  "edge_params",
  "morphology_params",
  "marker_method",
  "marker_params",
  "distance_threshold",
  "min_contour_points",
  "filter_border_particles",
  "min_diameter",
  "max_diameter",
  "exclude_linear_contour",
  "linear_contour_threshold",
  "exclude_dark_particle",
  "dark_particle_percentile",
  "pixel_to_micro",
  "pixel_to_micro_auto",
  "diameter_lower_percentile",
  "diameter_upper_percentile",
  "circularity_mode",
  "aspect_ratio_mode",
  "enable_boundary_analysis",
  "enable_nnd_analysis",
  "enable_dbscan",
  "sem_preprocessing_params",
];

function isOurFormat(d: Dict): boolean {
  if (d.format === "particlelab.params" && isDict(d.params)) return true;
  // 포장 없는 AnalysisParams 자체
  const nested = ["scale", "sem", "watershed", "hybrid", "measure", "after_filter", "spatial", "output"];
  return nested.filter((k) => isDict(d[k]) || d[k] === null).length >= 2 || (typeof d.engine === "string" && nested.some((k) => k in d));
}

function isLegacyFormat(d: Dict): boolean {
  if (isDict(d.analysis_params)) return true;
  return LEGACY_FLAT_KEYS.filter((k) => k in d).length >= 3;
}

const num = (v: unknown, fallback: number): number => {
  const n = typeof v === "string" ? Number(v) : v;
  return typeof n === "number" && Number.isFinite(n) ? n : fallback;
};
const bool = (v: unknown, fallback: boolean): boolean => {
  if (typeof v === "boolean") return v;
  if (typeof v === "number") return v !== 0;
  if (typeof v === "string") {
    const s = v.trim().toLowerCase();
    if (["true", "1", "yes", "on"].includes(s)) return true;
    if (["false", "0", "no", "off", ""].includes(s)) return false;
  }
  return fallback;
};
const oneOf = <T extends string>(v: unknown, allowed: readonly T[], fallback: T): T =>
  typeof v === "string" && (allowed as readonly string[]).includes(v) ? (v as T) : fallback;

/** 1e10 처럼 "무한" 으로 쓰인 값은 무시 */
const isHuge = (v: unknown) => typeof v === "number" && (v >= 1e9 || !Number.isFinite(v));

/**
 * 레거시 데스크톱 포맷(particle_analyzer_settings.json / {"analysis_params": …, "sem_preprocessing_params": …})
 * → AnalysisParams. 명시되지 않은 필드는 `base` (기본 DEFAULT_PARAMS) 를 따른다.
 */
export function mapLegacyParams(input: Dict, base: AnalysisParams = DEFAULT_PARAMS): AnalysisParams {
  const ap: Dict = isDict(input.analysis_params) ? (input.analysis_params as Dict) : input;
  // sem_preprocessing_params 는 바깥 wrapper 또는 analysis_params 안쪽 어디에나 올 수 있음
  const semSrc: Dict = isDict(input.sem_preprocessing_params)
    ? (input.sem_preprocessing_params as Dict)
    : isDict(ap.sem_preprocessing_params)
      ? (ap.sem_preprocessing_params as Dict)
      : {};

  const p = deepClone(base);

  // ── scale
  if ("pixel_to_micro_auto" in ap) p.scale.mode = bool(ap.pixel_to_micro_auto, p.scale.mode === "auto") ? "auto" : "manual";
  if ("pixel_to_micro" in ap) p.scale.pixel_to_micro = num(ap.pixel_to_micro, p.scale.pixel_to_micro);
  if ("pixel_to_micro_default" in ap) p.scale.auto_default = num(ap.pixel_to_micro_default, p.scale.auto_default);

  // ── sem (sem_preprocessing_params 우선, 없으면 flat 의 bottom_crop_ratio)
  const crop = semSrc.bottom_crop_ratio ?? ap.bottom_crop_ratio;
  if (crop !== undefined) p.sem.bottom_crop_ratio = num(crop, p.sem.bottom_crop_ratio);
  if ("brightness" in semSrc) p.sem.brightness = num(semSrc.brightness, p.sem.brightness);
  if ("contrast" in semSrc) p.sem.contrast = num(semSrc.contrast, p.sem.contrast);
  if ("gamma" in semSrc) p.sem.gamma = num(semSrc.gamma, p.sem.gamma);
  if ("use_clahe" in semSrc) p.sem.use_clahe = bool(semSrc.use_clahe, p.sem.use_clahe);
  if ("clahe_clip" in semSrc) p.sem.clahe_clip = num(semSrc.clahe_clip, p.sem.clahe_clip);
  if ("clahe_tile" in semSrc) p.sem.clahe_tile = num(semSrc.clahe_tile, p.sem.clahe_tile);
  if ("use_usm" in semSrc) p.sem.use_usm = bool(semSrc.use_usm, p.sem.use_usm);
  if ("usm_amount" in semSrc) p.sem.usm_amount = num(semSrc.usm_amount, p.sem.usm_amount);
  if ("usm_radius" in semSrc) p.sem.usm_radius = num(semSrc.usm_radius, p.sem.usm_radius);

  // ── watershed
  const blur = semSrc.blur_kernel_size ?? ap.blur_kernel_size;
  if (Array.isArray(blur) && blur.length >= 2) {
    p.watershed.blur_kernel = [oddify(blur[0]), oddify(blur[1])];
  } else if (typeof blur === "number") {
    p.watershed.blur_kernel = [oddify(blur), oddify(blur)];
  }
  if ("enhance_edges" in ap) p.watershed.enhance_edges = bool(ap.enhance_edges, p.watershed.enhance_edges);
  if (isDict(ap.edge_params)) {
    const e = ap.edge_params;
    p.watershed.edge = {
      weight: num(e.weight, p.watershed.edge.weight),
      kernel: oddify(e.kernel ?? p.watershed.edge.kernel),
      method: oneOf<EdgeMethod>(e.method, ["laplacian", "sobelx", "sobely", "scharr"], p.watershed.edge.method),
    };
  }
  if ("morphology_params" in ap) {
    const m = ap.morphology_params;
    if (m === null || m === undefined || (isDict(m) && Object.keys(m).length === 0)) {
      p.watershed.morphology = null;
    } else if (isDict(m)) {
      const mo = (k: string, d: { kernel: number; iterations: number }) => {
        const v = m[k];
        if (!isDict(v)) return d;
        return { kernel: num(v.kernel, d.kernel), iterations: num(v.iterations, d.iterations) };
      };
      p.watershed.morphology = {
        open: mo("open", DEFAULT_MORPHOLOGY.open),
        close: mo("close", DEFAULT_MORPHOLOGY.close),
        erode: mo("erode", DEFAULT_MORPHOLOGY.erode),
        dilate: mo("dilate", DEFAULT_MORPHOLOGY.dilate),
      };
    }
  }
  if ("marker_method" in ap) {
    p.watershed.marker_method = oneOf<MarkerMethod>(
      ap.marker_method,
      ["peak_local_max", "dilate_threshold", "erosion", "blob_detection"],
      p.watershed.marker_method,
    );
  }
  if (isDict(ap.marker_params)) {
    const mk = ap.marker_params;
    const M = p.watershed.marker;
    p.watershed.marker = {
      peak_min_distance: num(mk.peak_min_distance, M.peak_min_distance),
      distance_threshold: num(mk.distance_threshold ?? ap.distance_threshold, M.distance_threshold),
      dilate_kernel: num(mk.dilate_kernel, M.dilate_kernel),
      dilate_iter: num(mk.dilate_iter, M.dilate_iter),
      erosion_kernel: num(mk.erosion_kernel, M.erosion_kernel),
      erosion_iter: num(mk.erosion_iter, M.erosion_iter),
      blob_min_sigma: num(mk.blob_min_sigma, M.blob_min_sigma),
      blob_max_sigma: num(mk.blob_max_sigma, M.blob_max_sigma),
      blob_num_sigma: num(mk.blob_num_sigma, M.blob_num_sigma),
      blob_threshold: num(mk.blob_threshold, M.blob_threshold),
    };
  } else if ("distance_threshold" in ap) {
    p.watershed.marker.distance_threshold = num(ap.distance_threshold, p.watershed.marker.distance_threshold);
  }
  if ("min_size" in ap) p.watershed.min_size = num(ap.min_size, p.watershed.min_size);

  // ── measure
  if ("filter_border_particles" in ap) p.measure.filter_border_particles = bool(ap.filter_border_particles, p.measure.filter_border_particles);
  if ("min_contour_points" in ap) p.measure.min_contour_points = num(ap.min_contour_points, p.measure.min_contour_points);
  if ("min_diameter" in ap) p.measure.min_diameter = num(ap.min_diameter, p.measure.min_diameter);
  // max_diameter: 1e10 = 무한 → 무시. 스키마에 대응 필드가 없어 유한값도 보존하지 않음.
  if ("exclude_linear_contour" in ap) p.measure.exclude_linear_contour = bool(ap.exclude_linear_contour, p.measure.exclude_linear_contour);
  const lin = ap.linear_contour_percentile ?? ap.linear_contour_threshold;
  if (lin !== undefined && !isHuge(lin)) p.measure.linear_contour_percentile = num(lin, p.measure.linear_contour_percentile);
  if ("exclude_dark_particle" in ap) p.measure.exclude_dark_particle = bool(ap.exclude_dark_particle, p.measure.exclude_dark_particle);
  if ("dark_particle_percentile" in ap) p.measure.dark_particle_percentile = num(ap.dark_particle_percentile, p.measure.dark_particle_percentile);
  if ("curvature_percentile" in ap) p.measure.curvature_percentile = num(ap.curvature_percentile, p.measure.curvature_percentile);

  // ── after_filter
  const af = p.after_filter;
  if ("diameter_lower_percentile" in ap) af.diameter_lower_percentile = num(ap.diameter_lower_percentile, af.diameter_lower_percentile);
  if ("diameter_upper_percentile" in ap) af.diameter_upper_percentile = num(ap.diameter_upper_percentile, af.diameter_upper_percentile);
  if ("circularity_mode" in ap) af.circularity_mode = oneOf<FilterMode>(ap.circularity_mode, ["value", "percentile"], af.circularity_mode);
  if ("circularity_value" in ap) af.circularity_value = num(ap.circularity_value, af.circularity_value);
  if ("circularity_percentile" in ap) af.circularity_percentile = num(ap.circularity_percentile, af.circularity_percentile);
  if ("aspect_ratio_mode" in ap) af.aspect_ratio_mode = oneOf<FilterMode>(ap.aspect_ratio_mode, ["value", "percentile"], af.aspect_ratio_mode);
  if ("aspect_ratio_value" in ap) af.aspect_ratio_value = num(ap.aspect_ratio_value, af.aspect_ratio_value);
  if ("aspect_ratio_percentile" in ap) af.aspect_ratio_percentile = num(ap.aspect_ratio_percentile, af.aspect_ratio_percentile);
  // 레거시: circularity_threshold / aspect_ratio_threshold (값 모드)
  if ("circularity_threshold" in ap && !("circularity_value" in ap)) af.circularity_value = num(ap.circularity_threshold, af.circularity_value);
  if ("aspect_ratio_threshold" in ap && !("aspect_ratio_value" in ap)) af.aspect_ratio_value = num(ap.aspect_ratio_threshold, af.aspect_ratio_value);

  // ── spatial
  const sp = p.spatial;
  if ("enable_boundary_analysis" in ap) sp.enable_boundary = bool(ap.enable_boundary_analysis, sp.enable_boundary);
  if ("boundary_contact_threshold" in ap) sp.boundary_contact_threshold = num(ap.boundary_contact_threshold, sp.boundary_contact_threshold);
  if ("enable_nnd_analysis" in ap) sp.enable_nnd = bool(ap.enable_nnd_analysis, sp.enable_nnd);
  if ("enable_dbscan" in ap) sp.enable_dbscan = bool(ap.enable_dbscan, sp.enable_dbscan);
  if ("dbscan_eps_auto" in ap) sp.dbscan_eps_auto = bool(ap.dbscan_eps_auto, sp.dbscan_eps_auto);
  if ("dbscan_eps_factor" in ap) sp.dbscan_eps_factor = num(ap.dbscan_eps_factor, sp.dbscan_eps_factor);
  if ("dbscan_eps_value" in ap) sp.dbscan_eps_value = num(ap.dbscan_eps_value, sp.dbscan_eps_value);
  else if ("dbscan_eps" in ap) sp.dbscan_eps_value = num(ap.dbscan_eps, sp.dbscan_eps_value);
  if ("dbscan_min_samples_auto" in ap) sp.dbscan_min_samples_auto = bool(ap.dbscan_min_samples_auto, sp.dbscan_min_samples_auto);
  if ("dbscan_min_samples_factor" in ap) sp.dbscan_min_samples_factor = num(ap.dbscan_min_samples_factor, sp.dbscan_min_samples_factor);
  if ("dbscan_min_samples_value" in ap) sp.dbscan_min_samples_value = num(ap.dbscan_min_samples_value, sp.dbscan_min_samples_value);
  else if ("dbscan_min_samples" in ap) sp.dbscan_min_samples_value = num(ap.dbscan_min_samples, sp.dbscan_min_samples_value);

  // ── output (레거시에 있을 수도 있음)
  if ("make_ppt" in ap) p.output.make_ppt = bool(ap.make_ppt, p.output.make_ppt);
  if ("make_contour_only" in ap) p.output.make_contour_only = bool(ap.make_contour_only, p.output.make_contour_only);
  if ("make_normal_dist" in ap) p.output.make_normal_dist = bool(ap.make_normal_dist, p.output.make_normal_dist);
  if ("jpeg_quality" in ap) p.output.jpeg_quality = num(ap.jpeg_quality, p.output.jpeg_quality);

  // 레거시는 전부 watershed
  p.engine = "watershed";
  return normalizeParams(p);
}

export interface ImportResult {
  params: AnalysisParams;
  source: "particlelab" | "legacy";
}

/**
 * JSON 텍스트 → AnalysisParams.
 * (a) 자체 포맷 {format:"particlelab.params", params} 또는 AnalysisParams 그대로
 * (b) 레거시 데스크톱 flat dict / {"analysis_params": …, "sem_preprocessing_params": …}
 * 인식 불가 시 throw.
 */
export function importParamsJson(text: string, base: AnalysisParams = DEFAULT_PARAMS): ImportResult {
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    throw new Error("JSON 파싱 실패: 올바른 JSON 파일이 아닙니다.");
  }
  if (!isDict(parsed)) throw new Error("JSON 최상위가 객체가 아닙니다.");

  if (isOurFormat(parsed)) {
    const src = parsed.format === "particlelab.params" ? (parsed.params as Dict) : parsed;
    const merged = mergeWithDefaults(base, src);
    const eng = oneOf<EngineId>(merged.engine, ["watershed", "ai_hybrid"], base.engine);
    const mode = oneOf<ScaleMode>(merged.scale?.mode, ["auto", "manual"], base.scale.mode);
    const out = normalizeParams({ ...merged, engine: eng, scale: { ...merged.scale, mode } });
    return { params: out, source: "particlelab" };
  }
  if (isLegacyFormat(parsed)) {
    return { params: mapLegacyParams(parsed, base), source: "legacy" };
  }
  throw new Error("알 수 없는 파라미터 파일 형식입니다.");
}
