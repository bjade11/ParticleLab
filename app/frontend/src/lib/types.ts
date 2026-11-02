// ParticleLab 공용 타입 — DESIGN.md §1 (AnalysisParams), §2.1 (BatchResult), §3 (API) 와 1:1 대응.

// ───────────────────────── §1 AnalysisParams ─────────────────────────

export type EngineId = "watershed" | "ai_hybrid";
export type ScaleMode = "auto" | "manual";
export type EdgeMethod = "laplacian" | "sobelx" | "sobely" | "scharr";
export type MarkerMethod = "peak_local_max" | "dilate_threshold" | "erosion" | "blob_detection";
export type FilterMode = "value" | "percentile";
export type DeviceId = "auto" | "cpu" | "cuda";

export interface ScaleParams {
  mode: ScaleMode;
  /** manual 일 때 μm/px. auto 실패 시 fallback 으로도 사용 */
  pixel_to_micro: number;
  /** auto 모드에서 파일명 매칭 실패 시 값 */
  auto_default: number;
}

export interface SemParams {
  /** 0~1, 하단 스케일바 제거 */
  bottom_crop_ratio: number;
  brightness: number;
  contrast: number;
  gamma: number;
  use_clahe: boolean;
  clahe_clip: number;
  clahe_tile: number;
  use_usm: boolean;
  usm_amount: number;
  usm_radius: number;
}

export interface EdgeParams {
  weight: number;
  kernel: number;
  method: EdgeMethod;
}

export interface MorphOp {
  kernel: number;
  iterations: number;
}

export interface MorphologyParams {
  open: MorphOp;
  close: MorphOp;
  erode: MorphOp;
  dilate: MorphOp;
}

export interface MarkerParams {
  peak_min_distance: number;
  /** dilate_threshold: dist > thr * max */
  distance_threshold: number;
  dilate_kernel: number;
  dilate_iter: number;
  erosion_kernel: number;
  erosion_iter: number;
  blob_min_sigma: number;
  blob_max_sigma: number;
  blob_num_sigma: number;
  blob_threshold: number;
}

export interface WatershedParams {
  /** 홀수 [x, y] */
  blur_kernel: [number, number];
  enhance_edges: boolean;
  edge: EdgeParams;
  /** null 이면 모폴로지 생략. 순서 open→close→erode→dilate */
  morphology: MorphologyParams | null;
  marker_method: MarkerMethod;
  marker: MarkerParams;
  /** remove_small_objects */
  min_size: number;
}

export interface HybridParams {
  high_thresh: number;
  low_thresh: number;
  boundary_thickness: number;
  min_anchor_area: number;
  balance_floor: number;
  tau_scale: number;
  yolo_round_strength: number;
  yolo_remove_overlaps: boolean;
  yolo_remove_enclosed: boolean;
  enclosed_ratio: number;
  yolo_conf: number;
  yolo_iou: number;
  yolo_imgsz: number;
  unet_imgsz: number | null;
  device: DeviceId | string;
  /** "basic" = basic_pt/U-net++ , 또는 서버 절대경로 */
  unet_weights: string;
  yolo_weights: string;
}

export interface MeasureParams {
  filter_border_particles: boolean;
  min_contour_points: number;
  /** μm */
  min_diameter: number;
  exclude_linear_contour: boolean;
  linear_contour_percentile: number;
  exclude_dark_particle: boolean;
  dark_particle_percentile: number;
  /** 0 이면 끔 */
  curvature_percentile: number;
}

export interface AfterFilterParams {
  diameter_lower_percentile: number;
  diameter_upper_percentile: number;
  enable_circularity: boolean;
  circularity_mode: FilterMode;
  circularity_value: number;
  circularity_percentile: number;
  enable_aspect_ratio: boolean;
  aspect_ratio_mode: FilterMode;
  aspect_ratio_value: number;
  aspect_ratio_percentile: number;
}

export interface SpatialParams {
  enable_boundary: boolean;
  boundary_contact_threshold: number;
  enable_nnd: boolean;
  enable_dbscan: boolean;
  dbscan_eps_auto: boolean;
  dbscan_eps_factor: number;
  dbscan_eps_value: number;
  dbscan_min_samples_auto: boolean;
  dbscan_min_samples_factor: number;
  dbscan_min_samples_value: number;
}

export interface OutputParams {
  make_ppt: boolean;
  make_contour_only: boolean;
  make_normal_dist: boolean;
  jpeg_quality: number;
}

export interface AnalysisParams {
  engine: EngineId;
  scale: ScaleParams;
  sem: SemParams;
  watershed: WatershedParams;
  hybrid: HybridParams;
  measure: MeasureParams;
  after_filter: AfterFilterParams;
  spatial: SpatialParams;
  output: OutputParams;
}

// ───────────────────────── §3.1 튜너 ─────────────────────────

export interface TunerPreprocessResponse {
  preview_png: string;
  shape: [number, number];
}

export type TunerViewKey =
  | "original"
  | "preprocessed"
  | "binary"
  | "distance"
  | "markers"
  | "labels"
  | "labeled_before"
  | "labeled_after"
  | "contour_before";

export type TunerViews = Record<TunerViewKey, string | null>;

export interface DValues {
  d10: number;
  d50: number;
  d90: number;
  span: number;
  mean: number;
  std: number;
  mode: number;
  cv: number;
  n: number;
}

/** 입자 1개 행. 컬럼명은 백엔드 DataFrame 과 동일 (공백·단위 포함). */
export interface ParticleRow {
  Label: number;
  x: number;
  y: number;
  "Diameter (μm)": number;
  Circularity: number;
  "Aspect Ratio": number;
  "Area (μm^2)": number;
  Solidity: number;
  "Mean Intensity": number;
  [key: string]: number | string | null | undefined;
}

export interface TunerSegmentResponse {
  engine: EngineId;
  pixel_to_micro: number;
  shape: [number, number];
  n_before: number;
  n_after: number;
  views: TunerViews;
  particles_before: ParticleRow[];
  particles_after: ParticleRow[];
  d_values_before: DValues;
  d_values_after: DValues;
  timing_ms: { preprocess: number; segment: number; measure: number; [k: string]: number };
}

// ───────────────────────── §2.1 / §3.2 배치 ─────────────────────────

export type JobStatus = "queued" | "running" | "done" | "error" | "cancelled";

export interface ImageResult {
  filename: string;
  sample: string;
  pixel_to_micro: number;
  n_before: number;
  n_after: number;
  /** 카테고리 → output_dir 기준 상대경로 */
  files: Record<string, string>;
}

/** Summary 시트의 1행. 컬럼명은 particle_core.stats.summary_row 가 결정 → 느슨하게 받는다. */
export type SummaryRow = Record<string, number | string | null | undefined>;

export interface BatchResult {
  output_dir: string;
  excel_path: string;
  pptx_path: string | null;
  images: ImageResult[];
  samples: SummaryRow[];
  /** sample → {hist_diameter, hist_circularity, hist_aspect, summary_before, summary_after} */
  sample_files: Record<string, Record<string, string>>;
  /** normal_dist 등 */
  extra_files: Record<string, string[]>;
  warnings: string[];
}

/** GET /api/batch/jobs 목록 항목 */
export interface BatchJobSummary {
  job_id: string;
  name: string;
  status: JobStatus;
  progress: number;
  created_at: string | number | null;
  output_dir: string;
  n_images: number;
}

/** GET /api/batch/jobs/{id} */
export interface BatchJob {
  job_id: string;
  name?: string;
  status: JobStatus;
  progress: number;
  message: string;
  log: string[];
  started_at: string | number | null;
  finished_at: string | number | null;
  error: string | null;
  result: BatchResult | null;
  output_dir?: string;
  n_images?: number;
}

export interface CreateJobResponse {
  job_id: string;
  output_dir: string;
  n_images: number;
}

// ───────────────────────── §3.3 서버 폴더 ─────────────────────────

export interface FsList {
  path: string;
  parent: string | null;
  dirs: { name: string; path: string }[];
  n_images: number;
  images: string[];
}

// ───────────────────────── 기존 유지 ─────────────────────────

export interface HealthResponse {
  ok?: boolean;
  status?: string;
  [k: string]: unknown;
}

export interface BasicPtResponse {
  [k: string]: unknown;
}
