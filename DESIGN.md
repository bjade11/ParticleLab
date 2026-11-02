# ParticleLab — 통합 설계 명세

SEM 입자 분석 웹앱. 기존 두 앱을 합친다.

- **기반 (주요 기능)**: `particle_analyzer` / `particle_parameter_tuner` (PyQt 데스크톱, watershed 기반).
  배치 분석(폴더 → Excel/PPT/이미지), 파라미터 튜너(단일 이미지 미리보기)를 웹으로 옮긴다.
- **추가 옵션**: `LIB-Particle_Lab_v2` 의 AI 분할 엔진 (U-Net × YOLO Hybrid).
  분할(segmentation) 단계만 교체 가능한 **엔진**으로 만들고, 그 뒤의 특성 추출 → 필터 → 통계 → Excel/PPT 는 공유한다.

원본 코드 위치 (읽기 전용 참고):
- `C:\posco2\particle_analyzer, particle_parameter_tuner\` — `particle_analysis_functions.py`, `particle_analyzer_threads.py`, `particle_analyzer_app.py`, `particle_parameter_tuner.py`
- `C:\posco2\LIB-Particle_Lab_v2-main\` — `hybrid_core.py`, `app/backend/app/api/hybrid.py`, `app/frontend/src/app/page.tsx`

## 0. 디렉터리

```
C:\posco2\ParticleLab\
  particle_core/            ← 2번 앱의 순수 로직 포팅 (PyQt 의존 0, matplotlib Agg)
  hybrid_core.py, unet_*.py, weight_maps.py, evaluate.py   ← 1번 AI 코어 (그대로)
  basic_pt/U-net++, basic_pt/yolo_11                        ← 기본 가중치
  app/backend/app/
      main.py               FastAPI 엔트리
      api/hybrid.py         기존 AI 진단 엔드포인트 (그대로 유지)
      api/tuner.py          튜너 API
      api/batch.py          배치 job API
      api/fs.py             서버 폴더 탐색 API
      jobs.py               job 레지스트리 (스레드 실행, 진행률)
  app/frontend/src/
      app/page.tsx          튜너 페이지  (/)
      app/batch/page.tsx    배치 분석 페이지 (/batch)
      app/hybrid-lab/page.tsx   기존 v2 AI 진단 콘솔 (복사본, 그대로)
      components/...        공용 컴포넌트
      lib/api.ts, lib/params.ts, lib/types.ts
  runs/                     배치 결과 기본 출력 위치 (gitignore)
  START.bat / SETUP.bat / STOP.bat / launcher.py
```

## 1. 통합 파라미터 스키마 (`AnalysisParams`, JSON)

프런트 ↔ 백엔드 ↔ particle_core 가 모두 이 구조를 쓴다. 백엔드는 pydantic 모델로 검증하고,
particle_core 는 같은 필드명을 가진 dataclass 를 쓴다. 기본값은 2번 앱의 UI 기본값을 따른다.

```jsonc
{
  "engine": "watershed",            // "watershed" | "ai_hybrid"
  "scale": {
    "mode": "auto",                 // "auto": 파일명에서 배율 추출 (get_pixel_to_micro_ratio) / "manual"
    "pixel_to_micro": 0.198,        // manual 일 때 μm/px. auto 실패 시 fallback 으로도 사용
    "auto_default": 0.7             // auto 모드에서 파일명 매칭 실패 시 값
  },
  "sem": {                          // SEM 전처리 (apply_sem_preprocessing). 순서: crop → B/C → gamma → CLAHE → USM
    "bottom_crop_ratio": 0.074,     // 0~1, 하단 스케일바 제거
    "brightness": 0,  "contrast": 1.0,  "gamma": 1.0,
    "use_clahe": false, "clahe_clip": 2.0, "clahe_tile": 8,
    "use_usm": false,   "usm_amount": 0.6, "usm_radius": 1.5
  },
  "watershed": {                    // engine == watershed 일 때
    "blur_kernel": [3, 3],          // 홀수. GaussianBlur → Otsu 직전에 1회만 적용 (원본의 이중 블러 버그 수정)
    "enhance_edges": false,
    "edge": { "weight": 0.5, "kernel": 3, "method": "laplacian" },   // laplacian|sobelx|sobely|scharr
    "morphology": {                 // null 이면 모폴로지 생략. 순서 open→close→erode→dilate
      "open":   { "kernel": 3, "iterations": 1 },
      "close":  { "kernel": 3, "iterations": 1 },
      "erode":  { "kernel": 3, "iterations": 1 },
      "dilate": { "kernel": 3, "iterations": 1 }
    },
    "marker_method": "peak_local_max",   // peak_local_max | dilate_threshold | erosion | blob_detection
    "marker": {
      "peak_min_distance": 5,
      "distance_threshold": 0.3,    // dilate_threshold: dist > thr * max
      "dilate_kernel": 3, "dilate_iter": 1,
      "erosion_kernel": 5, "erosion_iter": 2,
      "blob_min_sigma": 3.0, "blob_max_sigma": 30.0, "blob_num_sigma": 10, "blob_threshold": 0.1
    },
    "min_size": 30                  // remove_small_objects (원본은 항상 30 고정이던 버그 수정: 실제 전달)
  },
  "hybrid": {                       // engine == ai_hybrid 일 때 (hybrid_core.HybridParams + 추론 옵션)
    "high_thresh": 0.75, "low_thresh": 0.40, "boundary_thickness": 3, "min_anchor_area": 10,
    "balance_floor": 0.25, "tau_scale": 1.0, "yolo_round_strength": 3,
    "yolo_remove_overlaps": true, "yolo_remove_enclosed": true, "enclosed_ratio": 0.9,
    "yolo_conf": 0.15, "yolo_iou": 0.5, "yolo_imgsz": 640, "unet_imgsz": null, "device": "auto",
    "unet_weights": "basic",        // "basic" = basic_pt/U-net++ , 또는 서버 절대경로
    "yolo_weights": "basic"
  },
  "measure": {                      // 라벨맵 → 입자 특성 (regionprops) 단계의 필터
    "filter_border_particles": true,
    "min_contour_points": 30,
    "min_diameter": 0.0,            // μm
    "exclude_linear_contour": false, "linear_contour_percentile": 10.0,
    "exclude_dark_particle": false,  "dark_particle_percentile": 10.0,
    "curvature_percentile": 0.0     // 0 이면 끔
  },
  "after_filter": {                 // Before → After 필터 (원본 threads.py 572-591 과 동일 의미)
    "diameter_lower_percentile": 0.0, "diameter_upper_percentile": 0.0,
    "circularity_mode": "value",    // value | percentile
    "circularity_value": 0.7, "circularity_percentile": 5.0,
    "aspect_ratio_mode": "value",
    "aspect_ratio_value": 2.0, "aspect_ratio_percentile": 5.0
  },
  "spatial": {
    "enable_boundary": false, "boundary_contact_threshold": 2.0,
    "enable_nnd": false,
    "enable_dbscan": false,
    "dbscan_eps_auto": true, "dbscan_eps_factor": 2.5, "dbscan_eps_value": 5.0,
    "dbscan_min_samples_auto": true, "dbscan_min_samples_factor": 0.01, "dbscan_min_samples_value": 2
  },
  "output": {
    "make_ppt": true, "make_contour_only": true, "make_normal_dist": true,
    "jpeg_quality": 95
  }
}
```

## 2. particle_core (Python 패키지, 헤드리스)

규칙:
- PyQt5 import 금지. `matplotlib.use("Agg")` 를 패키지 `__init__` 에서 가장 먼저 호출.
- `plt.close('all')` 금지 → 각 함수는 자기 figure 만 `plt.close(fig)`.
- 전역 `np.random.seed` 금지 → `np.random.default_rng(seed)` 사용.
- 로깅은 `logging.getLogger("particle_core")`, 핫패스에서 debug 수준 이하.
- 한글 경로: 읽기 `np.fromfile + cv2.imdecode`, 쓰기 `cv2.imencode(...)[1].tofile()` (io_utils 로 통일).
- 모든 공개 함수는 타입힌트 + 짧은 docstring.

### 2.1 모듈과 공개 시그니처

```python
# particle_core/params.py — §1 과 1:1 대응하는 dataclass 들 (+ from_dict / to_dict)
ScaleParams, SemParams, WatershedParams, HybridEngineParams, MeasureParams,
AfterFilterParams, SpatialParams, OutputParams, AnalysisParams

# particle_core/io_utils.py
imread_unicode(path, flags=cv2.IMREAD_UNCHANGED) -> np.ndarray | None
imwrite_unicode(path, img, jpeg_quality=95) -> bool
to_gray_u8(img) -> np.ndarray                        # 2D uint8
natural_key(s) -> list
sample_name_of(filename) -> str                      # stem 에서 첫 '-' 앞, 없으면 첫 '_' 앞, 없으면 stem
pixel_to_micro_from_name(filename, default) -> float # 정규식 (\d+(?:\.\d+)?)k → (27/136)/mag
list_images(folder) -> list[Path]                    # .jpg .jpeg .png .tif .tiff .bmp, natural sort, 비재귀
crop_bottom(gray, ratio) -> np.ndarray

# particle_core/preprocess.py
apply_sem_preprocessing(gray_u8, sem: SemParams) -> np.ndarray   # crop 포함, 블러는 포함하지 않음
preview_preprocessing(gray_u8, sem, blur_kernel) -> np.ndarray   # 튜너 미리보기용: 위 + 블러

# particle_core/segment.py
@dataclass
class SegmentationResult:
    labels: np.ndarray        # int32 (H,W), 0=배경
    binary: np.ndarray | None # bool, watershed 만
    distance: np.ndarray | None
    markers: np.ndarray | None
    n_labels: int
segment_watershed(gray_pre: np.ndarray, ws: WatershedParams) -> SegmentationResult
#   = 원본 analyze_particles 429~599 의 엣지강화 → 블러 → Otsu(자동 반전) → 모폴로지 → EDT → 마커 → watershed → remove_small_objects(min_size) → label

# particle_core/engines.py
class SegmentationEngine(Protocol):
    name: str
    def prepare(self) -> None: ...                       # 모델 로드 등 (1회)
    def segment(self, gray_pre: np.ndarray) -> SegmentationResult: ...
class WatershedEngine:  (ws: WatershedParams)
# HybridAIEngine 은 app 측(engines_ai.py)에서 구현 — 이 패키지는 Protocol 과 WatershedEngine 만 제공
def make_engine(params: AnalysisParams) -> SegmentationEngine   # watershed 만 처리, ai_hybrid 면 NotImplementedError

# particle_core/measure.py
measure_particles(labels, gray_for_intensity, pixel_to_micro, mp: MeasureParams)
    -> tuple[pd.DataFrame, dict[int, np.ndarray]]
#   원본 analyze_particles 620~745 의 regionprops 루프 + 원형도 지표 + 평균 밝기 + 곡률 + 필터들.
#   contours_map[label] = (N,1,2) int32 cv2 윤곽선 — 원본에서 비어 있던 버그 수정: 반드시 채운다.
#   DataFrame 컬럼 (원본과 동일 명칭 유지):
#   Label, Area (px^2), Area (μm^2), Perimeter (px), Perimeter (μm), Circularity,
#   Major Axis (px), Major Axis (μm), Minor Axis (px), Minor Axis (μm), Aspect Ratio, Diameter (μm),
#   Area_Ratio, Perimeter_Ratio, Radius_Ratio, x, y, Mean Intensity, Mean Curvature, Solidity,
#   Equivalent_Diameter_Pixel, Image_Width, Image_Height

# particle_core/filters.py
apply_after_filter(df_before: pd.DataFrame, af: AfterFilterParams) -> pd.DataFrame   # 원본 threads 572-591

# particle_core/stats.py
calculate_d_values(df) -> dict   # d10,d50,d90,span,mean,std,mode,cv,n
collect_distribution_statistics(df, feature) -> dict
clean_dataframe(df) -> pd.DataFrame     # threads.py 버전 의미
summary_row(sample, df_before, df_after) -> dict   # Summary 시트 1행

# particle_core/spatial.py
calculate_nearest_neighbor_distances(df, pixel_to_micro) -> pd.DataFrame
calculate_agglomeration_metrics(df) -> dict
visualize_agglomeration(df, pixel_to_micro, save_path) -> None
analyze_clusters_with_dbscan(df, pixel_to_micro, sp: SpatialParams) -> tuple[pd.DataFrame, dict]
visualize_dbscan_clusters(df, save_path) -> None
calculate_boundary_distances(df, contours_map, contact_threshold, pixel_to_micro, labeled_mask) -> tuple[pd.DataFrame, list]
calculate_boundary_metrics(df) -> dict
visualize_boundary_contacts(df, contours_map, contact_pairs, img_shape, save_path, pixel_to_micro, contact_threshold, base_gray, labeled_mask) -> None
#   base_gray 는 labels 와 같은 크기(전처리·크롭된 영상)를 받는다 — 원본의 크기 불일치 버그 수정

# particle_core/viz.py
draw_labeled_image(df, labels, base_gray, contours_map, seed=42) -> np.ndarray(BGR)
draw_contour_only_image(df, labels, base_gray, contours_map, seed=42) -> np.ndarray(BGR)
save_comparison_image(img_before, img_after, path, line_width=5) -> None
save_histogram_comparison(before_values, after_values, feature_name, path) -> None
save_summary_card(sample, stats: dict, path, tag, font_path) -> None     # PIL + arial.ttf (프로젝트 루트 arial.ttf, 없으면 기본 폰트)
save_normal_distribution_plots(excel_path, out_dir) -> None               # matplotlib style 'seaborn-v0_8' fallback 'default'
encode_png_data_url(img_bgr_or_gray) -> str
colorize_labels(labels, seed=7) -> np.ndarray(BGR)

# particle_core/report.py
write_excel(path, sample_sheets: dict[str, pd.DataFrame], stats_sheets: dict[str, pd.DataFrame]) -> None
#   시트 순서: 샘플별 → Summary, Circularity Stats, Aspect Ratio Stats, Perimeter Stats, Area Ratio Stats,
#   Perimeter Ratio Stats, Radius Ratio Stats, Solidity Stats, Agglomeration Stats, DBSCAN Stats, Boundary Stats
#   시트명 31자 제한 + 중복 시 _2, _3 접미
create_summary_ppt(comparison_dir, summary_dir, out_path) -> None
#   원본 create_particle_summary_ppt. 샘플 매칭은 prefix 가 아니라 정확한 샘플명(`{sample}_` 접두 + 구분자) 으로

# particle_core/pipeline.py
@dataclass
class BatchInput:
    images: list[Path]          # 분석 대상 (업로드든 서버 폴더든 여기 모아서 전달)
    output_dir: Path
@dataclass
class ImageResult:              # 이미지 1장 결과 (프런트 갤러리용)
    filename: str; sample: str; pixel_to_micro: float
    n_before: int; n_after: int
    files: dict[str, str]       # 카테고리 → output_dir 기준 상대경로 (labeled_before, labeled_after, contour_before, contour_after, comparison, boundary_before, boundary_after, aggl_before, aggl_after, dbscan_before, dbscan_after)
@dataclass
class BatchResult:
    output_dir: str
    excel_path: str; pptx_path: str | None
    images: list[ImageResult]
    samples: list[dict]         # Summary 시트의 행들 (dict)
    sample_files: dict[str, dict[str, str]]  # sample → {hist_diameter, hist_circularity, hist_aspect, summary_before, summary_after}
    extra_files: dict[str, list[str]]        # normal_dist 등
    warnings: list[str]
def run_batch(inp: BatchInput, params: AnalysisParams, engine: SegmentationEngine,
              progress_cb: Callable[[int, str], None] | None = None,
              cancel_check: Callable[[], bool] | None = None) -> BatchResult
```

### 2.2 run_batch 동작 (원본 ParticleAnalysisThread.run 을 따르되 버그 수정)

이미지 1장당:
1. `imread_unicode` → `to_gray_u8` → `apply_sem_preprocessing` (크롭 1회, 블러 없음) = `gray_pre`
2. `engine.segment(gray_pre)` → labels. watershed 엔진은 내부에서 블러 1회.
3. `measure_particles(labels, gray_pre, p2m, measure)` → `df_before`, `contours_map`
4. `apply_after_filter` → `df_after`. 두 df 에 `ImageName`, `Filter`("Before"/"After") 컬럼 추가
5. 표시용 베이스 영상 = `gray_pre` (labels 와 크기 동일). 출력:
   - `labeled_images/before/{stem}_before_labeled.jpg`, `labeled_images/after/{stem}_after_labeled.jpg`
   - `labeled_images/before_contour_only/…`, `after_contour_only/…` (output.make_contour_only)
   - `comparison_images/{stem}_comparison.jpg`
6. spatial.enable_boundary → `boundary_analysis/{stem}_before_boundary.jpg`, `_after_boundary.jpg` + metrics 행
7. spatial.enable_nnd → `agglomeration_visualization/{stem}_before_agglomeration.png`, `_after_…` + metrics
8. spatial.enable_dbscan → `dbscan_visualization/{stem}_before_dbscan.png`, `_after_…` + stats
9. progress_cb(percent, f"{i}/{n} {filename}")

샘플(파일명 그룹) 단위:
- Before 전체 concat → 샘플 단위로 after_filter 재적용 → 샘플 시트 (Before+After 행)
- `histograms/After_SampleWide/{sample}_{Diameter|Circularity|Aspect_Ratio}_SampleWide_hist_comparison.png`
- `summary_images/{sample}_Before_summary.jpg`, `_After_summary.jpg`
- Summary 행 + 분포 통계 행들, 공간분석 평균 행

배치 끝: `particle_analysis_result.xlsx`, (make_normal_dist) `normal_dist/`, (make_ppt) `particle_analysis_result.pptx`.
예외는 이미지 단위로 잡아서 `warnings` 에 넣고 계속 진행. cancel_check() 가 True 면 지금까지 결과로 Excel 만 쓰고 종료.

## 3. 백엔드 API (FastAPI, 포트 8765)

기존 유지: `GET /api/health`, `GET /api/basic_pt`, `GET /api/basic_pt/{which}`, `POST /api/run/hybrid`.

### 3.1 튜너
```
POST /api/tuner/preprocess   multipart: image(file), params(str, AnalysisParams JSON)
  → { "preview_png": dataURL, "shape": [h,w] }          # sem + watershed.blur_kernel 적용 미리보기

POST /api/tuner/segment      multipart: image(file), params(str)
  → {
      "engine": "watershed",
      "pixel_to_micro": 0.198,
      "shape": [h,w],
      "n_before": int, "n_after": int,
      "views": {                                 # 모두 PNG dataURL
        "original": …, "preprocessed": …, "binary": … | null, "distance": … | null, "markers": … | null,
        "labels": …(colorized), "labeled_before": …, "labeled_after": …, "contour_before": …
      },
      "particles_before": [ {Label, x, y, "Diameter (μm)", Circularity, "Aspect Ratio", "Area (μm^2)", Solidity, "Mean Intensity"} ],
      "particles_after":  [ … ],
      "d_values_before": {d10,d50,d90,span,mean,std,mode,cv,n},
      "d_values_after":  {…},
      "timing_ms": {"preprocess":…, "segment":…, "measure":…}
    }
```
`hybrid.unet_weights`/`yolo_weights` 가 "basic" 이면 서버의 basic_pt 사용. AI 엔진은 서버 프로세스 내에 가중치 경로 기준으로 캐시(LRU 1개).

### 3.2 배치 job
```
POST /api/batch/jobs        multipart:
    params (str, AnalysisParams JSON)
    input_mode (str): "server_path" | "upload"
    input_path (str, server_path 일 때 폴더 절대경로)
    output_dir (str, 선택. 비우면 <project>/runs/<YYYYmmdd_HHMMSS>_<engine>)
    job_name (str, 선택)
    files (file[], upload 일 때. 서버는 runs/<job>/input/ 에 저장)
  → { "job_id": "…", "output_dir": "…", "n_images": int }

GET  /api/batch/jobs                → [ {job_id, name, status, progress, created_at, output_dir, n_images} ]
GET  /api/batch/jobs/{id}           → { job_id, status: queued|running|done|error|cancelled, progress 0-100,
                                        message, log: [str], started_at, finished_at, error: str|null,
                                        result: BatchResult(dict) | null }
POST /api/batch/jobs/{id}/cancel    → {ok}
GET  /api/batch/jobs/{id}/files/{relpath:path}   → 출력 파일 (이미지/xlsx/pptx). output_dir 밖 접근 차단
GET  /api/batch/jobs/{id}/zip       → 출력 폴더 zip (스트리밍)
```
job 은 `jobs.py` 의 레지스트리(dict + threading.Thread, 동시 1개 실행, 나머지 queued). 서버 재시작 시 휘발.

### 3.3 서버 폴더 탐색
```
GET /api/fs/list?path=C:\…   → { "path", "parent", "dirs": [{name, path}], "n_images": int, "images": [name,…(최대 50)] }
GET /api/fs/drives           → ["C:\\", "D:\\"]  (Windows) / ["/"] (그 외)
```

## 4. 프런트 (Next.js 14, App Router, Tailwind)

공통:
- `components/AppShell.tsx`: 상단바 (로고 "Particle Lab", 네비 탭: **튜너** `/`, **배치 분석** `/batch`, **AI 진단** `/hybrid-lab`), 백엔드 상태 점, 기본 모델 상태.
- `lib/params.ts`: `DEFAULT_PARAMS: AnalysisParams`, `useParams()` 훅 — localStorage `plab.params` 에 저장, 튜너와 배치 페이지가 공유. JSON 내보내기/불러오기. 기존 데스크톱 앱의 `particle_analyzer_settings.json` / `{"analysis_params": …}` 형식 import 도 받아서 매핑.
- `lib/api.ts`: `API_BASE` + 위 엔드포인트 래퍼 (`tunerPreprocess`, `tunerSegment`, `createBatchJob`, `getJob`, `listJobs`, `cancelJob`, `jobFileUrl`, `jobZipUrl`, `fsList`, `fsDrives`).
- `lib/types.ts`: AnalysisParams, TunerSegmentResponse, BatchJob, BatchResult 타입.
- 공용 UI: `ImageViewer` (줌/팬/다운로드; hybrid-lab/page.tsx 의 `ImageBox` 를 참고해 독립 컴포넌트로), `NumberField`/`SliderField`/`Toggle`/`Select`, `ParamSection`(접이식), `EngineSwitch`(watershed / AI Hybrid 세그먼트 버튼), `Dropzone`.
- 스타일: 기존 `globals.css`/`tailwind.config.ts` 토큰 유지 (흰 배경, 검정 primary 버튼 `.btn-primary`, `.app-card`). 라이트 전용. 한국어 UI.

### 4.1 튜너 `/`
3컬럼. 좌(320px): 이미지 드롭존(1장) + 엔진 스위치 + 스케일(auto/manual) + "분할 실행" 버튼.
중앙: `ImageViewer` + 뷰 선택 칩 (원본 / 전처리 / 이진화 / 거리맵 / 마커 / 라벨 / Before 라벨 / After 라벨 / 윤곽선). AI 엔진이면 이진화·거리맵·마커 칩은 비활성.
우(가변 폭): 파라미터 패널 — 섹션: SEM 전처리(실시간 미리보기: 값 변경 300ms 디바운스 후 `/api/tuner/preprocess` 호출, 뷰 자동 "전처리"), 분할(엔진별로 watershed 섹션 또는 hybrid 섹션), 특성 필터(measure), After 필터, 스케일.
하단 결과 바: n_before / n_after, D10/D50/D90, 입자 표 (정렬 가능, 최대 500행), "파라미터 JSON 내보내기", "JSON 불러오기", "배치 분석으로 →" (현재 params 저장 후 /batch 이동).

### 4.2 배치 분석 `/batch`
좌: 입력 방식 토글 (서버 폴더 / 업로드). 서버 폴더: 경로 입력 + 폴더 탐색 모달(`/api/fs/*`) + 이미지 수 표시. 업로드: 폴더 드롭/선택 (`<input webkitdirectory multiple>`), 파일 수·용량 표시. 출력 폴더 입력(선택), job 이름. 엔진 스위치 + "분석 시작".
우: 파라미터 패널 (튜너와 같은 컴포넌트, 섹션 전부 + 공간분석 + 출력 옵션).
중앙: 실행 중이면 진행률 바 + 메시지 + 로그; 완료되면
  - 다운로드 바: Excel / PPT / 전체 zip / 출력 폴더 경로 복사
  - 샘플 Summary 표 (Sample, Count_Before, Count_After, D10, D50, D90, Span, Mean, Std, CV)
  - 결과 갤러리: 카테고리 탭 (라벨 Before / After / 비교 / 윤곽선 / 히스토그램 / 요약카드 / 경계 / 응집도 / DBSCAN / 정규분포) × 썸네일 그리드 → 클릭 시 `ImageViewer` 라이트박스. 이미지 URL 은 `jobFileUrl(id, relpath)`.
  - 경고 목록 (warnings)
하단: 최근 job 목록 (`/api/batch/jobs`) — 클릭하면 결과 다시 열기.
폴링: 700ms, status done/error/cancelled 에서 중단.

### 4.3 AI 진단 `/hybrid-lab`
기존 v2 page.tsx 복사본. `AppShell` 만 위에 얹고 내용은 건드리지 않는다.

## 5. 실행/배포
- `START.bat`/`SETUP.bat`/`STOP.bat` 는 v2 의 수정판 유지 (CRLF, CUDA torch 선설치). `app/backend/requirements.txt` 에 `scikit-learn`, `openpyxl`(이미 있음), `python-pptx`(이미 있음) 확인.
- `SETUP.bat` 재실행 조건에 `app/backend/requirements.txt` 변경 감지는 넣지 않음. 대신 `START.bat` 에 `--rebuild` 인자 처리 (venv 외 node 빌드만 다시).
