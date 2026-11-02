# ParticleLab — SEM 입자 분석 (웹)

기존 두 프로그램을 하나로 합친 웹앱입니다.

| 기능 | 출처 |
|---|---|
| 배치 분석 (폴더 → 라벨 이미지 · 히스토그램 · 요약카드 · 경계/응집도/DBSCAN · Excel 11시트 · PPT) | `particle_analyzer` (PyQt 데스크톱) |
| 파라미터 튜너 (단일 이미지, SEM 전처리 실시간 미리보기, 분할 미리보기, 파라미터 JSON) | `particle_parameter_tuner` |
| **AI Hybrid 분할 엔진** (U-Net × YOLO) — 위 두 기능에서 watershed 대신 선택 가능 | `LIB-Particle_Lab_v2` |
| AI 모델 진단 콘솔 (정답 마스크 대비 7지표, 12개 진단 뷰) | `LIB-Particle_Lab_v2` (그대로 포함, `/hybrid-lab`) |

분할(segmentation) 단계만 **엔진**으로 교체되고, 그 뒤의 특성 추출 → Before/After 필터 → 통계 → Excel/PPT 는 두 엔진이 공유합니다.

## 실행 (Windows)

사전 설치: Python 3.10+ (PATH 등록), Node.js 18+.

1. `START.bat` 더블클릭. 처음이면 `SETUP.bat` 이 자동 실행되어 venv 생성, PyTorch(NVIDIA GPU 있으면 CUDA 12.1), 패키지 설치, 프런트 빌드까지 약 5~10분.
2. 브라우저가 `http://localhost:3001` 로 열립니다.
3. 종료: `STOP.bat`.

수동 실행:

```powershell
# 백엔드 (포트 8765)
.\venv\Scripts\Activate.ps1
$env:PYTHONPATH = "$PWD;$PWD\app\backend"
cd app\backend; python -m uvicorn app.main:app --host 127.0.0.1 --port 8765

# 프런트 (포트 3001)
cd app\frontend; npm run dev
```

코드를 수정한 뒤에는 `SETUP.bat` 을 다시 실행해 패키지와 프런트 빌드를 갱신하세요.

## 화면

- **튜너** `/` — 이미지 1장을 올리고 엔진(Watershed / AI Hybrid)과 파라미터를 조정하며 전처리·이진화·거리맵·마커·라벨 결과를 바로 확인합니다. 만족스러운 파라미터는 JSON 으로 저장하거나 "배치 분석으로" 넘깁니다.
- **배치 분석** `/batch` — 서버 폴더 경로를 입력(또는 폴더 업로드)하고 실행하면 진행률이 표시되고, 끝나면 Excel / PPT / zip 다운로드와 결과 갤러리가 나옵니다. 결과는 기본으로 `runs/<시각>_<이름>/` 에 저장됩니다.
- **AI 진단** `/hybrid-lab` — 기존 Particle Lab v2 콘솔.

## 폴더 구조

```
particle_core/        분석 코어 (헤드리스). params / io_utils / preprocess / segment / engines /
                      measure / filters / stats / spatial / viz / report / pipeline
hybrid_core.py, unet_*.py, evaluate.py   AI 코어
basic_pt/             기본 가중치 (U-net++, yolo_11 — 확장자 없음)
app/backend/app/      FastAPI: main.py, jobs.py, engines_ai.py, api/{tuner,batch,fs,hybrid}.py
app/frontend/         Next.js 14
runs/                 배치 결과 (자동 생성)
DESIGN.md             설계 명세 (파라미터 스키마, API, 모듈 계약)
```

## 원본 대비 수정된 동작

데스크톱 버전의 다음 문제를 고쳤습니다. 수치가 원본과 미세하게 다를 수 있습니다.

- 하단 크롭과 Gaussian 블러가 두 번 적용되던 것을 한 번으로 수정. 경계 분석 이미지가 크기 불일치로 저장되지 않던 문제도 함께 해결.
- 윤곽선 맵이 비어 있어 Boundary Stats 가 항상 0이고 contour-only 이미지가 빈 배경이던 문제 수정.
- `min_size`(작은 객체 제거 임계)가 UI 값과 무관하게 30 으로 고정되던 문제 수정.
- PPT 생성 시 샘플 `S1` 이 `S10` 이미지까지 끌어오던 접두 매칭 수정.
- 최신 matplotlib 에서 `seaborn` 스타일 오류로 정규분포 그림이 생성되지 않던 문제 수정.

## API

`http://localhost:8765/docs` 에서 전체 스키마를 볼 수 있습니다. 주요 엔드포인트와 파라미터 스키마는 `DESIGN.md` 참고.
