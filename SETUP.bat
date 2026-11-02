@echo off
chcp 65001 >nul
setlocal

cd /d "%~dp0"

echo [설정] Python 가상환경 생성
if not exist "venv" (
  python -m venv venv
  if errorlevel 1 (
    echo Python venv 생성 실패.
    exit /b 1
  )
)
call "venv\Scripts\activate.bat"

echo [설정] pip 업그레이드
python -m pip install --upgrade pip wheel --quiet

REM torch 를 requirements 보다 먼저 설치 — ultralytics 가 PyPI 의 CPU 전용 torch 를 끌어오는 것을 방지.
REM NVIDIA GPU 가 있으면 CUDA 12.1 빌드, 없으면 CPU 빌드.
where nvidia-smi >nul 2>&1
if errorlevel 1 (
  echo [설정] PyTorch 설치 ^(CPU^)
  pip install --quiet torch torchvision
) else (
  echo [설정] PyTorch 설치 ^(CUDA 12.1, 약 2.5GB^)
  pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu121
)
if errorlevel 1 exit /b 1

echo [설정] 백엔드 Python 패키지 설치 (약 2분)
pip install --quiet -r app\backend\requirements.txt
if errorlevel 1 exit /b 1

echo [설정] Node 패키지 설치
pushd app\frontend
if exist "package-lock.json" (
  call npm ci --silent --no-audit --no-fund
  if errorlevel 1 call npm install --silent --no-audit --no-fund
) else (
  call npm install --silent --no-audit --no-fund
)
if errorlevel 1 (
  popd
  exit /b 1
)

echo [설정] 프런트엔드 빌드 (약 30초)
call npm run build
if errorlevel 1 (
  popd
  exit /b 1
)
popd

echo.
echo [완료] 설정 끝.
exit /b 0
