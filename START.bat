@echo off
REM Particle Lab v2 - Windows launcher (double-click)
chcp 65001 >nul
setlocal EnableExtensions EnableDelayedExpansion

REM 스크립트 위치로 이동
cd /d "%~dp0"

echo.
echo ===========================================
echo    ParticleLab - 자동 실행 중...
echo ===========================================
echo.

REM Python / Node 확인
where python >nul 2>&1
if errorlevel 1 (
  echo [에러] Python 이 설치되어 있지 않습니다.
  echo        설치: https://www.python.org/downloads/
  pause
  exit /b 1
)
where node >nul 2>&1
if errorlevel 1 (
  echo [에러] Node.js 가 설치되어 있지 않습니다.
  echo        설치: https://nodejs.org/
  pause
  exit /b 1
)

REM 첫 실행이면 setup
set NEED_SETUP=0
if not exist "venv" set NEED_SETUP=1
if not exist "app\frontend\node_modules" set NEED_SETUP=1
if not exist "app\frontend\.next" set NEED_SETUP=1

if "%NEED_SETUP%"=="1" (
  echo [정보] 처음 실행 - 환경 설정 ^(약 5분^)
  call "%~dp0SETUP.bat"
  if errorlevel 1 (
    echo 설정 실패.
    pause
    exit /b 1
  )
)

REM venv 활성화
call "venv\Scripts\activate.bat"

REM 기존 프로세스 정리 (8765 / 3001)
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8765 " ^| findstr LISTENING') do (
  taskkill /F /PID %%a >nul 2>&1
)
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":3001 " ^| findstr LISTENING') do (
  taskkill /F /PID %%a >nul 2>&1
)

if not exist "logs" mkdir "logs"

REM 백엔드 시작 (별도 창)
echo [시작] 백엔드 (포트 8765)
set PYTHONPATH=%CD%;%CD%\app\backend
start "Particle Lab — Backend" /MIN cmd /c "cd app\backend && python -m uvicorn app.main:app --host 127.0.0.1 --port 8765 > ..\..\logs\backend.log 2>&1"

REM 백엔드 준비 대기 (최대 60초)
echo   대기 중...
set /a count=0
:wait_backend
timeout /t 1 /nobreak >nul
curl -s -o nul http://127.0.0.1:8765/api/health 2>nul
if not errorlevel 1 goto backend_ready
set /a count+=1
if !count! lss 60 goto wait_backend
echo [에러] 백엔드 시작 실패. logs\backend.log 확인.
pause
exit /b 1
:backend_ready

REM 프런트엔드 시작 (별도 창)
echo [시작] 프런트엔드 (포트 3001)
start "Particle Lab — Frontend" /MIN cmd /c "cd app\frontend && npm run start -- -p 3001 > ..\..\logs\frontend.log 2>&1"

REM 프런트 준비 대기 (최대 30초)
echo   대기 중...
set /a count=0
:wait_frontend
timeout /t 1 /nobreak >nul
curl -s -o nul http://127.0.0.1:3001 2>nul
if not errorlevel 1 goto frontend_ready
set /a count+=1
if !count! lss 30 goto wait_frontend
echo [에러] 프런트엔드 시작 실패. logs\frontend.log 확인.
pause
exit /b 1
:frontend_ready

REM Chrome 으로 열기 (없으면 기본 브라우저)
set CHROME=""
if exist "%ProgramFiles%\Google\Chrome\Application\chrome.exe" set CHROME="%ProgramFiles%\Google\Chrome\Application\chrome.exe"
if exist "%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe" set CHROME="%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"
if not %CHROME%=="" (
  start "" %CHROME% "http://localhost:3001"
) else (
  start "" "http://localhost:3001"
)

echo.
echo ===========================================
echo    ParticleLab 실행 중
echo    접속: http://localhost:3001
echo ===========================================
echo.
echo  이 창을 닫거나 STOP.bat 을 실행하면
echo  서버가 종료됩니다.
echo.
echo  로그: logs\backend.log  logs\frontend.log
echo.
pause
