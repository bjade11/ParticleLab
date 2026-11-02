@echo off
chcp 65001 >nul
echo [정리] Particle Lab 프로세스 종료

for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8765 " ^| findstr LISTENING') do (
  taskkill /F /PID %%a >nul 2>&1
)
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":3001 " ^| findstr LISTENING') do (
  taskkill /F /PID %%a >nul 2>&1
)
echo [완료]
timeout /t 1 /nobreak >nul
