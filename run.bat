@echo off
echo ============================================
echo   Biot-Savart Pose Optimization
echo ============================================
echo.

if not exist build\Release\biot_tracking.exe (
    echo [ERROR] biot_tracking.exe not found!
    echo Please run build.bat first.
    pause
    exit /b 1
)

echo Running Biot-Savart optimizer...
echo.

REM --- OpenMP P-Core Pinning (12 Threads) ---
set OMP_NUM_THREADS=12
set OMP_PROC_BIND=close
set OMP_PLACES=cores(6)

build\Release\biot_tracking.exe

REM --- Clear OpenMP Environment Variables ---
set OMP_NUM_THREADS=
set OMP_PROC_BIND=
set OMP_PLACES=

echo.
echo ============================================
echo   Program finished
echo ============================================
echo.
echo Results saved to:
echo   - results\opt_summary.csv
echo   - results\pose_error_summary.csv
echo.
pause