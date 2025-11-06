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

build\Release\biot_tracking.exe

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