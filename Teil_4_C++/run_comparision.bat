@echo off
REM ========================================
REM   Model Comparison Script
REM   Compare Biot-Savart vs Dipole
REM ========================================

echo ========================================
echo   Biot-Savart vs Dipole Comparison
echo ========================================
echo.

REM Check if executable exists
if not exist "build\Release\model_comparison.exe" (
    if not exist "build\Debug\model_comparison.exe" (
        echo [ERROR] Executable not found!
        echo Please run build.bat first.
        pause
        exit /b 1
    )
    set EXECUTABLE=build\Debug\model_comparison.exe
) else (
    set EXECUTABLE=build\Release\model_comparison.exe
)

echo Running comparison...
echo.

%EXECUTABLE%

if %errorlevel%==0 (
    echo.
    echo ========================================
    echo   Comparison Complete!
    echo ========================================
    echo.
    echo Results saved to:
    echo   - results/biot_savart_results.csv
    echo   - results/dipole_results.csv
    echo   - results/model_comparison_summary.txt
    echo.
) else (
    echo.
    echo [ERROR] Comparison failed!
    echo Error code: %errorlevel%
)

pause