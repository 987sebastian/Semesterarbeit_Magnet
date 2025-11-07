@echo off
chcp 65001 >nul
REM ========================================
REM   Multi-Core Performance Benchmark
REM ========================================

echo ========================================
echo   Multi-Core Performance Benchmark
echo   Biot-Savart Magnetic Tracking
echo ========================================
echo.

REM Check if executable exists
if not exist "build\Release\benchmark_parallel.exe" (
    if not exist "build\Debug\benchmark_parallel.exe" (
        echo [ERROR] Executable not found!
        echo Please run build.bat first.
        pause
        exit /b 1
    )
    set EXECUTABLE=build\Debug\benchmark_parallel.exe
) else (
    set EXECUTABLE=build\Release\benchmark_parallel.exe
)

echo Running multi-core benchmark...
echo This will test performance from 1 to 16 cores.
echo.
echo NOTE: This may take several minutes!
echo.

%EXECUTABLE%

if %errorlevel%==0 (
    echo.
    echo ========================================
    echo   Benchmark Complete!
    echo ========================================
    echo.
    echo Results saved to:
    echo   - results/benchmark_parallel.csv
    echo.
    echo To visualize results, run:
    echo   python visualize_benchmark.py
    echo.
) else (
    echo.
    echo [ERROR] Benchmark failed!
    echo Error code: %errorlevel%
)

pause