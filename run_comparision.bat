@echo off
REM Run model comparison
REM 运行模型对比

echo ================================================
echo   Running Model Comparison
echo   运行模型对比
echo ================================================
echo.

cd /d "%~dp0"

REM Check if executable exists
if exist build\Release\model_comparison.exe (
    echo Running: build\Release\model_comparison.exe
    echo.
    build\Release\model_comparison.exe
    set EXEC_FOUND=1
) else if exist build\model_comparison.exe (
    echo Running: build\model_comparison.exe
    echo.
    build\model_comparison.exe
    set EXEC_FOUND=1
) else (
    echo [ERROR] Executable not found!
    echo 找不到可执行文件！
    echo.
    echo Please build the project first:
    echo 请先编译项目：
    echo   ^> build.bat
    echo.
    pause
    exit /b 1
)

if %EXEC_FOUND%==1 (
    echo.
    echo ================================================
    echo Comparison complete! / 对比完成！
    echo ================================================
    echo.
    echo Results saved to: / 结果已保存到：
    echo   - results\biot_savart_results.csv
    echo   - results\dipole_results.csv
    echo.
    echo To visualize results, run:
    echo 要可视化结果，请运行：
    echo   ^> python visualize_comparison.py
    echo.
)

pause