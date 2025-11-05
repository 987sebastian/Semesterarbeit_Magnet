@echo off
REM Run script for Windows / Windows运行脚本

echo ================================================
echo   Running Biot-Savart Pose Optimization
echo   运行位姿优化程序
echo ================================================
echo.

REM Change to script directory / 切换到脚本所在目录
cd /d "%~dp0"

REM Check if executable exists in expected location / 检查可执行文件存在性
if exist build\Release\biot_tracking.exe (
    echo Running: build\Release\biot_tracking.exe
    echo.
    build\Release\biot_tracking.exe
    goto :finish
)

REM Try Debug build if Release not found / 如果Release版本不存在，尝试Debug版本
if exist build\Debug\biot_tracking.exe (
    echo Running: build\Debug\biot_tracking.exe
    echo.
    build\Debug\biot_tracking.exe
    goto :finish
)

REM Error handling / 错误处理
echo.
echo [ERROR] Executable not found! / 找不到可执行文件！
echo.
echo Please build the project first by running:
echo 请先运行以下命令编译项目:
echo   build.bat
echo.
echo Expected paths / 期望的路径:
echo   - build\Release\biot_tracking.exe (Release build)
echo   - build\Debug\biot_tracking.exe (Debug build)
echo.
echo Or in Visual Studio / 或在 Visual Studio 中:
echo   - Build -^> Build Solution
echo   - 生成 -^> 生成解决方案
echo.
pause
exit /b 1

:finish
echo.
echo ================================================
echo Program finished / 程序结束
echo ================================================
echo.
echo Results output location / 结果输出位置:
echo   - results\opt_summary.csv (Position and direction estimates)
echo   - results\pose_error_summary.csv (Error analysis)
echo.
pause