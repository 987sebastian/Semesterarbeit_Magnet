@echo off
REM Build script for Windows / Windows编译脚本

echo ================================================
echo   Biot-Savart Tracking Build Script
echo   位姿追踪C++项目编译脚本
echo ================================================
echo.

REM Check if build directory exists / 检查编译目录
if not exist build (
    echo Creating build directory... / 创建编译目录...
    mkdir build
) else (
    echo Build directory already exists / 编译目录已存在
)

echo.
echo ================================================
echo Running CMake... / 运行CMake配置...
echo ================================================
cd build
cmake ..

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] CMake configuration failed! / CMake配置失败！
    echo.
    echo Possible issues / 可能的问题:
    echo   1. CMake not installed / CMake未安装
    echo   2. No C++ compiler found / 找不到C++编译器
    echo   3. Missing Eigen library / 缺少Eigen库
    echo.
    pause
    exit /b 1
)

echo.
echo ================================================
echo Building project... / 编译项目...
echo ================================================
cmake --build . --config Release

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Build failed! / 编译失败！
    echo.
    echo Please check the error messages above.
    echo 请检查上面的错误信息。
    echo.
    pause
    exit /b 1
)

echo.
echo ================================================
echo Build completed successfully! / 编译成功完成！
echo ================================================
echo.
echo Executable location / 可执行文件位置:
echo   build\Release\biot_tracking.exe
echo.
echo To run the program / 运行程序:
echo   1. Direct run / 直接运行: run.bat
echo   2. Command line / 命令行: build\Release\biot_tracking.exe
echo.

cd ..
pause