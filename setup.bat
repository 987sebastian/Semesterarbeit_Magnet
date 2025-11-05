@echo off
REM Quick setup script for model comparison
REM 模型对比快速安装脚本

echo ================================================
echo   Model Comparison Setup
echo   模型对比环境配置
echo ================================================
echo.

REM Get project root directory
set PROJECT_ROOT=%~dp0
cd /d "%PROJECT_ROOT%"

echo [1/4] Checking directory structure...
echo 检查目录结构...

REM Create necessary directories
if not exist "results" mkdir results
if not exist "data" mkdir data

echo   OK: Directories ready
echo.

echo [2/4] Copying comparison files...
echo 复制对比文件...

REM List of files to check
set FILES=dipole_field.hpp dipole_field.cpp pose_optimizer_generic.hpp pose_optimizer_generic.cpp model_adapters.hpp model_adapters.cpp main_comparison.cpp

set ALL_PRESENT=1
for %%F in (%FILES%) do (
    if not exist "%%F" (
        echo   MISSING: %%F
        set ALL_PRESENT=0
    ) else (
        echo   OK: %%F
    )
)

if %ALL_PRESENT%==0 (
    echo.
    echo [ERROR] Some comparison files are missing!
    echo Please copy all files from the outputs folder to the project root.
    echo.
    pause
    exit /b 1
)

echo   OK: All comparison files present
echo.

echo [3/4] Checking data files...
echo 检查数据文件...

if not exist "data\current_calibration_Biot.json" (
    echo   WARNING: data\current_calibration_Biot.json not found
    echo   警告：校准文件未找到
)

if not exist "data\Observational_data.csv" (
    echo   WARNING: data\Observational_data.csv not found  
    echo   警告：观测数据文件未找到
)

echo.

echo [4/4] Checking dependencies...
echo 检查依赖...

REM Check for CMake
where cmake >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo   WARNING: CMake not found in PATH
    echo   警告：未找到CMake
) else (
    echo   OK: CMake found
)

REM Check for Visual Studio
where cl >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo   WARNING: Visual Studio compiler not found
    echo   警告：未找到Visual Studio编译器
    echo   Please run this script from "x64 Native Tools Command Prompt"
    echo   请从"x64 Native Tools Command Prompt"运行此脚本
) else (
    echo   OK: Visual Studio compiler found
)

REM Check for Python
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo   WARNING: Python not found
    echo   警告：未找到Python（可视化需要）
) else (
    echo   OK: Python found
)

echo.
echo ================================================
echo Setup complete! / 配置完成！
echo ================================================
echo.
echo Next steps / 后续步骤:
echo   1. Review CMakeLists.txt and config.hpp
echo      检查CMakeLists.txt和config.hpp
echo.
echo   2. Build the project:
echo      编译项目：
echo      ^> build.bat
echo.
echo   3. Run model comparison:
echo      运行模型对比：
echo      ^> run_comparison.bat
echo.
echo   4. Visualize results:
echo      可视化结果：
echo      ^> python visualize_comparison.py
echo.
pause