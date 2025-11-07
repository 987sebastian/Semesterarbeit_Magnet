@echo off
chcp 65001 >nul
REM Build script for Windows

echo ================================================
echo   Biot-Savart Tracking Build Script
echo ================================================
echo.

REM Check if build directory exists
if not exist build (
    echo Creating build directory...
    mkdir build
) else (
    echo Build directory already exists
)

echo.
echo ================================================
echo Running CMake...
echo ================================================
cd build
cmake ..

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] CMake configuration failed!
    echo.
    echo Possible issues:
    echo   1. CMake not installed
    echo   2. No C++ compiler found
    echo   3. Missing Eigen library
    echo.
    pause
    exit /b 1
)

echo.
echo ================================================
echo Building project...
echo ================================================
cmake --build . --config Release

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Build failed!
    echo.
    echo Please check the error messages above.
    echo.
    pause
    exit /b 1
)

echo.
echo ================================================
echo Build completed successfully!
echo ================================================
echo.
echo Executable locations:
echo   - build\Release\biot_tracking.exe
echo   - build\Release\model_comparison.exe
echo   - build\Release\test_model_accuracy.exe
echo.
echo To run programs:
echo   1. run.bat                   (Biot-Savart tracking)
echo   2. run_comparision.bat       (Model comparison)
echo   3. run_test_accuracy.bat     (Theoretical test)
echo.

cd ..
pause