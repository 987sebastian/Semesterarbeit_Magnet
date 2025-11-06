@echo off
chcp 65001 >nul
REM ========================================
REM   Model Theoretical Accuracy Test
REM   Test Biot-Savart vs Dipole WITHOUT calibration
REM ========================================

echo ========================================
echo   Model Theoretical Accuracy Test
echo   Biot-Savart vs Dipole (No Calibration)
echo ========================================
echo.

REM Check if executable exists
if not exist "build\Release\test_model_accuracy.exe" (
    if not exist "build\Debug\test_model_accuracy.exe" (
        echo [ERROR] Executable not found!
        echo Please run build.bat first.
        pause
        exit /b 1
    )
    set EXECUTABLE=build\Debug\test_model_accuracy.exe
) else (
    set EXECUTABLE=build\Release\test_model_accuracy.exe
)

echo Running theoretical accuracy test...
echo This will compare the two models without any calibration matrix.
echo.

%EXECUTABLE%

if %errorlevel%==0 (
    echo.
    echo ========================================
    echo   Test Complete!
    echo ========================================
    echo.
    echo This test shows the INHERENT difference between models.
    echo The results help explain why Dipole performs better
    echo with current calibration matrix.
    echo.
) else (
    echo.
    echo [ERROR] Test failed!
    echo Error code: %errorlevel%
)

pause