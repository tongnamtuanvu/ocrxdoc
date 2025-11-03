@echo off
echo ============================================
echo Build ung dung thanh file .exe
echo ============================================
echo.

REM Kiem tra Python
python --version >nul 2>&1
if errorlevel 1 (
    echo LOI: Python chua duoc cai dat hoac chua co trong PATH!
    pause
    exit /b 1
)

echo Dang chay build script...
python build_exe.py

if errorlevel 1 (
    echo.
    echo LOI: Build that bai!
    pause
    exit /b 1
)

echo.
echo ============================================
echo Build thanh cong!
echo ============================================
echo.
echo File .exe duoc tao tai: dist\OCR_App.exe
echo.
echo LUU Y:
echo 1. Copy thu muc 'models' vao cung thu muc voi .exe (neu can)
echo 2. Thu muc 'history' se duoc tao tu dong khi chay
echo.
pause

