#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script để build ứng dụng thành file .exe
"""

import subprocess
import sys
import os

def check_pyinstaller():
    """Kiểm tra PyInstaller đã được cài đặt chưa"""
    try:
        import PyInstaller
        print("✓ PyInstaller đã được cài đặt")
        return True
    except ImportError:
        print("✗ PyInstaller chưa được cài đặt")
        print("Đang cài đặt PyInstaller...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
            print("✓ PyInstaller đã được cài đặt thành công")
            return True
        except subprocess.CalledProcessError:
            print("✗ Không thể cài đặt PyInstaller")
            return False

def build_exe():
    """Build ứng dụng thành file .exe"""
    print("=" * 60)
    print("Build ứng dụng thành file .exe")
    print("=" * 60)
    
    # Kiểm tra PyInstaller
    if not check_pyinstaller():
        return False
    
    # Kiểm tra models directory
    if not os.path.exists("models"):
        print("⚠ Cảnh báo: Thư mục 'models' không tồn tại!")
        print("Ứng dụng sẽ cần model để hoạt động.")
        response = input("Tiếp tục build? (y/n): ")
        if response.lower() != 'y':
            return False
    
    # PyInstaller command
    cmd = [
        "pyinstaller",
        "--name=OCR_App",
        "--onefile",
        "--windowed",
        "--noconsole",
        "--icon=NONE",
        "--add-data=models;models",
        "--hidden-import=torch",
        "--hidden-import=transformers",
        "--hidden-import=PIL",
        "--hidden-import=PyQt6.QtCore",
        "--hidden-import=PyQt6.QtGui",
        "--hidden-import=PyQt6.QtWidgets",
        "--hidden-import=pdf2image",
        "--hidden-import=docx",
        "--hidden-import=sqlite3",
        "--exclude-module=matplotlib",
        "--exclude-module=numpy",
        "--exclude-module=scipy",
        "--exclude-module=pandas",
        "--exclude-module=jupyter",
        "--exclude-module=IPython",
        "--collect-all=PyQt6",
        "main.py"
    ]
    
    print("\nĐang build...")
    print("Command:", " ".join(cmd))
    print("\nLưu ý:")
    print("- Quá trình này có thể mất 5-15 phút tùy vào tốc độ máy")
    print("- File .exe sẽ có kích thước lớn (~500MB - 1GB) do chứa PyTorch")
    print("- Nếu build lỗi, thử build lại hoặc kiểm tra dependencies\n")
    
    try:
        subprocess.check_call(cmd)
        print("\n" + "=" * 60)
        print("✓ Build thành công!")
        print("=" * 60)
        print(f"\nFile .exe được tạo tại: dist\\OCR_App.exe")
        print("\n" + "=" * 60)
        print("HƯỚNG DẪN SỬ DỤNG:")
        print("=" * 60)
        print("1. File .exe đã bao gồm tất cả dependencies")
        print("2. Thư mục 'models' đã được include trong .exe")
        print("3. Khi chạy lần đầu, ứng dụng sẽ tạo thư mục 'history' để lưu kết quả")
        print("4. Copy file .exe ra bất kỳ đâu và chạy trực tiếp")
        print("\n" + "=" * 60)
        return True
    except subprocess.CalledProcessError as e:
        print("\n" + "=" * 60)
        print("✗ Build thất bại!")
        print("=" * 60)
        print(f"Lỗi: {e}")
        print("\nThử các giải pháp sau:")
        print("1. Chạy lại với quyền Administrator")
        print("2. Kiểm tra antivirus có block không")
        print("3. Xóa thư mục 'build' và 'dist' rồi build lại")
        return False

if __name__ == "__main__":
    success = build_exe()
    if not success:
        input("\nNhấn Enter để thoát...")
    sys.exit(0 if success else 1)
