# Hướng Dẫn Build Ứng Dụng OCR Thành File .exe

## ⚡ Cách Nhanh Nhất

**Double-click file `build.bat`** hoặc chạy:
```bash
python build_exe.py
```

## 📋 Yêu Cầu Trước Khi Build

1. ✅ Python 3.8+ đã cài đặt
2. ✅ Tất cả dependencies đã cài: `pip install -r requirements.txt`
3. ✅ Models đã tải về (thư mục `models/` phải tồn tại)

## 🔧 Chi Tiết Build

### Bước 1: Kiểm Tra Dependencies

```bash
pip install -r requirements.txt
pip install pyinstaller
```

### Bước 2: Chạy Build

**Cách 1 (Khuyến nghị):**
```bash
python build_exe.py
```

**Cách 2:**
```bash
python build.bat
```

**Cách 3 (Thủ công):**
```bash
pyinstaller --name=OCR_App --onefile --windowed --noconsole --add-data=models;models --hidden-import=torch --hidden-import=transformers --hidden-import=PIL --hidden-import=PyQt6 main.py
```

## 📦 Kết Quả

Sau khi build thành công:
- File `.exe` sẽ ở: **`dist/OCR_App.exe`**
- File này đã bao gồm TẤT CẢ dependencies (PyTorch, transformers, PyQt6, etc.)
- Thư mục `models/` đã được include trong .exe

## ⚠️ Lưu Ý Quan Trọng

### Kích Thước File
- File `.exe` sẽ **RẤT LỚN** (~500MB - 1.5GB) do chứa PyTorch
- Đây là **BÌNH THƯỜNG**, không phải lỗi

### Thời Gian Build
- Lần đầu: **5-15 phút** (tùy máy)
- Các lần sau: **Nhanh hơn** (do cache)

### Models
- Thư mục `models/` **PHẢI** tồn tại trước khi build
- Models sẽ được include vào .exe
- Người dùng không cần tải model riêng

## 🚀 Phân Phối

1. Copy file `dist/OCR_App.exe` ra nơi bạn muốn
2. File này **ĐỘC LẬP**, có thể chạy trên máy Windows khác:
   - ✅ Không cần cài Python
   - ✅ Không cần cài dependencies
   - ✅ Chỉ cần đủ RAM/VRAM

## 🐛 Xử Lý Lỗi

### Lỗi: "PyInstaller not found"
```bash
pip install pyinstaller
```

### Lỗi: "Module not found"
```bash
pip install -r requirements.txt
```

### Lỗi: "Models directory not found"
- Đảm bảo thư mục `models/` tồn tại
- Đảm bảo đã tải cả 2 model (4B và 2B)

### Build Bị Dừng / Crash
- Chạy lại với quyền **Administrator**
- Tắt antivirus tạm thời
- Xóa thư mục `build/` và `dist/` rồi build lại

### File .exe Không Chạy Được
- Kiểm tra Windows Defender / Antivirus có block không
- Chạy với quyền Administrator
- Kiểm tra log trong thư mục `history/`

## 📝 Build Với Console (Để Debug)

Nếu muốn xem console output khi chạy .exe:

Sửa file `build_exe.py`, thay `--noconsole` thành `--console`:
```python
"--console",  # Thay vì --noconsole
```

## 🎯 Tips Tối Ưu

1. **Build trên máy có đủ RAM** (16GB+)
2. **Tắt các ứng dụng khác** khi build để tránh crash
3. **Dùng SSD** để build nhanh hơn
4. **Kiểm tra disk space** trước khi build (cần ~5-10GB trống)

## 📞 Hỗ Trợ

Nếu gặp lỗi, kiểm tra:
1. Log trong terminal khi build
2. File `ocr_history.db` trong thư mục chạy .exe
3. Thư mục `history/` có được tạo không

