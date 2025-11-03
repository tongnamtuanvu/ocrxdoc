# Ứng Dụng OCR với AI

Ứng dụng OCR (Optical Character Recognition) sử dụng AI để nhận dạng và trích xuất văn bản từ hình ảnh và các file (JPG, PNG, JPEG, PDF, DOCX, TXT).

## ✨ Tính Năng

- 📷 **OCR Hình Ảnh**: Hỗ trợ JPG, PNG, JPEG
- 📄 **OCR File**: Hỗ trợ PDF, DOCX, TXT
- 🤖 **2 Model AI**: 
  - Model OCR mặc định (4B) - Chính xác hơn
  - Model OCR nhẹ (2B) - Nhanh hơn
- 🖥️ **GPU/CPU**: Tự động phát hiện và sử dụng GPU nếu có
- 🎯 **ROI Selection**: Chọn vùng tùy chỉnh để OCR
- 📦 **Batch Processing**: Xử lý nhiều file cùng lúc (không giới hạn)
- 📚 **Lịch Sử**: Lưu trữ và quản lý kết quả OCR với SQLite
- ✏️ **CRUD History**: Chỉnh sửa, xóa kết quả OCR
- ⚡ **Tự Động**: Tự động load model khi khởi động (tùy chọn)

## 📋 Yêu Cầu Hệ Thống

### Phần Mềm
- Python 3.8 hoặc cao hơn
- Windows 10/11
- RAM: Tối thiểu 16GB (khuyến nghị 32GB+)
- GPU: Khuyến nghị (NVIDIA với CUDA support) - VRAM tối thiểu 8GB

### Dependencies
Tất cả dependencies được liệt kê trong `requirements.txt`

## 🚀 Cài Đặt

### Bước 1: Clone Repository

```bash
git clone https://github.com/tongnamtuanvu/ocr-app.git
cd ocr-app
```

### Bước 2: Cài Đặt Dependencies

```bash
pip install -r requirements.txt
```

### Bước 3: Tải Models

Models cần được tải về thủ công do kích thước lớn:

1. **Model OCR mặc định (4B)**:
   - Tải từ: [Hugging Face - Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)
   - Đặt vào thư mục: `./models/Qwen3-VL-4B-Instruct/`

2. **Model OCR nhẹ (2B)**:
   - Tải từ: [Hugging Face - Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)
   - Đặt vào thư mục: `./models/Qwen3-VL-2B-Instruct/`

### Bước 4: Cài Đặt Poppler (Cho PDF)

Tải Poppler từ [poppler-windows](https://github.com/oschwartz10612/poppler-windows/releases/) và giải nén vào thư mục `./poppler/`.

Hoặc sử dụng pip:
```bash
pip install pdf2image
```

## 🎮 Sử Dụng

### Chạy Ứng Dụng

```bash
python main.py
```

### Giao Diện

1. **Tab OCR**:
   - Chọn model (OCR mặc định hoặc OCR nhẹ)
   - Chọn thiết bị (GPU hoặc CPU)
   - Chọn file/hình ảnh để OCR
   - Nhấn "Sử dụng AI này" để load model (nếu chưa load)
   - Nhấn "Xử Lý" để bắt đầu OCR
   - Xem kết quả trong ô kết quả

2. **Tab Lịch Sử**:
   - Xem tất cả kết quả OCR đã lưu
   - Chỉnh sửa kết quả
   - Xóa kết quả (đơn hoặc nhiều)
   - Xem preview hình ảnh

### Tính Năng ROI

- Tích vào checkbox "Chọn vùng tùy chỉnh (ROI)"
- Click và kéo trên hình ảnh để chọn vùng cần OCR
- Chỉ vùng được chọn sẽ được xử lý

### Batch Processing

1. Nhấn "Chọn Nhiều File"
2. Chọn nhiều file cùng lúc
3. Ứng dụng sẽ xử lý tuần tự từng file
4. Kết quả được lưu tự động vào Lịch Sử

## 📦 Build Thành File .exe

### Cách 1: Sử Dụng Script (Khuyến Nghị)

```bash
python build_exe.py
```

Hoặc double-click file `build.bat` (sau khi tạo)

### Cách 2: Build Thủ Công

1. Cài đặt PyInstaller:
```bash
pip install pyinstaller
```

2. Chạy build:
```bash
pyinstaller --name=OCR_App --onefile --windowed --noconsole --add-data=models;models --hidden-import=torch --hidden-import=transformers --hidden-import=PIL --hidden-import=PyQt6 main.py
```

### Kết Quả

File `.exe` sẽ được tạo tại: `dist/OCR_App.exe`

### Lưu Ý Khi Build

1. **Kích thước file**: File `.exe` sẽ RẤT LỚN (~500MB - 1.5GB) do chứa PyTorch - đây là BÌNH THƯỜNG
2. **Thời gian build**: Lần đầu có thể mất 5-15 phút
3. **Models**: Thư mục `models/` PHẢI tồn tại trước khi build
4. **Phân phối**: File `.exe` độc lập, có thể chạy trên máy Windows khác mà không cần cài Python

### Chi Tiết Build

Xem file `BUILD_GUIDE.md` (nếu có) để biết thêm chi tiết về build và xử lý lỗi.

## 🔧 Cấu Hình

### Tham Số OCR

- **Max Tokens**: 3000 (mặc định)
- **Temperature**: 0.2 (mặc định)
- Có thể điều chỉnh trong giao diện

### Auto-Load Model

Tính năng tự động load model khi khởi động có thể được bật/tắt trong code:
- Mặc định: TẮT
- Cần đủ RAM/VRAM và paging file đủ lớn

## 📝 Lưu Ý Quan Trọng

### Paging File

Ứng dụng yêu cầu paging file (virtual memory) đủ lớn:
- **Model 4B**: Tối thiểu 8GB paging file
- **Model 2B**: Tối thiểu 4GB paging file

Nếu gặp lỗi "paging file too small", hãy:
1. Mở System Properties (Win+R → `sysdm.cpl`)
2. Tab Advanced → Settings → Advanced
3. Virtual memory → Change
4. Tăng paging file lên ít nhất 8GB (hoặc để System managed)
5. Restart máy

### GPU vs CPU

- **GPU**: Nhanh hơn, sử dụng VRAM
- **CPU**: Chậm hơn, sử dụng RAM
- Ứng dụng sẽ khuyến nghị dùng GPU nếu phát hiện có GPU

### Memory Requirements

- **Model 4B**: 
  - GPU mode: ~8GB VRAM
  - CPU mode: ~8GB RAM + 8GB paging file
- **Model 2B**:
  - GPU mode: ~4GB VRAM
  - CPU mode: ~4GB RAM + 4GB paging file

## 🐛 Xử Lý Lỗi

### Lỗi: "Model not found"
- Đảm bảo đã tải models và đặt đúng thư mục
- Kiểm tra đường dẫn: `./models/Qwen3-VL-4B-Instruct/` và `./models/Qwen3-VL-2B-Instruct/`

### Lỗi: "Paging file too small"
- Xem phần [Paging File](#paging-file) ở trên

### Lỗi: "Out of memory"
- Giảm kích thước hình ảnh
- Dùng Model 2B thay vì 4B
- Tăng paging file
- Đóng các ứng dụng khác

### Lỗi: "CUDA not available"
- Kiểm tra GPU có hỗ trợ CUDA không
- Cài đặt CUDA Toolkit và cuDNN
- Hoặc dùng CPU mode

### Application Crash
- Kiểm tra file `crash_log.txt` để xem chi tiết lỗi
- Đảm bảo đủ RAM/VRAM và paging file
- Thử chạy lại với model 2B

## 📄 License

Dự án này sử dụng các thư viện mã nguồn mở. Vui lòng xem file LICENSE (nếu có) hoặc tham khảo license của các dependencies.

## 🙏 Acknowledgments

- [Qwen3-VL](https://github.com/QwenLM/Qwen2-VL) - Model AI
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [PyQt6](https://www.riverbankcomputing.com/software/pyqt/)
- [PyTorch](https://pytorch.org/)

## 📞 Support

Nếu gặp vấn đề, vui lòng:
1. Kiểm tra file `crash_log.txt`
2. Kiểm tra README này
3. Tạo issue trên GitHub

## 🎯 Roadmap

- [ ] Hỗ trợ video OCR
- [ ] Export kết quả ra nhiều format (JSON, CSV, etc.)
- [ ] OCR nhiều ngôn ngữ
- [ ] Batch processing với progress bar chi tiết hơn
- [ ] Tùy chỉnh prompt templates

