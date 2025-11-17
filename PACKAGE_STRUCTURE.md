# Cấu Trúc Package ocrxdoc

## Tổng Quan

Package `ocrxdoc` đã được tạo thành công như một Python framework có thể import và sử dụng như các open-source framework khác.

## Cấu Trúc Thư Mục

```
ocrdocs/
├── ocrxdoc/                    # Package chính
│   ├── __init__.py           # Exports API chính
│   ├── core.py               # OCREngine - class chính
│   ├── models.py             # ModelLoader - quản lý model
│   ├── processors.py         # FileProcessor - xử lý file
│   └── utils.py              # Utility functions
│
├── examples/                  # Ví dụ sử dụng
│   ├── __init__.py
│   ├── basic_usage.py        # Ví dụ cơ bản
│   └── batch_processing.py  # Ví dụ batch processing
│
├── setup.py                  # Setup script cho pip
├── pyproject.toml            # Modern Python packaging
├── MANIFEST.in               # Files to include in package
├── requirements_ocrxdoc.txt  # Dependencies
├── README_OCRXDOC.md         # Documentation
├── INSTALL_GUIDE.md          # Hướng dẫn cài đặt
├── test_installation.py      # Script kiểm tra cài đặt
└── PACKAGE_STRUCTURE.md      # File này

```

## Cách Sử Dụng

### 1. Cài Đặt Package

```bash
# Development mode (có thể chỉnh sửa code)
pip install -e .

# Hoặc cài đặt bình thường
pip install .
```

### 2. Import và Sử Dụng

```python
from ocrxdoc import OCREngine

# Khởi tạo
engine = OCREngine(model_size="4B", device="auto")

# Load model
engine.load_model()

# OCR
result = engine.ocr("image.jpg", prompt="Extract all text")
print(result)
```

### 3. Batch Processing

```python
from ocrxdoc import OCREngine

engine = OCREngine(model_size="4B")
engine.load_model()

files = ["img1.jpg", "img2.png", "doc.pdf"]
results = engine.ocr_batch(files)

for file_path, result in results:
    print(f"{file_path}: {result}")
```

## API Chính

### OCREngine

Class chính để thực hiện OCR:

- `__init__()`: Khởi tạo engine
- `load_model()`: Load model và processor
- `ocr()`: OCR một file
- `ocr_batch()`: OCR nhiều file
- `set_generation_params()`: Cập nhật tham số generation
- `cleanup()`: Dọn dẹp temp files

### ModelLoader

Quản lý việc load model:

- `load()`: Load model và processor
- `get_model()`: Lấy model đã load
- `get_processor()`: Lấy processor đã load

### FileProcessor

Xử lý các loại file:

- `process_file()`: Xử lý file và trả về image path
- `process_pdf()`: Convert PDF sang image
- `process_docx()`: Convert DOCX sang image
- `process_txt()`: Convert TXT sang image

### Utilities

- `is_cuda_available()`: Kiểm tra CUDA
- `get_cuda_info()`: Lấy thông tin CUDA
- `get_device()`: Lấy device phù hợp

## Tính Năng

✅ Không phụ thuộc PyQt6 (core framework)
✅ API sạch, dễ sử dụng
✅ Hỗ trợ GPU/CPU tự động
✅ Hỗ trợ nhiều loại file (image, PDF, DOCX, TXT)
✅ Batch processing
✅ ROI selection
✅ Customizable generation parameters
✅ Có thể cài đặt qua pip

## So Sánh với Code Gốc

| Tính Năng | Code Gốc (main.py) | ocrxdoc Framework |
|-----------|-------------------|-------------------|
| GUI | ✅ PyQt6 | ❌ (tách riêng) |
| Core OCR | ✅ | ✅ |
| Model Loading | ✅ | ✅ |
| File Processing | ✅ | ✅ |
| Batch Processing | ✅ | ✅ |
| Import được | ❌ | ✅ |
| Cài đặt qua pip | ❌ | ✅ |
| API sạch | ❌ | ✅ |

## Lưu Ý

1. Models vẫn cần được tải thủ công và đặt vào `./models/`
2. Poppler cần thiết cho PDF (có thể tự động tìm trong `./poppler/`)
3. Framework không bao gồm GUI - chỉ core OCR functionality
4. Có thể sử dụng cùng với code GUI gốc (main.py) nếu cần

## Next Steps

1. Test framework với các file thực tế
2. Có thể publish lên PyPI nếu muốn
3. Có thể tạo thêm GUI wrapper sử dụng framework này
4. Có thể thêm các tính năng mới (export formats, etc.)

