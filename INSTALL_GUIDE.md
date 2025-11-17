# Hướng Dẫn Cài Đặt ocrxdoc Framework

## Cài Đặt Framework

### Cách 1: Cài đặt từ source code (Development Mode)

```bash
# Clone hoặc download project
cd ocrdocs

# Cài đặt ở chế độ development (có thể chỉnh sửa code)
pip install -e .

# Hoặc cài đặt bình thường
pip install .
```

### Cách 2: Cài đặt từ thư mục local

```bash
# Từ thư mục chứa setup.py
pip install .
```

### Cài đặt với các tính năng bổ sung

```bash
# Với hỗ trợ PDF
pip install .[pdf]

# Với hỗ trợ DOCX
pip install .[docx]

# Với tất cả tính năng
pip install .[all]
```

## Cài Đặt Dependencies

### Core Dependencies

```bash
pip install torch transformers Pillow accelerate
```

### Optional Dependencies

```bash
# Cho PDF
pip install pdf2image

# Cho DOCX
pip install python-docx

# Utilities
pip install psutil PyPDF2
```

## Tải Models

Models cần được tải thủ công do kích thước lớn:

1. **Model 4B (Mặc định)**:
   - Tải từ: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
   - Đặt vào: `./models/Qwen3-VL-4B-Instruct/`

2. **Model 2B**:
   - Tải từ: https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct
   - Đặt vào: `./models/Qwen3-VL-2B-Instruct/`

## Kiểm Tra Cài Đặt

Sau khi cài đặt, bạn có thể kiểm tra:

```python
from ocrxdoc import OCREngine, is_cuda_available

# Kiểm tra CUDA
print(f"CUDA available: {is_cuda_available()}")

# Khởi tạo engine
engine = OCREngine(model_size="4B")
print("OCR Engine initialized successfully!")
```

## Sử Dụng

Xem file `README_OCRXDOC.md` và thư mục `examples/` để biết cách sử dụng.

