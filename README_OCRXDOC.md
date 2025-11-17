# ocrxdoc - Python Framework for OCR

A clean, easy-to-use Python framework for OCR (Optical Character Recognition) using Qwen3-VL AI models. Supports images (JPG, PNG, JPEG), PDF, DOCX, and TXT files.

## Features

- 🖼️ **Image OCR**: Support for JPG, PNG, JPEG
- 📄 **Document OCR**: Support for PDF, DOCX, TXT
- 🤖 **Two AI Models**: 
  - 4B model (default) - More accurate
  - 2B model - Faster
- 🖥️ **GPU/CPU Support**: Automatic GPU detection and usage
- 🎯 **ROI Selection**: Select custom regions for OCR
- 📦 **Batch Processing**: Process multiple files at once
- ⚡ **Easy to Use**: Simple, clean API

## Installation

### Basic Installation

```bash
pip install ocrxdoc
```

### With PDF Support

```bash
pip install ocrxdoc[pdf]
```

### With DOCX Support

```bash
pip install ocrxdoc[docx]
```

### With All Features

```bash
pip install ocrxdoc[all]
```

## Quick Start

### Basic Usage

```python
from ocrxdoc import OCREngine

# Initialize OCR engine
engine = OCREngine(model_size="4B", device="auto")

# Load model
engine.load_model()

# Process an image
result = engine.ocr("path/to/image.jpg", prompt="Extract all text from this image")
print(result)
```

### Process Different File Types

```python
from ocrxdoc import OCREngine

engine = OCREngine(model_size="4B")
engine.load_model()

# Process image
result = engine.ocr("image.jpg")

# Process PDF
result = engine.ocr("document.pdf")

# Process DOCX
result = engine.ocr("document.docx")

# Process TXT
result = engine.ocr("text.txt")
```

### Batch Processing

```python
from ocrxdoc import OCREngine

engine = OCREngine(model_size="4B")
engine.load_model()

files = ["image1.jpg", "image2.png", "document.pdf"]

def progress_callback(current, total, filename):
    print(f"Processing {current}/{total}: {filename}")

results = engine.ocr_batch(files, progress_callback=progress_callback)

for file_path, result in results:
    print(f"{file_path}: {result[:100]}...")
```

### Custom Model Path

```python
from ocrxdoc import OCREngine

# Use custom model path
engine = OCREngine(
    model_path="./custom/models/Qwen3-VL-4B-Instruct",
    device="cuda:0"
)
engine.load_model()
```

### ROI (Region of Interest) Selection

```python
from ocrxdoc import OCREngine

engine = OCREngine(model_size="4B")
engine.load_model()

# OCR only a specific region: (x, y, width, height)
result = engine.ocr(
    "image.jpg",
    roi=(100, 100, 500, 300)  # Crop region before OCR
)
```

### Custom Generation Parameters

```python
from ocrxdoc import OCREngine

engine = OCREngine(
    model_size="4B",
    max_tokens=5000,
    temperature=0.1,
    top_p=0.9
)
engine.load_model()

# Or update after initialization
engine.set_generation_params(
    max_tokens=5000,
    temperature=0.1
)
```

## Model Setup

Models need to be downloaded manually due to their large size:

1. **4B Model (Default)**:
   - Download from: [Hugging Face - Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)
   - Place in: `./models/Qwen3-VL-4B-Instruct/`

2. **2B Model**:
   - Download from: [Hugging Face - Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen3-VL-2B-Instruct)
   - Place in: `./models/Qwen3-VL-2B-Instruct/`

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.57+
- Pillow 10.0+
- For PDF: pdf2image and Poppler
- For DOCX: python-docx

## System Requirements

- **RAM**: Minimum 16GB (recommended 32GB+)
- **GPU**: Recommended (NVIDIA with CUDA support) - VRAM minimum 8GB
- **Paging File**: Minimum 8GB for 4B model, 4GB for 2B model

## API Reference

### OCREngine

Main OCR engine class.

#### `__init__(model_path=None, model_size="4B", device="auto", dtype=None, poppler_path=None, max_tokens=3000, temperature=0.2, top_p=0.8, top_k=50, repetition_penalty=1.1)`

Initialize OCR engine.

#### `load_model()`

Load the OCR model and processor.

#### `ocr(file_path, prompt="...", roi=None)`

Perform OCR on a file.

- `file_path`: Path to file
- `prompt`: Prompt for OCR model
- `roi`: Optional region of interest as (x, y, width, height)

Returns: Extracted text string

#### `ocr_batch(file_paths, prompt="...", progress_callback=None)`

Perform OCR on multiple files.

- `file_paths`: List of file paths
- `prompt`: Prompt for OCR model
- `progress_callback`: Optional callback(current, total, filename)

Returns: List of tuples (file_path, ocr_result)

#### `set_generation_params(max_tokens=None, temperature=None, top_p=None, top_k=None, repetition_penalty=None)`

Update generation parameters.

#### `cleanup()`

Clean up temporary files.

## Examples

See `examples/` directory for more examples.

## License

MIT License

## Acknowledgments

- [Qwen3-VL](https://github.com/QwenLM/Qwen2-VL) - AI Model
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [PyTorch](https://pytorch.org/)

