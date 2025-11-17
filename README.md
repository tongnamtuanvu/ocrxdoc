# ocrxdoc

Python Framework for OCR using Qwen3-VL Models

A clean, easy-to-use Python framework for OCR (Optical Character Recognition) using Qwen3-VL AI models. Supports images (JPG, PNG, JPEG), PDF, DOCX, and TXT files.

## ✨ Features

- 🖼️ **Image OCR**: Support for JPG, PNG, JPEG
- 📄 **Document OCR**: Support for PDF, DOCX, TXT
- 🤖 **Two AI Models**: 
  - 4B model (default) - More accurate
  - 2B model - Faster
- 🖥️ **GPU/CPU Support**: Automatic GPU detection and usage
- 🎯 **ROI Selection**: Select custom regions for OCR
- 📦 **Batch Processing**: Process multiple files at once
- ⚡ **Easy to Use**: Simple, clean API

## 🚀 Quick Start

### Installation

```bash
pip install -e .
```

Or with optional features:

```bash
# With PDF support
pip install -e .[pdf]

# With DOCX support
pip install -e .[docx]

# With all features
pip install -e .[all]
```

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

## 📖 Documentation

- [Full Documentation](README_OCRXDOC.md)
- [Installation Guide](INSTALL_GUIDE.md)
- [Package Structure](PACKAGE_STRUCTURE.md)
- [Examples](examples/)

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.57+
- Pillow 10.0+

## 🤖 Model Setup

Models need to be downloaded manually:

1. **4B Model**: [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)
   - Place in: `./models/Qwen3-VL-4B-Instruct/`

2. **2B Model**: [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)
   - Place in: `./models/Qwen3-VL-2B-Instruct/`

## 📝 Examples

See the [examples](examples/) directory for more usage examples.

## 📄 License

MIT License

## 🙏 Acknowledgments

- [Qwen3-VL](https://github.com/QwenLM/Qwen2-VL) - AI Model
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [PyTorch](https://pytorch.org/)
