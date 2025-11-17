"""
ocrxdoc - Python Framework for OCR using Qwen3-VL Models

A clean, easy-to-use Python framework for OCR (Optical Character Recognition)
using Qwen3-VL AI models. Supports images (JPG, PNG, JPEG), PDF, DOCX, and TXT files.

Example:
    >>> from ocrxdoc import OCREngine
    >>> 
    >>> # Initialize OCR engine
    >>> engine = OCREngine(model_size="4B", device="auto")
    >>> 
    >>> # Load model
    >>> engine.load_model()
    >>> 
    >>> # Process an image
    >>> result = engine.ocr("path/to/image.jpg", prompt="Extract all text from this image")
    >>> print(result)
"""

__version__ = "1.0.0"
__author__ = "OCR Framework"

from ocrxdoc.core import OCREngine
from ocrxdoc.models import ModelLoader
from ocrxdoc.processors import FileProcessor
from ocrxdoc.utils import is_cuda_available, get_cuda_info

__all__ = [
    "OCREngine",
    "ModelLoader",
    "FileProcessor",
    "is_cuda_available",
    "get_cuda_info",
]

