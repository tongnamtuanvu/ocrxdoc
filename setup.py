"""
Setup script for ocrxdoc package
"""

from setuptools import setup, find_packages
import os

# Read README if exists
readme_path = os.path.join(os.path.dirname(__file__), "README_OCRXDOC.md")
long_description = ""
if os.path.exists(readme_path):
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()

setup(
    name="ocrxdoc",
    version="1.0.0",
    author="OCR Framework",
    description="Python Framework for OCR using Qwen3-VL Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tongnamtuanvu/ocrxdoc",
    license="MIT",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.57.0",
        "Pillow>=10.0.0",
        "accelerate>=0.20.0",
    ],
    extras_require={
        "pdf": ["pdf2image>=1.16.0"],
        "docx": ["python-docx>=1.1.0"],
        "all": [
            "pdf2image>=1.16.0",
            "python-docx>=1.1.0",
            "PyPDF2>=3.0.0",
            "psutil>=5.9.0",
        ],
    },
    keywords="ocr, ai, qwen, vision-language, document-processing",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/ocrxdoc/issues",
        "Source": "https://github.com/yourusername/ocrxdoc",
    },
)

