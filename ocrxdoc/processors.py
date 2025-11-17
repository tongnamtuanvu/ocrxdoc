"""
File processors for converting various file types to images for OCR
"""

import os
import tempfile
from typing import Optional, Tuple
from PIL import Image, ImageDraw, ImageFont

# Optional dependencies
try:
    from pdf2image import convert_from_path
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

try:
    from docx import Document
    DOCX_SUPPORT = True
except ImportError:
    DOCX_SUPPORT = False


class FileProcessor:
    """Process various file types and convert them to images for OCR"""
    
    def __init__(self, poppler_path: Optional[str] = None):
        """
        Initialize file processor
        
        Args:
            poppler_path: Path to Poppler bin directory (for PDF processing)
        """
        self.poppler_path = poppler_path
        self.temp_files = []
    
    def find_poppler_path(self) -> Optional[str]:
        """Find Poppler path automatically"""
        # Check common locations
        poppler_base = "./poppler"
        if os.path.exists(poppler_base):
            # Check for nested structure: poppler-XX.XX.X/Library/bin
            for item in os.listdir(poppler_base):
                item_path = os.path.join(poppler_base, item)
                if os.path.isdir(item_path) and item.startswith("poppler-"):
                    bin_path = os.path.join(item_path, "Library", "bin")
                    if os.path.exists(bin_path):
                        return os.path.abspath(bin_path)
            
            # Check for direct structure: poppler/Library/bin
            bin_path = os.path.join(poppler_base, "Library", "bin")
            if os.path.exists(bin_path):
                return os.path.abspath(bin_path)
        
        return None
    
    def process_file(self, file_path: str) -> Tuple[str, str]:
        """
        Process a file and return image path and file type
        
        Args:
            file_path: Path to the file to process
        
        Returns:
            Tuple of (image_path, file_type)
            file_type can be: 'image', 'pdf', 'docx', 'txt'
        
        Raises:
            ValueError: If file type is not supported or processing fails
        """
        if not os.path.exists(file_path):
            raise ValueError(f"File not found: {file_path}")
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
            return file_path, 'image'
        elif file_ext == '.pdf':
            return self.process_pdf(file_path)
        elif file_ext == '.docx':
            return self.process_docx(file_path)
        elif file_ext == '.txt':
            return self.process_txt(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
    
    def process_pdf(self, file_path: str) -> Tuple[str, str]:
        """
        Convert PDF first page to image
        
        Args:
            file_path: Path to PDF file
        
        Returns:
            Tuple of (image_path, 'pdf')
        """
        if not PDF_SUPPORT:
            raise ImportError(
                "PDF support requires pdf2image. Install with: pip install pdf2image"
            )
        
        poppler_path = self.poppler_path or self.find_poppler_path()
        
        try:
            # Convert first page to image
            if poppler_path and os.path.exists(os.path.join(poppler_path, "pdftoppm.exe")):
                images = convert_from_path(
                    file_path,
                    first_page=1,
                    last_page=1,
                    dpi=200,
                    poppler_path=poppler_path
                )
            else:
                # Try system PATH poppler
                images = convert_from_path(file_path, first_page=1, last_page=1, dpi=200)
            
            if not images:
                raise ValueError("Could not extract image from PDF. Ensure Poppler is installed.")
            
            # Save to temp file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            images[0].save(temp_file.name, 'PNG')
            self.temp_files.append(temp_file.name)
            
            return temp_file.name, 'pdf'
        
        except Exception as e:
            error_msg = str(e)
            if "poppler" in error_msg.lower() or "pdftoppm" in error_msg.lower():
                raise ValueError(
                    f"Poppler not found. Please install Poppler:\n"
                    f"1. Download from: https://github.com/oschwartz10612/poppler-windows/releases\n"
                    f"2. Add Poppler bin folder to PATH\n\n"
                    f"Error: {error_msg}"
                )
            raise ValueError(f"Error processing PDF: {error_msg}")
    
    def process_docx(self, file_path: str) -> Tuple[str, str]:
        """
        Convert DOCX to image
        
        Args:
            file_path: Path to DOCX file
        
        Returns:
            Tuple of (image_path, 'docx')
        """
        if not DOCX_SUPPORT:
            raise ImportError(
                "DOCX support requires python-docx. Install with: pip install python-docx"
            )
        
        try:
            # Read DOCX
            doc = Document(file_path)
            
            # Extract text
            text_content = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
            
            if not text_content.strip():
                raise ValueError("DOCX file appears to be empty")
            
            # Create image from text
            img = self._text_to_image(text_content, max_lines=50)
            
            # Save to temp file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            img.save(temp_file.name, 'PNG')
            self.temp_files.append(temp_file.name)
            
            return temp_file.name, 'docx'
        
        except Exception as e:
            raise ValueError(f"Error processing DOCX: {str(e)}")
    
    def process_txt(self, file_path: str) -> Tuple[str, str]:
        """
        Convert TXT to image
        
        Args:
            file_path: Path to TXT file
        
        Returns:
            Tuple of (image_path, 'txt')
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if not content.strip():
                raise ValueError("TXT file appears to be empty")
            
            # Create image from text
            img = self._text_to_image(content, max_lines=100)
            
            # Save to temp file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            img.save(temp_file.name, 'PNG')
            self.temp_files.append(temp_file.name)
            
            return temp_file.name, 'txt'
        
        except Exception as e:
            raise ValueError(f"Error processing TXT: {str(e)}")
    
    def _text_to_image(self, text: str, max_lines: int = 100) -> Image.Image:
        """Convert text to image"""
        lines = text.split('\n')
        max_width = max(len(line) for line in lines) if lines else 80
        img_width = min(max_width * 10, 1200)
        img_height = min(len(lines) * 25 + 40, 3000)
        
        # Create image
        img = Image.new('RGB', (img_width, img_height), color='white')
        draw = ImageDraw.Draw(img)
        
        # Try to use default font
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        # Draw text
        y = 20
        for line in lines[:max_lines]:
            draw.text((20, y), line[:120], fill='black', font=font)
            y += 25
        
        return img
    
    def cleanup(self):
        """Clean up temporary files"""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except:
                pass
        self.temp_files = []
    
    def __del__(self):
        """Cleanup on deletion"""
        self.cleanup()

