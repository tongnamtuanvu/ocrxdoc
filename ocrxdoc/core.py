"""
Core OCR engine for ocrxdoc framework
"""

import os
from typing import Optional, Union, List, Tuple, Callable
from PIL import Image

from ocrxdoc.models import ModelLoader
from ocrxdoc.processors import FileProcessor
from ocrxdoc.utils import get_device


class OCREngine:
    """
    Main OCR engine for processing images and documents
    
    Example:
        >>> engine = OCREngine(model_size="4B", device="auto")
        >>> engine.load_model()
        >>> result = engine.ocr("image.jpg", prompt="Extract all text")
        >>> print(result)
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        model_size: str = "4B",
        device: str = "auto",
        dtype: Optional[str] = None,
        poppler_path: Optional[str] = None,
        max_tokens: int = 3000,
        temperature: float = 0.2,
        top_p: float = 0.8,
        top_k: int = 50,
        repetition_penalty: float = 1.1
    ):
        """
        Initialize OCR engine
        
        Args:
            model_path: Custom path to model directory. If None, uses default based on model_size
            model_size: Model size ("2B" or "4B"). Default: "4B"
            device: Device to use ("auto", "cuda:0", or "cpu"). Default: "auto"
            dtype: Data type ("float16", "float32", or None for auto)
            poppler_path: Path to Poppler bin directory (for PDF processing)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Repetition penalty
        """
        # Determine model path
        if model_path is None:
            if model_size == "2B":
                model_path = "./models/Qwen3-VL-2B-Instruct"
            elif model_size == "4B":
                model_path = "./models/Qwen3-VL-4B-Instruct"
            else:
                raise ValueError(f"Invalid model_size: {model_size}. Must be '2B' or '4B'")
        
        self.model_path = model_path
        self.model_size = model_size
        self.device = get_device(device)
        self.dtype = dtype
        
        # Initialize components
        self.model_loader = ModelLoader(
            model_path=self.model_path,
            device=self.device,
            dtype=self.dtype
        )
        self.file_processor = FileProcessor(poppler_path=poppler_path)
        
        # Generation parameters
        self.generation_params = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
        }
        
        # Model and processor (loaded after load_model())
        self.model = None
        self.processor = None
        self._model_loaded = False
    
    def load_model(self):
        """Load the OCR model and processor"""
        if self._model_loaded:
            print("Model already loaded.")
            return
        
        print(f"Loading OCR model ({self.model_size})...")
        self.model, self.processor = self.model_loader.load()
        self._model_loaded = True
        print("OCR engine ready!")
    
    def ocr(
        self,
        file_path: str,
        prompt: str = "Extract all text from this image. Return only the text content, no explanations.",
        roi: Optional[Tuple[int, int, int, int]] = None
    ) -> str:
        """
        Perform OCR on a file
        
        Args:
            file_path: Path to image, PDF, DOCX, or TXT file
            prompt: Prompt for the OCR model
            roi: Optional region of interest as (x, y, width, height) to crop before OCR
        
        Returns:
            Extracted text string
        
        Raises:
            RuntimeError: If model is not loaded
            ValueError: If file cannot be processed
        """
        if not self._model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Process file to get image
        image_path, file_type = self.file_processor.process_file(file_path)
        
        try:
            # Load image
            image = Image.open(image_path).convert("RGB")
            
            # Apply ROI if specified
            if roi is not None:
                x, y, width, height = roi
                image = image.crop((x, y, x + width, y + height))
            
            # Prepare messages
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # Prepare inputs
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
            inputs = inputs.to(self.model.device)
            
            # Generate
            generated_ids = self.model.generate(**inputs, **self.generation_params)
            
            # Decode
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            
            return output_text[0] if output_text else ""
        
        finally:
            # Cleanup temp files (except original image files)
            if file_type != 'image':
                self.file_processor.cleanup()
    
    def ocr_batch(
        self,
        file_paths: List[str],
        prompt: str = "Extract all text from this image. Return only the text content, no explanations.",
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> List[Tuple[str, str]]:
        """
        Perform OCR on multiple files
        
        Args:
            file_paths: List of file paths to process
            prompt: Prompt for the OCR model
            progress_callback: Optional callback function(current, total, current_file)
        
        Returns:
            List of tuples (file_path, ocr_result)
        """
        if not self._model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        results = []
        total = len(file_paths)
        
        for index, file_path in enumerate(file_paths):
            if progress_callback:
                progress_callback(index + 1, total, os.path.basename(file_path))
            
            try:
                result = self.ocr(file_path, prompt=prompt)
                results.append((file_path, result))
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                results.append((file_path, f"Error: {str(e)}"))
        
        return results
    
    def set_generation_params(
        self,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None
    ):
        """Update generation parameters"""
        if max_tokens is not None:
            self.generation_params["max_new_tokens"] = max_tokens
        if temperature is not None:
            self.generation_params["temperature"] = temperature
        if top_p is not None:
            self.generation_params["top_p"] = top_p
        if top_k is not None:
            self.generation_params["top_k"] = top_k
        if repetition_penalty is not None:
            self.generation_params["repetition_penalty"] = repetition_penalty
    
    def cleanup(self):
        """Clean up temporary files"""
        self.file_processor.cleanup()
    
    def __del__(self):
        """Cleanup on deletion"""
        self.cleanup()

