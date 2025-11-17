"""
Model loading functionality for ocrxdoc framework
"""

import os
import warnings
from typing import Optional, Dict, Any
from ocrxdoc.utils import get_torch, is_cuda_available


class ModelLoader:
    """Load and manage Qwen3-VL models"""
    
    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        dtype: Optional[str] = None,
        low_cpu_mem_usage: bool = True,
        max_memory: Optional[Dict[str, str]] = None
    ):
        """
        Initialize model loader
        
        Args:
            model_path: Path to model directory
            device: Device to use ("auto", "cuda:0", or "cpu")
            dtype: Data type ("float16", "float32", or None for auto)
            low_cpu_mem_usage: Use low CPU memory mode
            max_memory: Maximum memory per device (e.g., {"0": "10GiB"})
        """
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.low_cpu_mem_usage = low_cpu_mem_usage
        self.max_memory = max_memory
        self.model = None
        self.processor = None
    
    def _get_device_map(self) -> str:
        """Get device map string"""
        if self.device == "auto":
            if is_cuda_available():
                return "cuda:0"
            return "cpu"
        return self.device
    
    def _get_dtype(self):
        """Get dtype tensor"""
        torch = get_torch()
        device_map = self._get_device_map()
        
        if self.dtype:
            if self.dtype == "float16":
                return torch.float16
            elif self.dtype == "float32":
                return torch.float32
            else:
                warnings.warn(f"Unknown dtype: {self.dtype}. Using auto.")
        
        # Auto select based on device
        if device_map.startswith("cuda"):
            return torch.float16  # GPU typically uses float16
        else:
            return torch.float32  # CPU typically uses float32
    
    def load(self):
        """
        Load model and processor
        
        Returns:
            Tuple of (model, processor)
        """
        if not os.path.exists(self.model_path):
            raise ValueError(f"Model path does not exist: {self.model_path}")
        
        try:
            from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        except ImportError:
            raise ImportError(
                "transformers library is required. Install with: pip install transformers"
            )
        
        device_map = self._get_device_map()
        dtype = self._get_dtype()
        
        print(f"Loading model from: {self.model_path}")
        print(f"Device: {device_map}, dtype: {dtype}")
        
        # Check memory if psutil is available
        try:
            import psutil
            ram = psutil.virtual_memory()
            ram_available_gb = ram.available / (1024**3)
            print(f"Available RAM: {ram_available_gb:.1f}GB")
            
            if ram_available_gb < 4:
                warnings.warn(
                    f"Low RAM available: {ram_available_gb:.1f}GB. "
                    f"At least 4GB free RAM is recommended."
                )
        except ImportError:
            pass
        
        # Load model
        try:
            print("Loading model checkpoint... (This may take 2-5 minutes)")
            
            if device_map.startswith("cuda"):
                # GPU mode
                self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                    self.model_path,
                    dtype=dtype,
                    device_map=device_map,
                    low_cpu_mem_usage=self.low_cpu_mem_usage,
                    max_memory=self.max_memory if self.max_memory else {0: "10GiB"},
                )
            else:
                # CPU mode
                self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                    self.model_path,
                    dtype=dtype,
                    device_map="cpu",
                    low_cpu_mem_usage=self.low_cpu_mem_usage,
                )
            
            print("Model loaded successfully!")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")
        
        # Load processor
        try:
            print("Loading processor...")
            self.processor = AutoProcessor.from_pretrained(self.model_path)
            print("Processor loaded successfully!")
        except Exception as e:
            raise RuntimeError(f"Failed to load processor: {str(e)}")
        
        return self.model, self.processor
    
    def get_model(self):
        """Get loaded model"""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self.model
    
    def get_processor(self):
        """Get loaded processor"""
        if self.processor is None:
            raise RuntimeError("Processor not loaded. Call load() first.")
        return self.processor

