"""
Utility functions for ocrxdoc framework
"""

import warnings

# Lazy import torch
_torch = None

def get_torch():
    """Lazy import torch - only import when needed"""
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch

def is_cuda_available():
    """Check if CUDA is available"""
    try:
        torch = get_torch()
        return torch.cuda.is_available()
    except:
        return False

def get_cuda_info():
    """Get CUDA information"""
    try:
        torch = get_torch()
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            pytorch_version = torch.__version__
            
            warnings_list = []
            if cuda_version:
                cuda_major = float('.'.join(cuda_version.split('.')[:2]))
                if cuda_major >= 13.0:
                    warnings_list.append(
                        f"CUDA {cuda_version} may not be officially supported by PyTorch {pytorch_version}. "
                        f"Recommended: CUDA 11.8 or 12.1."
                    )
                elif cuda_major < 11.0:
                    warnings_list.append(
                        f"CUDA {cuda_version} is too old. Recommended: CUDA 11.8 or 12.1."
                    )
            
            return {
                'available': True,
                'device_name': torch.cuda.get_device_name(0),
                'cuda_version': cuda_version,
                'pytorch_version': pytorch_version,
                'gpu_memory': torch.cuda.get_device_properties(0).total_memory / (1024**3),
                'warnings': warnings_list
            }
    except Exception as e:
        warnings.warn(f"Error getting CUDA info: {e}")
    
    return {'available': False, 'warnings': []}

def get_device(device_preference="auto"):
    """
    Get the appropriate device based on preference
    
    Args:
        device_preference: "auto", "cuda", or "cpu"
    
    Returns:
        str: Device string ("cuda:0" or "cpu")
    """
    if device_preference == "auto":
        if is_cuda_available():
            return "cuda:0"
        return "cpu"
    elif device_preference == "cuda":
        if is_cuda_available():
            return "cuda:0"
        warnings.warn("CUDA requested but not available. Falling back to CPU.")
        return "cpu"
    else:
        return "cpu"

