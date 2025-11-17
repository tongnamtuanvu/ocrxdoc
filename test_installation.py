"""
Script to test ocrxdoc installation
"""

def test_imports():
    """Test if all modules can be imported"""
    print("Testing imports...")
    try:
        from ocrxdoc import OCREngine, ModelLoader, FileProcessor
        from ocrxdoc import is_cuda_available, get_cuda_info
        print("✓ All imports successful!")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_cuda():
    """Test CUDA availability"""
    print("\nTesting CUDA...")
    try:
        from ocrxdoc import is_cuda_available, get_cuda_info
        
        cuda_available = is_cuda_available()
        print(f"CUDA available: {cuda_available}")
        
        if cuda_available:
            info = get_cuda_info()
            print(f"GPU: {info.get('device_name', 'Unknown')}")
            print(f"CUDA Version: {info.get('cuda_version', 'Unknown')}")
            print(f"GPU Memory: {info.get('gpu_memory', 0):.1f} GB")
        
        return True
    except Exception as e:
        print(f"✗ CUDA test error: {e}")
        return False

def test_model_paths():
    """Test if model paths exist"""
    print("\nTesting model paths...")
    import os
    
    model_2b = "./models/Qwen3-VL-2B-Instruct"
    model_4b = "./models/Qwen3-VL-4B-Instruct"
    
    if os.path.exists(model_2b):
        print(f"✓ 2B model found: {model_2b}")
    else:
        print(f"✗ 2B model not found: {model_2b}")
    
    if os.path.exists(model_4b):
        print(f"✓ 4B model found: {model_4b}")
    else:
        print(f"✗ 4B model not found: {model_4b}")
    
    return True

def test_engine_init():
    """Test engine initialization"""
    print("\nTesting engine initialization...")
    try:
        from ocrxdoc import OCREngine
        
        engine = OCREngine(model_size="4B", device="auto")
        print("✓ Engine initialized successfully!")
        print(f"  Model path: {engine.model_path}")
        print(f"  Device: {engine.device}")
        return True
    except Exception as e:
        print(f"✗ Engine initialization error: {e}")
        return False

def main():
    """Run all tests"""
    print("="*50)
    print("ocrxdoc Installation Test")
    print("="*50)
    
    results = []
    results.append(("Imports", test_imports()))
    results.append(("CUDA", test_cuda()))
    results.append(("Model Paths", test_model_paths()))
    results.append(("Engine Init", test_engine_init()))
    
    print("\n" + "="*50)
    print("Test Results:")
    print("="*50)
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{name}: {status}")
    
    all_passed = all(result for _, result in results)
    print("\n" + "="*50)
    if all_passed:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed. Please check the errors above.")
    print("="*50)

if __name__ == "__main__":
    main()

