"""
Basic usage example for ocrxdoc framework
"""

from ocrxdoc import OCREngine

def main():
    # Initialize OCR engine with 4B model
    print("Initializing OCR engine...")
    engine = OCREngine(model_size="4B", device="auto")
    
    # Load model (this may take a few minutes)
    print("Loading model...")
    engine.load_model()
    
    # Process an image
    print("\nProcessing image...")
    result = engine.ocr(
        "path/to/your/image.jpg",
        prompt="Extract all text from this image. Return only the text content."
    )
    
    print("\nOCR Result:")
    print(result)
    
    # Cleanup
    engine.cleanup()

if __name__ == "__main__":
    main()

