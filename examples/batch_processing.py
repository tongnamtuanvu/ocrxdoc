"""
Batch processing example for ocrxdoc framework
"""

from ocrxdoc import OCREngine

def progress_callback(current, total, filename):
    """Progress callback for batch processing"""
    print(f"[{current}/{total}] Processing: {filename}")

def main():
    # Initialize OCR engine
    print("Initializing OCR engine...")
    engine = OCREngine(model_size="4B", device="auto")
    
    # Load model
    print("Loading model...")
    engine.load_model()
    
    # List of files to process
    files = [
        "image1.jpg",
        "image2.png",
        "document.pdf",
        "document.docx",
        "text.txt"
    ]
    
    # Process all files
    print("\nStarting batch processing...")
    results = engine.ocr_batch(
        files,
        prompt="Extract all text from this image. Return only the text content.",
        progress_callback=progress_callback
    )
    
    # Print results
    print("\n" + "="*50)
    print("Results:")
    print("="*50)
    for file_path, result in results:
        print(f"\nFile: {file_path}")
        print(f"Result: {result[:200]}...")  # First 200 chars
    
    # Cleanup
    engine.cleanup()

if __name__ == "__main__":
    main()

