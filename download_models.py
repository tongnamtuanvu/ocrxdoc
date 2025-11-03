#!/usr/bin/env python3
"""
Script để tải cả 2 model Qwen3-VL về (2B và 4B)
Chạy: python download_models.py
"""

import os
import sys
from pathlib import Path

def download_model(model_name, output_dir):
    """
    Tải model từ Hugging Face
    
    Args:
        model_name: Tên model trên Hugging Face (vd: "Qwen/Qwen3-VL-4B-Instruct")
        output_dir: Thư mục lưu model
    """
    try:
        from huggingface_hub import snapshot_download
        
        print(f"\n{'='*80}")
        print(f"Đang tải model: {model_name}")
        print(f"Thư mục đích: {output_dir}")
        print(f"{'='*80}\n")
        
        # Tạo thư mục nếu chưa có
        os.makedirs(output_dir, exist_ok=True)
        
        # Tải model
        snapshot_download(
            repo_id=model_name,
            local_dir=output_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
            max_workers=4
        )
        
        print(f"\n[OK] Da tai xong model: {model_name}")
        print(f"   Luu tai: {os.path.abspath(output_dir)}\n")
        return True
        
    except ImportError:
        print("[LOI] Chua cai dat huggingface_hub!")
        print("   Cai dat: pip install huggingface_hub")
        return False
    except Exception as e:
        print(f"[LOI] Loi khi tai model {model_name}: {str(e)}")
        return False

def check_disk_space(required_gb=50):
    """Kiểm tra dung lượng ổ đĩa"""
    try:
        import shutil
        stat = shutil.disk_usage(".")
        free_gb = stat.free / (1024**3)
        
        print(f"\nDung lượng ổ đĩa:")
        print(f"  - Còn trống: {free_gb:.1f} GB")
        print(f"  - Cần thiết: ~{required_gb} GB")
        
        if free_gb < required_gb:
            print(f"\n[CANH BAO] Dung luong co the khong du!")
            response = input("Tiep tuc tai? (y/n): ")
            return response.lower() == 'y'
        return True
    except:
        return True

def main():
    print("="*80)
    print("SCRIPT TẢI MODEL QWEN3-VL")
    print("="*80)
    print("\nScript này sẽ tải 2 model:")
    print("  1. Qwen3-VL-2B-Instruct (~5-6 GB)")
    print("  2. Qwen3-VL-4B-Instruct (~10-12 GB)")
    print("\nTong cong: ~15-20 GB")
    print("\n[THOI GIAN] Uoc tinh: 30-60 phut (tuy toc do mang)")
    
    # Kiểm tra dung lượng
    if not check_disk_space(20):
        print("\n[HUY] Huy tai model.")
        return
    
    # Hỏi xác nhận
    print("\n" + "="*80)
    response = input("Bat dau tai? (y/n): ")
    if response.lower() != 'y':
        print("[HUY] Huy tai model.")
        return
    
    # Định nghĩa models
    models = [
        {
            "name": "Qwen/Qwen3-VL-2B-Instruct",
            "output_dir": "./models/Qwen3-VL-2B-Instruct",
            "size": "~5-6 GB"
        },
        {
            "name": "Qwen/Qwen3-VL-4B-Instruct",
            "output_dir": "./models/Qwen3-VL-4B-Instruct",
            "size": "~10-12 GB"
        }
    ]
    
    # Hỏi người dùng muốn tải model nào
    print("\n" + "="*80)
    print("Chon model muon tai:")
    print("  1. Chi tai Qwen3-VL-2B-Instruct (nhe, nhanh hon)")
    print("  2. Chi tai Qwen3-VL-4B-Instruct (chinh xac hon)")
    print("  3. Tai ca 2 model")
    choice = input("Nhap lua chon (1/2/3): ").strip()
    
    if choice == "1":
        models_to_download = [models[0]]
    elif choice == "2":
        models_to_download = [models[1]]
    elif choice == "3":
        models_to_download = models
    else:
        print("[LOI] Lua chon khong hop le!")
        return
    
    # Tải models
    success_count = 0
    for model in models_to_download:
        print(f"\n[MODEL] {model['name']} ({model['size']})")
        if download_model(model['name'], model['output_dir']):
            success_count += 1
    
    # Tổng kết
    print("\n" + "="*80)
    print("KET QUA TAI MODEL")
    print("="*80)
    print(f"[THANH CONG] {success_count}/{len(models_to_download)} model")
    
    if success_count == len(models_to_download):
        print("\n[HOAN THANH] Da tai xong tat ca model!")
        print("\nBan co the chay ung dung OCR ngay bay gio:")
        print("  python main.py")
    else:
        print("\n[CANH BAO] Mot so model chua tai xong. Vui long thu lai.")
    
    print("\nThu muc models:")
    for model in models_to_download:
        output_path = os.path.abspath(model['output_dir'])
        exists = "[OK]" if os.path.exists(output_path) else "[CHUA CO]"
        print(f"  {exists} {output_path}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[HUY] Da huy tai model (Ctrl+C)")
        print("   Ban co the chay lai script de tiep tuc tai (resume download)")
    except Exception as e:
        print(f"\n[LOI] {str(e)}")
        import traceback
        traceback.print_exc()

