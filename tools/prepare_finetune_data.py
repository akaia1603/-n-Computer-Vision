"""
tools/prepare_finetune_data.py
==============================
Chuẩn bị dữ liệu webcam để fine-tune DAN / POSTER trên Colab.

Cách dùng:
    1. python tools/capture_webcam_data.py    # thu thập ảnh
    2. python tools/prepare_finetune_data.py  # đóng gói
    3. Upload webcam_finetune_data.zip lên Colab
"""

import os
import sys
import shutil
import zipfile
from glob import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

EMOTIONS = {
    1: "surprise", 2: "fear", 3: "disgust", 4: "happiness",
    5: "sadness", 6: "anger", 7: "neutral"
}
DATA_DIR = "data/DATASET"
OUTPUT_ZIP = "webcam_finetune_data.zip"


def main():
    all_files = []
    for cid, name in EMOTIONS.items():
        paths = glob(os.path.join(DATA_DIR, "train", str(cid), "webcam_*.jpg"))
        all_files.extend(paths)

    if not all_files:
        print(f" Không tìm thấy ảnh webcam nào trong {DATA_DIR}/train/1-7/")
        print(f" Chạy 'python tools/capture_webcam_data.py' trước.")
        return

    print(f" Tìm thấy {len(all_files)} ảnh webcam:")
    for cid, name in EMOTIONS.items():
        count = len(glob(os.path.join(DATA_DIR, "train", str(cid), "webcam_*.jpg")))
        bar = "#" * max(1, count // 5)
        print(f"   {cid}. {name:12s}: {count:3d}  {bar}")

    # Tạo thư mục tạm với cấu trúc giống RAF-DB
    tmp_dir = "temp_webcam_data"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)

    for cid in EMOTIONS:
        src = os.path.join(DATA_DIR, "train", str(cid))
        dst = os.path.join(tmp_dir, "train", str(cid))
        os.makedirs(dst, exist_ok=True)
        for f in glob(os.path.join(src, "webcam_*.jpg")):
            shutil.copy2(f, dst)

    # Zip
    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(tmp_dir):
            for f in files:
                path = os.path.join(root, f)
                arcname = os.path.relpath(path, tmp_dir)
                zf.write(path, arcname)

    shutil.rmtree(tmp_dir)

    size_mb = os.path.getsize(OUTPUT_ZIP) / 1024 / 1024
    print(f"\n Đã đóng gói: {OUTPUT_ZIP} ({size_mb:.1f} MB)")
    print(f" Upload file này lên Google Colab để fine-tune.")


if __name__ == "__main__":
    main()
