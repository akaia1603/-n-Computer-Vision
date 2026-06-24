"""
tools/capture_webcam_data.py
=============================
Chụp ảnh khuôn mặt từ webcam để fine-tune model (console mode).

Cách dùng:
    python tools/capture_webcam_data.py

Phím tắt:
    1-7  Lưu ảnh vào class tương ứng (1=Surprise ... 7=Neutral)
    q    Thoát
"""

import os
import sys
import glob
import cv2
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

EMOTIONS = {
    1: "Surprise", 2: "Fear", 3: "Disgust", 4: "Happiness",
    5: "Sadness", 6: "Anger", 7: "Neutral"
}
SAVE_DIR = os.path.join("data", "DATASET", "train")
for cid in EMOTIONS:
    os.makedirs(os.path.join(SAVE_DIR, str(cid)), exist_ok=True)


def get_next_filename(class_id):
    existing = glob.glob(os.path.join(SAVE_DIR, str(class_id), "*.jpg"))
    return f"webcam_{len(existing) + 1:04d}.jpg"


def capture_from_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(" Không thể mở webcam")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    total_saved = 0
    latest_frame = None
    running = True

    # Thread đọc frame liên tục
    def read_loop():
        nonlocal latest_frame
        while running:
            ret, frame = cap.read()
            if ret:
                latest_frame = cv2.flip(frame, 1)

    t = threading.Thread(target=read_loop, daemon=True)
    t.start()
    time.sleep(0.5)

    TARGET = {1: 50, 2: 30, 3: 100, 4: 30, 5: 30, 6: 30, 7: 30}

    print("=" * 55)
    print("  THU THẬP DỮ LIỆU WEBCAM CHO FINE-TUNE")
    print("=" * 55)
    print(f"  KPI mục tiêu:")
    for cid, name in EMOTIONS.items():
        tag = "QUAN TRỌNG" if cid == 3 else "Tối thiểu"
        print(f"    [{cid}] {name:10s}: {TARGET[cid]} ảnh ({tag})")
    print("  Nhập số 1-7 + Enter để chụp, q=thoát")
    print("=" * 55)
    print(f"  Lưu vào: {SAVE_DIR}")
    print("=" * 55)

    try:
        while running:
            if latest_frame is None:
                time.sleep(0.1)
                continue

            print("\n" + "-" * 50)
            for cid, name in EMOTIONS.items():
                have = len(glob.glob(os.path.join(SAVE_DIR, str(cid), "*.jpg")))
                n = min(have, TARGET[cid])
                remain = TARGET[cid] - have
                flag = " [OK]" if have >= TARGET[cid] else f" [{have}/{TARGET[cid]}]"
                print(f"  [{cid}] {name:10s} {'#' * n}{'.' * max(0, remain)} {flag}")
            print("-" * 50)

            cmd = input("  Nhap so (1-7) de chup, q=thoat: ").strip()

            if cmd == 'q':
                break
            elif cmd in '1234567':
                class_id = int(cmd)
                filename = get_next_filename(class_id)
                save_path = os.path.join(SAVE_DIR, str(class_id), filename)
                cv2.imwrite(save_path, latest_frame)
                total_saved += 1
                print(f"  → Đã lưu: [{EMOTIONS[class_id]}] {save_path}")
            else:
                print(f"  Phím không hợp lệ.")
    except (KeyboardInterrupt, EOFError):
        pass

    running = False
    cap.release()
    print(f"\n  Tổng cộng đã lưu: {total_saved} ảnh")


if __name__ == "__main__":
    capture_from_camera()
