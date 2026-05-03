import os
import sys
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend cho lưu file
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# CẤU HÌNH
DATA_DIR = "data/cnn_dataset"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
TRAIN_CSV = os.path.join(DATA_DIR, "train_labels.csv")
TEST_CSV = os.path.join(DATA_DIR, "test_labels.csv")
OUTPUT_DIR = "outputs/eda"

# Mapping label → tên cảm xúc
EMOTION_MAP = {
    1: "Surprise",
    2: "Fear",
    3: "Disgust",
    4: "Happiness",
    5: "Sadness",
    6: "Anger",
    7: "Neutral"
}

EMOTION_MAP_VI = {
    1: "Ngạc nhiên",
    2: "Sợ hãi",
    3: "Ghê tởm",
    4: "Vui vẻ",
    5: "Buồn bã",
    6: "Tức giận",
    7: "Trung lập"
}

# Màu sắc cho mỗi cảm xúc (palette đẹp hơn)
EMOTION_COLORS = {
    1: "#FF6B6B",  # Surprise - đỏ san hô
    2: "#845EC2",  # Fear - tím
    3: "#4B8B3B",  # Disgust - xanh rêu
    4: "#FFD93D",  # Happiness - vàng
    5: "#4A90D9",  # Sadness - xanh dương
    6: "#FF4444",  # Anger - đỏ
    7: "#95A5A6",  # Neutral - xám
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. PHÂN TÍCH CẤU TRÚC THƯ MỤC
def analyze_directory_structure():
    """Phân tích cấu trúc thư mục dữ liệu"""
    print("1. PHÂN TÍCH CẤU TRÚC THƯ MỤC")
    
    results = {}
    
    for split_name, split_dir in [("Train", TRAIN_DIR), ("Test", TEST_DIR)]:
        print(f"\n{split_name} Directory: {split_dir}")
        class_counts = {}
        
        if not os.path.exists(split_dir):
            print(f"Thư mục không tồn tại!")
            continue
            
        subdirs = sorted(os.listdir(split_dir))
        print(f"Số lượng thư mục con: {len(subdirs)}")
        
        total = 0
        for subdir in subdirs:
            subdir_path = os.path.join(split_dir, subdir)
            if os.path.isdir(subdir_path):
                count = len([f for f in os.listdir(subdir_path) 
                           if os.path.isfile(os.path.join(subdir_path, f))])
                emotion_name = EMOTION_MAP.get(int(subdir), "Unknown")
                print(f"{subdir}/ ({emotion_name}): {count:,} files")
                class_counts[int(subdir)] = count
                total += count
        
        print(f"Tổng: {total:,} files")
        results[split_name.lower()] = class_counts
    
    return results

# 1.5 TẠO CSV NẾU CHƯA CÓ
def ensure_csv_files_exist():
    """Tự động tạo file CSV từ cấu trúc thư mục nếu chưa tồn tại"""
    for split, csv_path in [("train", TRAIN_CSV), ("test", TEST_CSV)]:
        if os.path.exists(csv_path):
            continue
            
        split_dir = os.path.join(DATA_DIR, split)
        if not os.path.exists(split_dir):
            continue
            
        print(f"Đang tự động tạo {csv_path} từ thư mục...")
        data = []
        for class_id in os.listdir(split_dir):
            class_path = os.path.join(split_dir, class_id)
            if not os.path.isdir(class_path): continue
            
            for img_name in os.listdir(class_path):
                if img_name.endswith(('.jpg', '.jpeg', '.png')):
                    data.append({"filename": img_name, "label": int(class_id)})
                    
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        print(f"Đã tạo {csv_path} với {len(df)} dòng.")

# 2. PHÂN TÍCH CSV LABELS
def analyze_csv_labels():
    """Phân tích file CSV labels"""
    print("2. PHÂN TÍCH CSV LABELS")
    
    results = {}
    
    for csv_name, csv_path in [("Train", TRAIN_CSV), ("Test", TEST_CSV)]:
        print(f"\n{csv_name} CSV: {csv_path}")
        
        if not os.path.exists(csv_path):
            print(f"File không tồn tại!")
            continue
        
        df = pd.read_csv(csv_path)
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {df.columns.tolist()}")
        print(f"   Dtypes:\n{df.dtypes.to_string()}")
        print(f"\n   Sample (5 dòng đầu):")
        print(f"{df.head().to_string()}")
        
        # Phân bố nhãn
        label_col = df.columns[-1]  # Cột nhãn
        label_counts = df[label_col].value_counts().sort_index()
        print(f"\n   Phân bố nhãn:")
        for label, count in label_counts.items():
            pct = count / len(df) * 100
            emotion = EMOTION_MAP.get(label, "Unknown")
            bar = " " * int(pct)
            print(f"   {label} ({emotion:10s}): {count:5d} ({pct:5.1f}%) {bar}")
        
        # Kiểm tra missing values
        missing = df.isnull().sum()
        print(f"\n   Missing values: {missing.sum()}")
        
        # Kiểm tra duplicates
        dups = df.duplicated().sum()
        print(f"   Duplicate rows: {dups}")
        
        results[csv_name.lower()] = df
    
    return results



# 3. KIỂM TRA TÍNH NHẤT QUÁN CSV vs THƯ MỤC
def check_consistency(csv_data, dir_data):
    """Kiểm tra sự nhất quán giữa CSV labels và thư mục ảnh"""
    print("3. KIỂM TRA TÍNH NHẤT QUÁN CSV vs THƯ MỤC")
    
    for split in ["train", "test"]:
        if split not in csv_data or split not in dir_data:
            continue
            
        df = csv_data[split]
        dir_counts = dir_data[split]
        
        print(f"\n{split.upper()} set:")
        
        # So sánh số lượng
        csv_total = len(df)
        dir_total = sum(dir_counts.values())
        print(f"   CSV entries: {csv_total}")
        print(f"   Directory files: {dir_total}")
        
        if csv_total != dir_total:
            print(f"KHÁC BIỆT: CSV ({csv_total}) ≠ Directory ({dir_total})")
        else:
            print(f"SỐ LƯỢNG KHỚP")
        
        # So sánh theo từng class
        label_col = df.columns[-1]
        csv_counts = df[label_col].value_counts().sort_index()
        
        print(f"\n   So sánh chi tiết theo class:")
        print(f"   {'Label':<8} {'CSV':>8} {'Dir':>8} {'Match':>8}")
        print(f"   {'-'*36}")
        for label in sorted(set(list(csv_counts.index) + list(dir_counts.keys()))):
            csv_c = csv_counts.get(label, 0)
            dir_c = dir_counts.get(label, 0)
            match = "y" if csv_c == dir_c else "n"
            print(f"   {label:<8} {csv_c:>8} {dir_c:>8} {match:>8}")


# 4. KIỂM TRA ẢNH LỖI / CORRUPT
def check_corrupted_images():
    """Quét toàn bộ ảnh, phát hiện file lỗi"""
    print("4. KIỂM TRA ẢNH LỖI / CORRUPT")
    
    corrupted = []
    shapes_set = set()
    total_checked = 0
    
    for split_name, split_dir in [("train", TRAIN_DIR), ("test", TEST_DIR)]:
        print(f"\nChecking {split_name}...")
        
        if not os.path.exists(split_dir):
            continue
        
        for class_dir in sorted(os.listdir(split_dir)):
            class_path = os.path.join(split_dir, class_dir)
            if not os.path.isdir(class_path):
                continue
            
            files = os.listdir(class_path)
            for fname in files:
                fpath = os.path.join(class_path, fname)
                total_checked += 1
                
                try:
                    img = cv2.imread(fpath)
                    if img is None:
                        corrupted.append(fpath)
                    else:
                        shapes_set.add(img.shape)
                except Exception as e:
                    corrupted.append(fpath)
    
    print(f"\nKết quả:")
    print(f"   Tổng ảnh kiểm tra: {total_checked:,}")
    print(f"   Ảnh lỗi: {len(corrupted)}")
    print(f"   Unique shapes: {shapes_set}")
    
    if corrupted:
        print(f"\nDanh sách ảnh lỗi:")
        for path in corrupted[:20]:
            print(f"      - {path}")
        if len(corrupted) > 20:
            print(f"      ... và {len(corrupted) - 20} ảnh khác")
    else:
        print(f"Không có ảnh lỗi!")
    
    return corrupted, shapes_set

# 5. THỐNG KÊ PIXEL
def compute_pixel_statistics():
    """Tính mean, std của pixel trên toàn bộ dataset"""
    print("5. THỐNG KÊ PIXEL (MEAN, STD)")
    
    pixel_sum = np.zeros(3, dtype=np.float64)
    pixel_sq_sum = np.zeros(3, dtype=np.float64)
    total_pixels = 0
    
    min_vals = np.array([255.0, 255.0, 255.0])
    max_vals = np.array([0.0, 0.0, 0.0])
    
    # Histogram data (lấy sample 2000 ảnh để tính nhanh)
    all_intensities = []
    sample_count = 0
    
    for split_dir in [TRAIN_DIR, TEST_DIR]:
        if not os.path.exists(split_dir):
            continue
        for class_dir in sorted(os.listdir(split_dir)):
            class_path = os.path.join(split_dir, class_dir)
            if not os.path.isdir(class_path):
                continue
            for fname in os.listdir(class_path):
                fpath = os.path.join(class_path, fname)
                img = cv2.imread(fpath)
                if img is None:
                    continue
                
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_float = img_rgb.astype(np.float64)
                
                n_pixels = img_float.shape[0] * img_float.shape[1]
                pixel_sum += img_float.reshape(-1, 3).sum(axis=0)
                pixel_sq_sum += (img_float.reshape(-1, 3) ** 2).sum(axis=0)
                total_pixels += n_pixels
                
                min_vals = np.minimum(min_vals, img_float.reshape(-1, 3).min(axis=0))
                max_vals = np.maximum(max_vals, img_float.reshape(-1, 3).max(axis=0))
                
                # Sample cho histogram
                if sample_count < 2000:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    all_intensities.extend(gray.flatten().tolist()[::10])  # subsample
                    sample_count += 1
    
    mean = pixel_sum / total_pixels
    std = np.sqrt(pixel_sq_sum / total_pixels - mean ** 2)
    
    print(f"\nPixel Statistics (RGB, range 0-255):")
    print(f"   Mean: R={mean[0]:.2f}, G={mean[1]:.2f}, B={mean[2]:.2f}")
    print(f"   Std:  R={std[0]:.2f}, G={std[1]:.2f}, B={std[2]:.2f}")
    print(f"   Min:  R={min_vals[0]:.0f}, G={min_vals[1]:.0f}, B={min_vals[2]:.0f}")
    print(f"   Max:  R={max_vals[0]:.0f}, G={max_vals[1]:.0f}, B={max_vals[2]:.0f}")
    
    print(f"\nNormalized (0-1):")
    print(f"   Mean: R={mean[0]/255:.4f}, G={mean[1]/255:.4f}, B={mean[2]/255:.4f}")
    print(f"   Std:  R={std[0]/255:.4f}, G={std[1]/255:.4f}, B={std[2]/255:.4f}")
    
    return mean, std, all_intensities

# 6. TRỰC QUAN HÓA — BIỂU ĐỒ PHÂN BỐ NHÃN
def plot_label_distribution(csv_data):
    """Vẽ bar chart phân bố nhãn cho train và test"""
    print("6. TRỰC QUAN HÓA — BIỂU ĐỒ PHÂN BỐ NHÃN")
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("RAF-DB: Phân bố Nhãn Cảm xúc", fontsize=18, fontweight='bold', y=1.02)
    
    for idx, (split, ax) in enumerate(zip(["train", "test"], axes)):
        if split not in csv_data:
            continue
        
        df = csv_data[split]
        label_col = df.columns[-1]
        counts = df[label_col].value_counts().sort_index()
        
        labels = [f"{EMOTION_MAP[i]}\n({EMOTION_MAP_VI[i]})" for i in counts.index]
        colors = [EMOTION_COLORS[i] for i in counts.index]
        
        bars = ax.bar(labels, counts.values, color=colors, edgecolor='white', linewidth=1.5, alpha=0.9)
        
        # Thêm số liệu lên đầu bar
        for bar, val in zip(bars, counts.values):
            pct = val / len(df) * 100
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
                    f'{val:,}\n({pct:.1f}%)',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_title(f'{split.upper()} Set ({len(df):,} ảnh)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Số lượng ảnh', fontsize=12)
        ax.set_xlabel('Cảm xúc', fontsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim(0, max(counts.values) * 1.25)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "label_distribution.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")

# 7. TRỰC QUAN HÓA — HISTOGRAM CƯỜNG ĐỘ PIXEL
def plot_pixel_histogram(all_intensities):
    """Vẽ histogram cường độ pixel"""
    print("7. TRỰC QUAN HÓA — HISTOGRAM CƯỜNG ĐỘ PIXEL")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.hist(all_intensities, bins=64, color='#4A90D9', edgecolor='white',
            alpha=0.8, density=True)
    ax.set_title('Phân bố Cường độ Pixel (Grayscale)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Cường độ pixel (0-255)', fontsize=13)
    ax.set_ylabel('Mật độ (Density)', fontsize=13)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Thêm mean line
    mean_val = np.mean(all_intensities)
    ax.axvline(mean_val, color='#FF4444', linestyle='--', linewidth=2, label=f'Mean = {mean_val:.1f}')
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "pixel_histogram.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")

# 8. TRỰC QUAN HÓA — RGB CHANNEL HISTOGRAM
def plot_rgb_histogram():
    """Vẽ histogram cho từng kênh R, G, B"""
    print("8. TRỰC QUAN HÓA — RGB CHANNEL HISTOGRAM")
    
    r_vals, g_vals, b_vals = [], [], []
    count = 0
    
    for class_dir in sorted(os.listdir(TRAIN_DIR)):
        class_path = os.path.join(TRAIN_DIR, class_dir)
        if not os.path.isdir(class_path):
            continue
        for fname in os.listdir(class_path)[:50]:  # Sample 50 ảnh/class
            img = cv2.imread(os.path.join(class_path, fname))
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            r_vals.extend(img_rgb[:, :, 0].flatten().tolist()[::5])
            g_vals.extend(img_rgb[:, :, 1].flatten().tolist()[::5])
            b_vals.extend(img_rgb[:, :, 2].flatten().tolist()[::5])
            count += 1
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(r_vals, bins=64, color='red', alpha=0.4, density=True, label='Red')
    ax.hist(g_vals, bins=64, color='green', alpha=0.4, density=True, label='Green')
    ax.hist(b_vals, bins=64, color='blue', alpha=0.4, density=True, label='Blue')
    
    ax.set_title('Phân bố RGB Channels', fontsize=16, fontweight='bold')
    ax.set_xlabel('Giá trị pixel (0-255)', fontsize=13)
    ax.set_ylabel('Mật độ (Density)', fontsize=13)
    ax.legend(fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "rgb_histogram.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")

# 9. TRỰC QUAN HÓA — SAMPLE IMAGES PER CLASS
def plot_sample_images(samples_per_class=5):
    """Hiển thị mẫu ảnh cho mỗi class cảm xúc"""
    print("9. TRỰC QUAN HÓA — MẪU DỮ LIỆU THEO TỪNG CLASS")
    
    n_classes = 7
    fig, axes = plt.subplots(n_classes, samples_per_class, 
                              figsize=(samples_per_class * 2.5, n_classes * 2.8))
    fig.suptitle('RAF-DB: Mẫu dữ liệu theo từng cảm xúc', 
                 fontsize=18, fontweight='bold', y=1.01)
    
    for class_idx in range(1, n_classes + 1):
        class_dir = os.path.join(TRAIN_DIR, str(class_idx))
        if not os.path.exists(class_dir):
            continue
        
        files = sorted(os.listdir(class_dir))
        # Lấy ngẫu nhiên
        np.random.seed(42)
        selected = np.random.choice(files, size=min(samples_per_class, len(files)), replace=False)
        
        for col_idx, fname in enumerate(selected):
            ax = axes[class_idx - 1][col_idx]
            img = cv2.imread(os.path.join(class_dir, fname))
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax.imshow(img_rgb)
            
            ax.axis('off')
            
            # Title cho cột đầu tiên
            if col_idx == 0:
                emotion_en = EMOTION_MAP[class_idx]
                emotion_vi = EMOTION_MAP_VI[class_idx]
                ax.set_ylabel(f'{emotion_en}\n({emotion_vi})', 
                            fontsize=11, fontweight='bold', rotation=0,
                            labelpad=80, va='center')
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "sample_images_per_class.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")

# 10. TRỰC QUAN HÓA — MEAN FACE PER CLASS
def plot_mean_faces():
    """Tính và hiển thị khuôn mặt trung bình cho mỗi class"""
    print("10. TRỰC QUAN HÓA — KHUÔN MẶT TRUNG BÌNH (MEAN FACE)")
    
    fig, axes = plt.subplots(1, 7, figsize=(21, 4))
    fig.suptitle('Khuôn mặt Trung bình theo từng Cảm xúc', 
                 fontsize=16, fontweight='bold')
    
    for class_idx in range(1, 8):
        class_dir = os.path.join(TRAIN_DIR, str(class_idx))
        if not os.path.exists(class_dir):
            continue
        
        files = os.listdir(class_dir)
        mean_face = np.zeros((100, 100, 3), dtype=np.float64)
        count = 0
        
        for fname in files:
            img = cv2.imread(os.path.join(class_dir, fname))
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                mean_face += img_rgb.astype(np.float64)
                count += 1
        
        if count > 0:
            mean_face /= count
            mean_face = mean_face.astype(np.uint8)
        
        ax = axes[class_idx - 1]
        ax.imshow(mean_face)
        ax.set_title(f'{EMOTION_MAP[class_idx]}\n({count} ảnh)', fontsize=10, fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "mean_faces.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")

# 11. TRỰC QUAN HÓA — IMBALANCE RATIO
def plot_imbalance_analysis(csv_data):
    """Phân tích và trực quan mức độ mất cân bằng"""
    print("11. PHÂN TÍCH MẤT CÂN BẰNG DỮ LIỆU")
    
    if "train" not in csv_data:
        return
    
    df = csv_data["train"]
    label_col = df.columns[-1]
    counts = df[label_col].value_counts().sort_index()
    
    max_count = counts.max()
    min_count = counts.min()
    imbalance_ratio = max_count / min_count
    
    print(f"\n   Class lớn nhất: {EMOTION_MAP[counts.idxmax()]} = {max_count:,}")
    print(f"   Class nhỏ nhất: {EMOTION_MAP[counts.idxmin()]} = {min_count:,}")
    print(f"   Imbalance Ratio (max/min): {imbalance_ratio:.1f}:1")
    
    # Pie chart
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # Pie chart
    labels = [f"{EMOTION_MAP[i]}\n{EMOTION_MAP_VI[i]}" for i in counts.index]
    colors = [EMOTION_COLORS[i] for i in counts.index]
    
    wedges, texts, autotexts = axes[0].pie(counts.values, labels=labels, 
                                            autopct='%1.1f%%', colors=colors,
                                            textprops={'fontsize': 10},
                                            pctdistance=0.75,
                                            startangle=90)
    for autotext in autotexts:
        autotext.set_fontweight('bold')
    axes[0].set_title('Tỉ lệ phân bố các cảm xúc (Train)', fontsize=14, fontweight='bold')
    
    # Imbalance ratio bar
    ratios = [(max_count / c) for c in counts.values]
    ratio_labels = [f"{EMOTION_MAP[i]}" for i in counts.index]
    bars_colors = [EMOTION_COLORS[i] for i in counts.index]
    
    bars = axes[1].barh(ratio_labels, ratios, color=bars_colors, edgecolor='white', alpha=0.9)
    axes[1].set_xlabel('Tỉ lệ (so với class lớn nhất)', fontsize=12)
    axes[1].set_title('Mức độ Mất cân bằng (Imbalance Ratio)', fontsize=14, fontweight='bold')
    axes[1].axvline(x=1, color='green', linestyle='--', linewidth=2, label='Balanced (1:1)')
    axes[1].legend(fontsize=11)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    
    for bar, ratio in zip(bars, ratios):
        axes[1].text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                    f'{ratio:.1f}x', va='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "imbalance_analysis.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")

# 12. TÓM TẮT EDA
def save_eda_summary(csv_data, corrupted, shapes_set, mean, std):
    """Lưu tóm tắt EDA thành file text"""
    summary_path = os.path.join(OUTPUT_DIR, "eda_summary.txt")
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("TÓM TẮT EDA — RAF-DB DATASET\n")
        
        f.write("1. TỔNG QUAN\n")
        f.write(f"   - Dataset: RAF-DB (Real-world Affective Faces Database)\n")
        f.write(f"   - Phiên bản: Aligned faces\n")
        
        if "train" in csv_data:
            f.write(f"   - Train set: {len(csv_data['train']):,} ảnh\n")
        if "test" in csv_data:
            f.write(f"   - Test set: {len(csv_data['test']):,} ảnh\n")
            f.write(f"   - Tổng: {len(csv_data.get('train', [])) + len(csv_data.get('test', [])):,} ảnh\n")
        
        f.write(f"\n2. ĐẶC TẢ ẢNH\n")
        f.write(f"   - Loại: Ảnh màu RGB (3 kênh)\n")
        f.write(f"   - Định dạng: JPEG (.jpg)\n")
        f.write(f"   - Kích thước: {shapes_set}\n")
        f.write(f"   - Dtype: uint8\n")
        
        f.write(f"\n3. THỐNG KÊ PIXEL\n")
        f.write(f"   - Mean (RGB): [{mean[0]:.2f}, {mean[1]:.2f}, {mean[2]:.2f}]\n")
        f.write(f"   - Std (RGB):  [{std[0]:.2f}, {std[1]:.2f}, {std[2]:.2f}]\n")
        f.write(f"   - Mean (normalized): [{mean[0]/255:.4f}, {mean[1]/255:.4f}, {mean[2]/255:.4f}]\n")
        f.write(f"   - Std (normalized):  [{std[0]/255:.4f}, {std[1]/255:.4f}, {std[2]/255:.4f}]\n")
        
        f.write(f"\n4. CHẤT LƯỢNG DỮ LIỆU\n")
        f.write(f"   - Ảnh lỗi/corrupt: {len(corrupted)}\n")
        
        if "train" in csv_data:
            df = csv_data["train"]
            label_col = df.columns[-1]
            counts = df[label_col].value_counts().sort_index()
            
            f.write(f"\n5. PHÂN BỐ NHÃN (TRAIN)\n")
            for label, count in counts.items():
                pct = count / len(df) * 100
                f.write(f"   {label} ({EMOTION_MAP[label]:10s}): {count:5d} ({pct:5.1f}%)\n")
            
            f.write(f"\n6. MẤT CÂN BẰNG\n")
            f.write(f"   - Class lớn nhất: {EMOTION_MAP[counts.idxmax()]} = {counts.max():,}\n")
            f.write(f"   - Class nhỏ nhất: {EMOTION_MAP[counts.idxmin()]} = {counts.min():,}\n")
            f.write(f"   - Imbalance ratio: {counts.max()/counts.min():.1f}:1\n")
        
        f.write(f"\n7. CÁC BIỂU ĐỒ ĐÃ TẠO\n")
        f.write(f"   - label_distribution.png\n")
        f.write(f"   - pixel_histogram.png\n")
        f.write(f"   - rgb_histogram.png\n")
        f.write(f"   - sample_images_per_class.png\n")
        f.write(f"   - mean_faces.png\n")
        f.write(f"   - imbalance_analysis.png\n")
    
    print(f"\nSaved EDA summary: {summary_path}")

# MAIN
def main():
    print("EDA — RAF-DB Dataset (Facial Emotion Recognition)")
    
    # 1. Phân tích cấu trúc thư mục
    dir_data = analyze_directory_structure()
    
    # Đảm bảo CSV tồn tại trước khi phân tích
    ensure_csv_files_exist()
    
    # 2. Phân tích CSV labels
    csv_data = analyze_csv_labels()
    
    # 3. Kiểm tra tính nhất quán
    check_consistency(csv_data, dir_data)
    
    # 4. Kiểm tra ảnh lỗi
    corrupted, shapes_set = check_corrupted_images()
    
    # 5. Thống kê pixel
    mean, std, all_intensities = compute_pixel_statistics()
    
    # 6. Biểu đồ phân bố nhãn
    plot_label_distribution(csv_data)
    
    # 7. Histogram cường độ pixel
    plot_pixel_histogram(all_intensities)
    
    # 8. RGB histogram
    plot_rgb_histogram()
    
    # 9. Sample images per class
    plot_sample_images(samples_per_class=6)
    
    # 10. Mean faces
    plot_mean_faces()
    
    # 11. Phân tích mất cân bằng
    plot_imbalance_analysis(csv_data)
    
    # 12. Lưu tóm tắt
    save_eda_summary(csv_data, corrupted, shapes_set, mean, std)

    print("Kết quả được lưu tại: outputs/eda/")

if __name__ == "__main__":
    main()
