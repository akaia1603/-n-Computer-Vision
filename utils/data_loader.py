"""
utils/data_loader.py
====================
Module quản lý dữ liệu cho dự án Nhận dạng Cảm xúc Khuôn mặt (FER).

Chức năng chính:
    - Load ảnh từ thư mục RAF-DB theo cấu trúc class folders
    - Tạo tf.data.Dataset với augmentation pipeline
    - Tính class weights để xử lý mất cân bằng dữ liệu
    - Hỗ trợ cả input size 100x100 (baseline) và 224x224 (pretrained)

Sử dụng:
    >>> from utils.data_loader import create_datasets
    >>> train_ds, val_ds, test_ds, class_weights = create_datasets(
    ...     data_dir='data', img_size=(100, 100), batch_size=32
    ... )

Yêu cầu: tensorflow, numpy, scikit-learn
"""

import os
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
import cv2


# ============================================================
# HẰNG SỐ
# ============================================================

# Mapping label ID → tên cảm xúc (tiếng Anh)
EMOTION_MAP = {
    1: "Surprise",
    2: "Fear",
    3: "Disgust",
    4: "Happiness",
    5: "Sadness",
    6: "Anger",
    7: "Neutral"
}



# Số lớp cảm xúc
NUM_CLASSES = 7

# Màu sắc cho visualize
EMOTION_COLORS = {
    1: "#FF6B6B",   # Surprise - đỏ san hô
    2: "#845EC2",   # Fear - tím
    3: "#4B8B3B",   # Disgust - xanh rêu
    4: "#FFD93D",   # Happiness - vàng
    5: "#4A90D9",   # Sadness - xanh dương
    6: "#FF4444",   # Anger - đỏ
    7: "#95A5A6",   # Neutral - xám
}

# ImageNet normalization (cho pretrained models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# RAF-DB normalization (calculated from training set)
RAF_MEAN = [146.6770, 114.6274, 102.3102]
RAF_STD = [67.6282, 61.7651, 61.3665]

def preprocess_raf_tf(image):
    """Tiền xử lý ảnh (TensorFlow) dùng mean/std của RAF-DB."""
    mean = tf.constant(RAF_MEAN, dtype=tf.float32)
    std = tf.constant(RAF_STD, dtype=tf.float32)
    return (image - mean) / std

def preprocess_raf_np(image):
    """Tiền xử lý ảnh (Numpy) dùng mean/std của RAF-DB."""
    image = np.array(image, dtype='float32')
    image[..., 0] = (image[..., 0] - RAF_MEAN[0]) / RAF_STD[0]
    image[..., 1] = (image[..., 1] - RAF_MEAN[1]) / RAF_STD[1]
    image[..., 2] = (image[..., 2] - RAF_MEAN[2]) / RAF_STD[2]
    return image


# ============================================================
# HÀM LOAD DỮ LIỆU TỪ THƯ MỤC
# ============================================================

def load_image_paths_and_labels(data_dir, split="train"):
    """
    Đọc đường dẫn ảnh và nhãn từ thư mục.
    
    Cấu trúc thư mục kỳ vọng:
        data_dir/
        ├── train/
        │   ├── 1/  (Surprise)
        │   ├── 2/  (Fear)
        │   ...
        │   └── 7/  (Neutral)
        └── test/
            ├── 1/ ... 7/
    
    Args:
        data_dir (str): Đường dẫn thư mục gốc chứa data
        split (str): "train" hoặc "test"
    
    Returns:
        image_paths (list[str]): Danh sách đường dẫn ảnh
        labels (list[int]): Nhãn tương ứng (0-indexed: 0-6)
    
    Lưu ý:
        - Nhãn được chuyển từ 1-indexed (RAF-DB) sang 0-indexed (model)
        - Label 1 (Surprise) → 0, Label 7 (Neutral) → 6
    """
    split_dir = os.path.join(data_dir, split)
    image_paths = []
    labels = []
    
    if not os.path.exists(split_dir):
        raise FileNotFoundError(f"Thư mục không tồn tại: {split_dir}")
    
    for class_folder in sorted(os.listdir(split_dir)):
        class_path = os.path.join(split_dir, class_folder)
        if not os.path.isdir(class_path):
            continue
        
        try:
            class_label = int(class_folder) - 1  # Chuyển sang 0-indexed
        except ValueError:
            continue
        
        for fname in os.listdir(class_path):
            fpath = os.path.join(class_path, fname)
            if os.path.isfile(fpath) and fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(fpath)
                labels.append(class_label)
    
    print(f"  📁 Loaded {split}: {len(image_paths):,} ảnh, {len(set(labels))} classes")
    return image_paths, labels


# ============================================================
# TÍNH CLASS WEIGHTS
# ============================================================

def compute_class_weights(labels):
    """
    Tính class weights để xử lý mất cân bằng dữ liệu.
    
    Sử dụng phương pháp 'balanced' từ scikit-learn:
        weight_i = n_samples / (n_classes * n_samples_i)
    
    Class ít mẫu hơn sẽ có weight lớn hơn, giúp model
    không bị thiên lệch về class chiếm đa số.
    
    Args:
        labels (list[int]): Danh sách nhãn (0-indexed)
    
    Returns:
        class_weight_dict (dict): {class_id: weight}
    
    Ví dụ:
        >>> weights = compute_class_weights([0, 0, 0, 1, 2])
        >>> # Class 0 (nhiều mẫu) → weight thấp
        >>> # Class 1, 2 (ít mẫu) → weight cao
    """
    unique_classes = np.unique(labels)
    weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=labels
    )
    class_weight_dict = dict(zip(unique_classes.astype(int), weights))
    
    print("  ⚖️ Class Weights:")
    for cls, w in sorted(class_weight_dict.items()):
        emotion = EMOTION_MAP.get(cls + 1, "Unknown")
        print(f"     {cls} ({emotion:10s}): {w:.4f}")
    
    return class_weight_dict


# ============================================================
# PREPROCESSING & AUGMENTATION
# ============================================================

def preprocess_image(image_path, label, img_size=(100, 100), augment=False):
    """
    Đọc và tiền xử lý một ảnh.
    
    Pipeline:
        1. Đọc file ảnh → decode JPEG
        2. Resize về kích thước chuẩn
        3. Normalize pixel values về [0, 1]
        4. (Optional) Áp dụng augmentation
    
    Args:
        image_path: Đường dẫn ảnh (tf.string)
        label: Nhãn (tf.int32)
        img_size: Kích thước output (height, width)
        augment: Có áp dụng augmentation không
    
    Returns:
        image: Tensor ảnh đã tiền xử lý [H, W, 3]
        label: Nhãn giữ nguyên
    """
    # Đọc file
    img_raw = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(img_raw, channels=3)
    
    # Resize
    image = tf.image.resize(image, img_size)
    
    # Normalize tạm về [0, 1] để augmentation hoạt động tốt
    image = tf.cast(image, tf.float32) / 255.0
    
    # Augmentation (chỉ áp dụng khi training)
    if augment:
        image = apply_augmentation(image)
        
    # Scale lại [0, 255] và chuẩn hóa theo RAF-DB
    image = image * 255.0
    image = preprocess_raf_tf(image)
    
    return image, label


def apply_augmentation(image):
    """
    Áp dụng data augmentation ngẫu nhiên.
    
    Các phép augmentation được chọn dựa trên đặc điểm bài toán FER:
    
    1. Random Horizontal Flip (50%):
       - Biểu cảm khuôn mặt đối xứng → flip ngang hợp lý
       - KHÔNG flip dọc (khuôn mặt bị lộn ngược → vô nghĩa)
    
    2. Random Brightness (±20%):
       - Mô phỏng điều kiện ánh sáng khác nhau
       - RAF-DB thu thập in-the-wild nên cần robust với ánh sáng
    
    3. Random Contrast (±20%):
       - Tăng/giảm độ tương phản
       - Giúp model học features ổn định hơn
    
    4. Random Rotation (±15°):
       - Khuôn mặt có thể hơi nghiêng
       - Giới hạn ±15° để giữ tính tự nhiên
    
    5. Random Zoom (±10%):
       - Mô phỏng khoảng cách camera khác nhau
    
    Args:
        image: Tensor ảnh [H, W, 3], range [0, 1]
    
    Returns:
        image: Tensor ảnh đã augment, clip về [0, 1]
    """
    # 1. Random horizontal flip
    image = tf.image.random_flip_left_right(image)
    
    # 2. Random brightness
    image = tf.image.random_brightness(image, max_delta=0.2)
    
    # 3. Random contrast
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    
    # 4. Random saturation
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    
    # 5. Random hue (nhẹ)
    image = tf.image.random_hue(image, max_delta=0.05)
    
    # Đảm bảo giá trị pixel trong [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)
    
    return image


# ============================================================
# TẠO TF.DATA.DATASET
# ============================================================

def create_datasets(data_dir="data", img_size=(100, 100), batch_size=32,
                    val_split=0.2, seed=42, augment_train=True):
    """
    Tạo tf.data.Dataset cho train, validation và test.
    
    Pipeline xử lý dữ liệu:
        1. Load đường dẫn ảnh + labels từ thư mục
        2. Split train → train + validation (stratified)
        3. Tạo tf.data.Dataset cho mỗi split
        4. Áp dụng augmentation cho train set
        5. Tính class weights từ train set
    
    Args:
        data_dir (str): Đường dẫn thư mục gốc ('data')
        img_size (tuple): Kích thước ảnh (H, W)
            - (100, 100) cho baseline CNN
            - (224, 224) cho pretrained models (ResNet50, MobileNetV2)
        batch_size (int): Batch size cho training
        val_split (float): Tỷ lệ validation (0.2 = 20%)
        seed (int): Random seed cho reproducibility
        augment_train (bool): Có augmentation cho train set không
    
    Returns:
        train_ds (tf.data.Dataset): Dataset train (augmented)
        val_ds (tf.data.Dataset): Dataset validation (không augment)
        test_ds (tf.data.Dataset): Dataset test (không augment)
        class_weights (dict): Class weights cho training
        info (dict): Thông tin bổ sung (num samples, etc.)
    
    Ví dụ:
        >>> train_ds, val_ds, test_ds, weights, info = create_datasets(
        ...     img_size=(224, 224), batch_size=16
        ... )
        >>> for images, labels in train_ds.take(1):
        ...     print(images.shape)  # (16, 224, 224, 3)
    """
    print("=" * 60)
    print("📦 LOADING DATASETS")
    print("=" * 60)
    
    # 1. Load paths và labels
    train_paths, train_labels = load_image_paths_and_labels(data_dir, "train")
    test_paths, test_labels = load_image_paths_and_labels(data_dir, "test")
    
    # 2. Split train → train + validation (stratified)
    train_paths_split, val_paths, train_labels_split, val_labels = train_test_split(
        train_paths, train_labels,
        test_size=val_split,
        random_state=seed,
        stratify=train_labels  # Giữ tỷ lệ class giống nhau
    )
    
    print(f"\n  📊 Split Summary:")
    print(f"     Train:      {len(train_paths_split):,} ảnh")
    print(f"     Validation: {len(val_paths):,} ảnh")
    print(f"     Test:       {len(test_paths):,} ảnh")
    
    # 3. Tính class weights từ train set
    class_weights = compute_class_weights(train_labels_split)
    
    # 4. Tạo datasets
    AUTOTUNE = tf.data.AUTOTUNE
    
    # --- Train Dataset (có augmentation) ---
    def parse_train(path, label):
        return preprocess_image(path, label, img_size=img_size, augment=augment_train)
    
    def parse_eval(path, label):
        return preprocess_image(path, label, img_size=img_size, augment=False)
    
    train_ds = tf.data.Dataset.from_tensor_slices(
        (train_paths_split, train_labels_split)
    )
    train_ds = (train_ds
                .shuffle(buffer_size=len(train_paths_split), seed=seed)
                .map(parse_train, num_parallel_calls=AUTOTUNE)
                .batch(batch_size)
                .prefetch(AUTOTUNE))
    
    # --- Validation Dataset (không augmentation) ---
    val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
    val_ds = (val_ds
              .map(parse_eval, num_parallel_calls=AUTOTUNE)
              .batch(batch_size)
              .prefetch(AUTOTUNE))
    
    # --- Test Dataset (không augmentation) ---
    test_ds = tf.data.Dataset.from_tensor_slices((test_paths, test_labels))
    test_ds = (test_ds
               .map(parse_eval, num_parallel_calls=AUTOTUNE)
               .batch(batch_size)
               .prefetch(AUTOTUNE))
    
    # 5. Thông tin bổ sung
    info = {
        "num_train": len(train_paths_split),
        "num_val": len(val_paths),
        "num_test": len(test_paths),
        "num_classes": NUM_CLASSES,
        "img_size": img_size,
        "batch_size": batch_size,
        "train_labels": train_labels_split,
        "val_labels": val_labels,
        "test_labels": test_labels,
        "test_paths": test_paths,
    }
    
    print(f"\n  ✅ Datasets created successfully!")
    print(f"     Image size: {img_size}")
    print(f"     Batch size: {batch_size}")
    
    return train_ds, val_ds, test_ds, class_weights, info


# ============================================================
# TIỆN ÍCH: LOAD ẢNH ĐƠN LẺ (dùng cho inference/demo)
# ============================================================

def load_single_image(image_path, img_size=(100, 100)):
    """
    Load và tiền xử lý một ảnh đơn lẻ cho inference.
    
    Args:
        image_path (str): Đường dẫn ảnh
        img_size (tuple): Kích thước resize
    
    Returns:
        image (np.ndarray): Ảnh [1, H, W, 3], normalized [0,1]
        image_original (np.ndarray): Ảnh gốc BGR (cho visualization)
    """
    # Đọc ảnh gốc
    image_original = cv2.imread(image_path)
    if image_original is None:
        raise ValueError(f"Không thể đọc ảnh: {image_path}")
    
    # Chuyển BGR → RGB
    image_rgb = cv2.cvtColor(image_original, cv2.COLOR_BGR2RGB)
    
    # Resize
    image_resized = cv2.resize(image_rgb, img_size)
    
    # Normalize
    image_normalized = preprocess_raf_np(image_resized)
    
    # Thêm batch dimension
    image_batch = np.expand_dims(image_normalized, axis=0)
    
    return image_batch, image_original


def load_image_from_array(image_array, img_size=(100, 100)):
    """
    Tiền xử lý ảnh từ numpy array (dùng cho webcam frame).
    
    Args:
        image_array (np.ndarray): Ảnh RGB [H, W, 3], uint8
        img_size (tuple): Kích thước resize
    
    Returns:
        image (np.ndarray): Ảnh [1, H, W, 3], normalized [0,1]
    """
    image_resized = cv2.resize(image_array, img_size)
    image_normalized = preprocess_raf_np(image_resized)
    return np.expand_dims(image_normalized, axis=0)


# ============================================================
# PREVIEW AUGMENTATION (cho EDA)
# ============================================================

def preview_augmentation(image_path, img_size=(100, 100), n_augments=8):
    """
    Tạo preview các phép augmentation cho một ảnh.
    Dùng trong EDA để minh họa augmentation pipeline.
    
    Args:
        image_path (str): Đường dẫn ảnh gốc
        img_size (tuple): Kích thước resize
        n_augments (int): Số phiên bản augmented
    
    Returns:
        original (np.ndarray): Ảnh gốc [H, W, 3]
        augmented_images (list[np.ndarray]): Danh sách ảnh augmented
    """
    # Load ảnh gốc
    img_raw = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(img_raw, channels=3)
    image = tf.image.resize(image, img_size)
    image = tf.cast(image, tf.float32) / 255.0
    
    original = image.numpy()
    augmented_images = []
    
    for _ in range(n_augments):
        aug_image = apply_augmentation(image)
        # Apply RAF normalization for viewing (scale back to [0, 1] manually or leave as is)
        # Here we just leave it in [0, 1] for previewing purposes
        augmented_images.append(aug_image.numpy())
    
    return original, augmented_images


if __name__ == "__main__":
    # Quick test
    print("Testing data_loader...")
    train_ds, val_ds, test_ds, weights, info = create_datasets(
        data_dir="data", img_size=(100, 100), batch_size=32
    )
    
    for batch_images, batch_labels in train_ds.take(1):
        print(f"\nBatch shape: {batch_images.shape}")
        print(f"Labels shape: {batch_labels.shape}")
        print(f"Pixel range: [{batch_images.numpy().min():.3f}, {batch_images.numpy().max():.3f}]")
    
    print("\n✅ data_loader test passed!")
