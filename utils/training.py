import os
import json
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================
# COMPILE MODEL
# ============================================================

def compile_model(model, learning_rate=1e-3, optimizer_name="adam"):
    """
    Compile model với optimizer, loss function và metrics.
    
    Cấu hình:
        - Optimizer: Adam (mặc định) hoặc SGD với momentum
        - Loss: Sparse Categorical Crossentropy
          (sparse vì labels là integer, không phải one-hot)
        - Metrics: Accuracy
    
    Tại sao dùng Adam?
        - Tự động điều chỉnh learning rate cho từng parameter
        - Hội tụ nhanh hơn SGD trên dataset nhỏ-vừa
        - Phù hợp khi chạy trên CPU (ít epochs → cần hội tụ nhanh)
    
    Args:
        model (keras.Model): Model cần compile
        learning_rate (float): Learning rate ban đầu
        optimizer_name (str): 'adam' hoặc 'sgd'
    
    Returns:
        model: Model đã compile (in-place, cũng trả về)
    """
    if optimizer_name.lower() == "adam":
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name.lower() == "sgd":
        optimizer = keras.optimizers.SGD(
            learning_rate=learning_rate,
            momentum=0.9,
            nesterov=True
        )
    else:
        raise ValueError(f"Optimizer không hỗ trợ: {optimizer_name}")
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"  ✅ Model compiled:")
    print(f"     Optimizer: {optimizer_name} (lr={learning_rate})")
    print(f"     Loss: sparse_categorical_crossentropy")
    print(f"     Metrics: accuracy")
    
    return model


# ============================================================
# CALLBACKS
# ============================================================

def get_callbacks(model_save_path, patience_early=10, patience_lr=5,
                  min_lr=1e-7, monitor='val_loss'):
    """
    Tạo danh sách callbacks cho training.
    
    Callbacks giúp tự động hóa quá trình training:
    
    1. ModelCheckpoint:
       - Lưu model tốt nhất dựa trên val_loss
       - Chỉ lưu khi val_loss cải thiện → tiết kiệm disk
    
    2. EarlyStopping:
       - Dừng training nếu val_loss không cải thiện sau N epochs
       - Tránh overfitting, tiết kiệm thời gian
       - restore_best_weights=True: Quay lại trọng số tốt nhất
    
    3. ReduceLROnPlateau:
       - Giảm learning rate khi val_loss đi ngang (plateau)
       - factor=0.5: Giảm lr còn 50%
       - Giúp model tinh chỉnh ở giai đoạn cuối training
    
    4. CSVLogger:
       - Ghi log training metrics theo từng epoch
       - Hữu ích cho phân tích sau training
    
    Args:
        model_save_path (str): Đường dẫn lưu model (.keras)
        patience_early (int): Số epochs chờ trước khi early stop
        patience_lr (int): Số epochs chờ trước khi giảm lr
        min_lr (float): Learning rate tối thiểu
        monitor (str): Metric để monitor ('val_loss' hoặc 'val_accuracy')
    
    Returns:
        callbacks (list): Danh sách Keras callbacks
    """
    # Tạo thư mục output nếu chưa có
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    log_dir = os.path.dirname(model_save_path)
    
    callbacks = [
        # 1. Lưu model tốt nhất
        keras.callbacks.ModelCheckpoint(
            filepath=model_save_path,
            monitor=monitor,
            save_best_only=True,
            mode='min' if 'loss' in monitor else 'max',
            verbose=1
        ),
        
        # 2. Early stopping
        keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=patience_early,
            restore_best_weights=True,
            verbose=1,
            mode='min' if 'loss' in monitor else 'max'
        ),
        
        # 3. Reduce learning rate
        keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=patience_lr,
            min_lr=min_lr,
            verbose=1,
            mode='min' if 'loss' in monitor else 'max'
        ),
        
        # 4. CSV Logger
        keras.callbacks.CSVLogger(
            os.path.join(log_dir, 'training_log.csv'),
            append=False
        ),
    ]
    
    print(f"  📋 Callbacks:")
    print(f"     ModelCheckpoint → {model_save_path}")
    print(f"     EarlyStopping (patience={patience_early})")
    print(f"     ReduceLROnPlateau (patience={patience_lr}, factor=0.5)")
    print(f"     CSVLogger → {log_dir}/training_log.csv")
    
    return callbacks


# ============================================================
# TRAINING
# ============================================================

def train_model(model, train_ds, val_ds, epochs=30, 
                class_weights=None, callbacks=None,
                verbose=1):
    """
    Huấn luyện model với đầy đủ cấu hình.
    
    Quy trình:
        1. Ghi nhận thời gian bắt đầu
        2. Gọi model.fit() với train/val data
        3. Sử dụng class_weights để xử lý imbalanced data
        4. Ghi nhận thời gian kết thúc
        5. In tóm tắt kết quả
    
    Args:
        model (keras.Model): Model đã compile
        train_ds (tf.data.Dataset): Training dataset
        val_ds (tf.data.Dataset): Validation dataset
        epochs (int): Số epochs tối đa
        class_weights (dict): Class weights {class_id: weight}
        callbacks (list): Danh sách callbacks
        verbose (int): Mức độ in log (0=silent, 1=progress bar, 2=one line/epoch)
    
    Returns:
        history (keras.callbacks.History): Lịch sử training
            history.history['loss'] - Train loss theo epochs
            history.history['accuracy'] - Train accuracy theo epochs
            history.history['val_loss'] - Val loss theo epochs
            history.history['val_accuracy'] - Val accuracy theo epochs
    """
    print("\n" + "=" * 60)
    print("🚀 BẮT ĐẦU HUẤN LUYỆN")
    print("=" * 60)
    print(f"  Model: {model.name}")
    print(f"  Epochs: {epochs}")
    print(f"  Class weights: {'Có' if class_weights else 'Không'}")
    print(f"  Callbacks: {len(callbacks) if callbacks else 0}")
    
    start_time = time.time()
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=verbose
    )
    
    elapsed = time.time() - start_time
    elapsed_min = elapsed / 60
    
    # Tóm tắt kết quả
    best_val_loss = min(history.history['val_loss'])
    best_val_acc = max(history.history['val_accuracy'])
    final_train_loss = history.history['loss'][-1]
    final_train_acc = history.history['accuracy'][-1]
    
    print("\n" + "=" * 60)
    print("📊 KẾT QUẢ HUẤN LUYỆN")
    print("=" * 60)
    print(f"  ⏱️ Thời gian: {elapsed_min:.1f} phút ({elapsed:.0f}s)")
    print(f"  📈 Epochs thực tế: {len(history.history['loss'])}/{epochs}")
    print(f"  📉 Train: loss={final_train_loss:.4f}, acc={final_train_acc:.4f}")
    print(f"  📈 Best Val: loss={best_val_loss:.4f}, acc={best_val_acc:.4f}")
    
    return history


# ============================================================
# TRỰC QUAN HÓA TRAINING HISTORY
# ============================================================

def plot_training_history(history, save_dir="outputs/training_logs", 
                          model_name="model"):
    """
    Vẽ biểu đồ training curves (loss và accuracy).
    
    Tạo 2 subplot:
        - Trái: Loss curve (train vs validation)
        - Phải: Accuracy curve (train vs validation)
    
    Các dấu hiệu cần chú ý:
        - Overfitting: val_loss tăng trong khi train_loss giảm
        - Underfitting: Cả train và val loss đều cao
        - Good fit: Cả hai curves hội tụ về giá trị tương đương
    
    Args:
        history: keras.callbacks.History hoặc dict chứa history
        save_dir (str): Thư mục lưu biểu đồ
        model_name (str): Tên model (dùng trong title và filename)
    
    Returns:
        save_path (str): Đường dẫn file ảnh đã lưu
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Hỗ trợ cả History object và dict
    if hasattr(history, 'history'):
        hist_dict = history.history
    else:
        hist_dict = history
    
    epochs = range(1, len(hist_dict['loss']) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'{model_name} — Training History', fontsize=16, fontweight='bold')
    
    # ---- Plot 1: Loss ----
    axes[0].plot(epochs, hist_dict['loss'], 'b-o', markersize=3,
                 label='Train Loss', linewidth=2, alpha=0.8)
    axes[0].plot(epochs, hist_dict['val_loss'], 'r-o', markersize=3,
                 label='Val Loss', linewidth=2, alpha=0.8)
    axes[0].set_title('Loss', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    
    # Đánh dấu best val loss
    best_epoch = np.argmin(hist_dict['val_loss']) + 1
    best_loss = min(hist_dict['val_loss'])
    axes[0].axvline(x=best_epoch, color='green', linestyle='--', alpha=0.5,
                    label=f'Best (epoch {best_epoch})')
    axes[0].scatter([best_epoch], [best_loss], color='green', s=100, zorder=5,
                    marker='*')
    axes[0].legend(fontsize=10)
    
    # ---- Plot 2: Accuracy ----
    axes[1].plot(epochs, hist_dict['accuracy'], 'b-o', markersize=3,
                 label='Train Accuracy', linewidth=2, alpha=0.8)
    axes[1].plot(epochs, hist_dict['val_accuracy'], 'r-o', markersize=3,
                 label='Val Accuracy', linewidth=2, alpha=0.8)
    axes[1].set_title('Accuracy', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    
    # Đánh dấu best val accuracy
    best_epoch_acc = np.argmax(hist_dict['val_accuracy']) + 1
    best_acc = max(hist_dict['val_accuracy'])
    axes[1].axvline(x=best_epoch_acc, color='green', linestyle='--', alpha=0.5,
                    label=f'Best (epoch {best_epoch_acc})')
    axes[1].scatter([best_epoch_acc], [best_acc], color='green', s=100, zorder=5,
                    marker='*')
    axes[1].legend(fontsize=10)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{model_name}_training_history.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✅ Training curves saved: {save_path}")
    return save_path


def save_training_info(history, model, save_dir, model_name, 
                       extra_info=None):
    """
    Lưu thông tin training thành JSON cho phân tích sau.
    
    Args:
        history: Training history
        model: Keras model
        save_dir (str): Thư mục lưu
        model_name (str): Tên model
        extra_info (dict): Thông tin bổ sung
    """
    os.makedirs(save_dir, exist_ok=True)
    
    hist_dict = history.history if hasattr(history, 'history') else history
    
    info = {
        "model_name": model_name,
        "total_params": int(model.count_params()),
        "trainable_params": int(sum(
            tf.keras.backend.count_params(w) for w in model.trainable_weights
        )),
        "epochs_trained": len(hist_dict['loss']),
        "best_val_loss": float(min(hist_dict['val_loss'])),
        "best_val_accuracy": float(max(hist_dict['val_accuracy'])),
        "final_train_loss": float(hist_dict['loss'][-1]),
        "final_train_accuracy": float(hist_dict['accuracy'][-1]),
        "history": {k: [float(v) for v in vals] for k, vals in hist_dict.items()},
    }
    
    if extra_info:
        info.update(extra_info)
    
    save_path = os.path.join(save_dir, f"{model_name}_training_info.json")
    with open(save_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"  ✅ Training info saved: {save_path}")
    return save_path
