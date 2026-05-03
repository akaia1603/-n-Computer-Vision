"""
utils/evaluation.py
===================
Module đánh giá mô hình cho bài toán FER.
Chức năng: Confusion Matrix, Classification Report, ROC-AUC, Error Analysis, So sánh models.
Yêu cầu: tensorflow, scikit-learn, matplotlib, seaborn, numpy
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, f1_score, precision_score, recall_score,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import tensorflow as tf


def evaluate_model(model, test_ds, class_names=None):
    """
    Đánh giá model trên test set với nhiều metrics.
    
    Metrics:
        - Accuracy: Tỉ lệ dự đoán đúng tổng thể
        - Weighted F1: F1 trung bình có trọng số (phản ánh thực tế)
        - Macro F1: F1 trung bình không trọng số (công bằng mọi class)
    
    Returns:
        results (dict): y_true, y_pred, y_proba, accuracy, weighted_f1, macro_f1, etc.
    """
    if class_names is None:
        class_names = ["Surprise", "Fear", "Disgust", "Happiness", 
                       "Sadness", "Anger", "Neutral"]
    
    print("=" * 60)
    print("📊 ĐÁNH GIÁ MÔ HÌNH")
    print("=" * 60)
    
    y_true, y_proba = [], []
    for images, labels in test_ds:
        predictions = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_proba.extend(predictions)
    
    y_true = np.array(y_true)
    y_proba = np.array(y_proba)
    y_pred = np.argmax(y_proba, axis=1)
    
    accuracy = accuracy_score(y_true, y_pred)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    
    per_class = {}
    prec = precision_score(y_true, y_pred, average=None)
    rec = recall_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average=None)
    for i, name in enumerate(class_names):
        per_class[name] = {
            'precision': float(prec[i]), 'recall': float(rec[i]),
            'f1': float(f1[i]), 'support': int(np.sum(y_true == i))
        }
    
    print(f"\n  Accuracy:    {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Weighted F1: {weighted_f1:.4f}")
    print(f"  Macro F1:    {macro_f1:.4f}")
    print(f"\n{report}")
    
    return {
        'y_true': y_true, 'y_pred': y_pred, 'y_proba': y_proba,
        'accuracy': accuracy, 'weighted_f1': weighted_f1,
        'macro_f1': macro_f1, 'classification_report': report,
        'per_class_metrics': per_class
    }


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None,
                          normalize=True, title=None):
    """Vẽ Confusion Matrix heatmap. normalize=True hiện tỉ lệ %."""
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm_display = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt, vmin, vmax = '.2%', 0, 1
    else:
        cm_display, fmt, vmin, vmax = cm, 'd', None, None
    
    if title is None:
        title = "Confusion Matrix" + (" (Normalized)" if normalize else "")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_display, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, vmin=vmin, vmax=vmax, linewidths=0.5,
                annot_kws={'size': 11, 'fontweight': 'bold'})
    ax.set_xlabel('Predicted Label', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=15, fontweight='bold', pad=15)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=11)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=11)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  ✅ Confusion matrix saved: {save_path}")
    plt.close()


def plot_roc_curves(y_true, y_proba, class_names, save_path=None):
    """Vẽ ROC Curves (One-vs-Rest) cho mỗi class. AUC > 0.8 = tốt."""
    num_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=range(num_classes))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
    auc_scores = {}
    
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        auc_scores[class_names[i]] = roc_auc
        ax.plot(fpr, tpr, color=colors[i], linewidth=2,
                label=f'{class_names[i]} (AUC = {roc_auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    ax.set_xlabel('False Positive Rate', fontsize=13)
    ax.set_ylabel('True Positive Rate', fontsize=13)
    ax.set_title('ROC Curves (One-vs-Rest)', fontsize=15, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  ✅ ROC curves saved: {save_path}")
    plt.close()
    return auc_scores


def analyze_errors(y_true, y_pred, y_proba, test_paths, class_names,
                   save_dir=None, top_k=5):
    """
    Phân tích ảnh bị dự đoán sai:
    - Confusion pairs (cặp class hay nhầm)
    - Confident errors (sai nhưng tự tin cao)
    - Per-class error rate
    """
    import cv2
    print("\n" + "=" * 60)
    print("🔍 ERROR ANALYSIS")
    print("=" * 60)
    
    error_mask = y_true != y_pred
    error_indices = np.where(error_mask)[0]
    total_errors = len(error_indices)
    print(f"\n  Tổng lỗi: {total_errors}/{len(y_true)} ({total_errors/len(y_true)*100:.1f}%)")
    
    # Confusion pairs
    cm = confusion_matrix(y_true, y_pred)
    np.fill_diagonal(cm, 0)
    confusion_pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if cm[i][j] > 0:
                confusion_pairs.append({
                    'true': class_names[i], 'pred': class_names[j],
                    'count': int(cm[i][j]),
                    'rate': float(cm[i][j]) / max(np.sum(y_true == i), 1)
                })
    confusion_pairs.sort(key=lambda x: x['count'], reverse=True)
    
    print(f"\n  Top Confusion Pairs:")
    for p in confusion_pairs[:10]:
        print(f"     {p['true']:>10s} → {p['pred']:<10s}: {p['count']:3d} ({p['rate']*100:.1f}%)")
    
    # Confident errors
    print(f"\n  Top Confident Errors:")
    error_conf = y_proba[error_indices, y_pred[error_indices]]
    sorted_err = error_indices[np.argsort(-error_conf)]
    for rank, idx in enumerate(sorted_err[:top_k]):
        print(f"     #{rank+1}: True={class_names[y_true[idx]]}, "
              f"Pred={class_names[y_pred[idx]]}, Conf={y_proba[idx, y_pred[idx]]:.3f}")
    
    # Per-class error rate
    print(f"\n  Per-class Error Rate:")
    per_class_errors = {}
    for i, name in enumerate(class_names):
        mask = y_true == i
        if mask.sum() > 0:
            errs = (y_pred[mask] != i).sum()
            per_class_errors[name] = {'errors': int(errs), 'total': int(mask.sum()),
                                       'error_rate': float(errs / mask.sum())}
            print(f"     {name:>10s}: {errs}/{mask.sum()} ({errs/mask.sum()*100:.1f}%)")
    
    # Plot error samples
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        n_classes = len(class_names)
        fig, axes = plt.subplots(n_classes, top_k, figsize=(top_k * 3, n_classes * 3))
        fig.suptitle('Error Analysis: Ảnh bị phân loại sai', fontsize=16, fontweight='bold')
        for ci in range(n_classes):
            errs_idx = np.where((y_true == ci) & (y_true != y_pred))[0]
            for col in range(top_k):
                ax = axes[ci][col] if n_classes > 1 else axes[col]
                if col < len(errs_idx):
                    img = cv2.imread(test_paths[errs_idx[col]])
                    if img is not None:
                        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                        ax.set_title(f'P:{class_names[y_pred[errs_idx[col]]]}\n'
                                    f'({y_proba[errs_idx[col], y_pred[errs_idx[col]]]:.2f})',
                                    fontsize=8, color='red', fontweight='bold')
                ax.axis('off')
                if col == 0:
                    ax.set_ylabel(class_names[ci], fontsize=10, fontweight='bold',
                                rotation=0, labelpad=60)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "error_samples.png"), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Error samples saved: {save_dir}/error_samples.png")
    
    return {'total_errors': total_errors, 'confusion_pairs': confusion_pairs[:10],
            'per_class_errors': per_class_errors}


def compare_models(results_dict, save_path=None):
    """So sánh nhiều models bằng bảng text và bar chart."""
    print("\n" + "=" * 60)
    print("📊 SO SÁNH MÔ HÌNH")
    print("=" * 60)
    
    model_names = list(results_dict.keys())
    header = f"  {'Model':<25} {'Accuracy':>10} {'W-F1':>10} {'M-F1':>10}"
    print(f"\n{header}\n  {'-' * 57}")
    
    data = []
    for name, res in results_dict.items():
        acc, wf1, mf1 = res['accuracy'], res['weighted_f1'], res['macro_f1']
        print(f"  {name:<25} {acc:>10.4f} {wf1:>10.4f} {mf1:>10.4f}")
        data.append([acc, wf1, mf1])
    
    if save_path:
        metrics_names = ['Accuracy', 'Weighted F1', 'Macro F1']
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(metrics_names))
        width = 0.8 / len(model_names)
        colors = ['#4A90D9', '#FF6B6B', '#4CAF50', '#FFD93D']
        for i, (name, d) in enumerate(zip(model_names, data)):
            offset = (i - len(model_names) / 2 + 0.5) * width
            bars = ax.bar(x + offset, d, width, label=name,
                         color=colors[i % len(colors)], alpha=0.85)
            for bar, val in zip(bars, d):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.set_ylabel('Score', fontsize=13)
        ax.set_title('So sánh Mô hình', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_names, fontsize=12)
        ax.legend(fontsize=11)
        ax.set_ylim(0, 1.15)
        ax.grid(axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  ✅ Comparison chart saved: {save_path}")
