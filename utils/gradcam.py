"""
utils/gradcam.py
================
Grad-CAM (Gradient-weighted Class Activation Mapping) cho FER.

Grad-CAM giải thích model "nhìn vào đâu" khi đưa ra dự đoán:
    - Tạo heatmap highlight vùng ảnh quan trọng nhất
    - Giúp verify model học đúng features (mắt, miệng, lông mày)
    - Phát hiện nếu model dựa vào background hoặc artifacts

Reference: Selvaraju et al., "Grad-CAM", ICCV 2017

Sử dụng:
    >>> from utils.gradcam import GradCAM
    >>> gradcam = GradCAM(model, layer_name='conv4')
    >>> heatmap = gradcam.compute_heatmap(image)
    >>> overlay = gradcam.overlay(image, heatmap)

Yêu cầu: tensorflow, numpy, opencv-python, matplotlib
"""

import numpy as np
import cv2
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


class GradCAM:
    """
    Grad-CAM: Trực quan hóa vùng ảnh model chú ý.
    
    Nguyên lý hoạt động:
        1. Forward pass: Lấy feature map từ conv layer cuối
        2. Backward pass: Tính gradient của class score theo feature map
        3. Global Average Pooling trên gradients → importance weights
        4. Weighted sum feature maps → heatmap
        5. ReLU → chỉ giữ vùng có ảnh hưởng positive
    
    Attributes:
        model (keras.Model): Model gốc
        layer_name (str): Tên conv layer để lấy feature map
        grad_model (keras.Model): Model con để tính gradient
    """
    
    def __init__(self, model, layer_name=None):
        """
        Khởi tạo Grad-CAM.
        
        Args:
            model (keras.Model): Model đã huấn luyện
            layer_name (str): Tên conv layer. Nếu None, tự động
                              tìm conv layer cuối cùng trong model.
        """
        self.model = model
        
        # Tự động tìm conv layer cuối cùng nếu không chỉ định
        if layer_name is None:
            layer_name = self._find_last_conv_layer()
        
        self.layer_name = layer_name
        
        # Tạo gradient model: input → [conv_output, predictions]
        self.grad_model = self._build_grad_model()
        
        print(f"  🔍 GradCAM initialized with layer: {self.layer_name}")
    
    def _find_last_conv_layer(self):
        """Tìm Conv2D layer cuối cùng trong model."""
        last_conv = None
        for layer in self.model.layers:
            if hasattr(layer, 'layers'):
                # Nested model (e.g., ResNet50 backbone)
                for sub_layer in layer.layers:
                    if isinstance(sub_layer, tf.keras.layers.Conv2D):
                        last_conv = f"{layer.name}/{sub_layer.name}"
            elif isinstance(layer, tf.keras.layers.Conv2D):
                last_conv = layer.name
        
        if last_conv is None:
            raise ValueError("Không tìm thấy Conv2D layer trong model!")
        
        return last_conv
    
    def _build_grad_model(self):
        """Xây dựng model con để trích xuất feature map và predictions."""
        # Tìm layer trong model (hỗ trợ cả nested models)
        target_layer = None
        
        for layer in self.model.layers:
            if layer.name == self.layer_name:
                target_layer = layer
                break
            if hasattr(layer, 'layers'):
                for sub_layer in layer.layers:
                    if sub_layer.name == self.layer_name:
                        target_layer = sub_layer
                        break
                    full_name = f"{layer.name}/{sub_layer.name}"
                    if full_name == self.layer_name:
                        target_layer = sub_layer
                        break
        
        if target_layer is None:
            # Fallback: tìm conv layer cuối trực tiếp
            for layer in reversed(self.model.layers):
                if isinstance(layer, tf.keras.layers.Conv2D):
                    target_layer = layer
                    self.layer_name = layer.name
                    break
        
        if target_layer is None:
            raise ValueError(f"Layer '{self.layer_name}' không tìm thấy!")
        
        grad_model = tf.keras.Model(
            inputs=self.model.input,
            outputs=[target_layer.output, self.model.output]
        )
        
        return grad_model
    
    def compute_heatmap(self, image, class_idx=None, eps=1e-8):
        """
        Tính Grad-CAM heatmap cho một ảnh.
        
        Quy trình:
            1. Forward pass → feature map + predictions
            2. Lấy score của class mục tiêu
            3. Tính gradient: d(score) / d(feature_map)
            4. Global Average Pooling trên gradients → weights
            5. Weighted combination: sum(weights * feature_maps)
            6. ReLU + normalize → heatmap [0, 1]
        
        Args:
            image (np.ndarray): Ảnh [1, H, W, 3] đã normalize
            class_idx (int): Class muốn visualize. None = predicted class
            eps (float): Epsilon tránh chia cho 0
        
        Returns:
            heatmap (np.ndarray): Heatmap [H, W], range [0, 1]
            predicted_class (int): Class được dự đoán
            confidence (float): Confidence của prediction
        """
        image_tensor = tf.cast(image, tf.float32)
        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(image_tensor)
            
            if class_idx is None:
                class_idx = tf.argmax(predictions[0])
            
            class_score = predictions[:, class_idx]
        
        # Tính gradients
        grads = tape.gradient(class_score, conv_outputs)
        
        # Global Average Pooling trên gradients
        weights = tf.reduce_mean(grads, axis=(1, 2))  # [1, C]
        
        # Weighted combination
        cam = tf.reduce_sum(
            conv_outputs * weights[:, tf.newaxis, tf.newaxis, :],
            axis=-1
        )[0]  # [H, W]
        
        # ReLU
        cam = tf.nn.relu(cam)
        
        # Normalize
        cam = cam / (tf.reduce_max(cam) + eps)
        heatmap = cam.numpy()
        
        predicted_class = int(tf.argmax(predictions[0]).numpy())
        confidence = float(predictions[0, predicted_class].numpy())
        
        return heatmap, predicted_class, confidence
    
    def overlay(self, original_image, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
        """
        Chồng heatmap lên ảnh gốc.
        
        Args:
            original_image (np.ndarray): Ảnh gốc [H, W, 3], uint8 hoặc float
            heatmap (np.ndarray): Heatmap [h, w], range [0, 1]
            alpha (float): Transparency (0=chỉ heatmap, 1=chỉ ảnh gốc)
            colormap: OpenCV colormap
        
        Returns:
            overlay (np.ndarray): Ảnh chồng [H, W, 3], uint8
        """
        # Ensure uint8
        if original_image.dtype == np.float32 or original_image.dtype == np.float64:
            if original_image.max() <= 1.0:
                original_image = (original_image * 255).astype(np.uint8)
            else:
                original_image = original_image.astype(np.uint8)
        
        # Resize heatmap
        h, w = original_image.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Overlay
        overlay = cv2.addWeighted(original_image, 1 - alpha, heatmap_colored, alpha, 0)
        
        return overlay


def visualize_gradcam_grid(model, image_paths, labels, class_names,
                           img_size=(100, 100), save_path=None,
                           samples_per_class=3, layer_name=None):
    """
    Tạo grid Grad-CAM cho nhiều ảnh, mỗi class vài mẫu.
    
    Args:
        model: Keras model
        image_paths (list): Đường dẫn ảnh
        labels (list): Nhãn (0-indexed)
        class_names (list): Tên classes
        img_size (tuple): Kích thước input model
        save_path (str): Đường dẫn lưu
        samples_per_class (int): Số mẫu mỗi class
        layer_name (str): Tên conv layer cho Grad-CAM
    """
    gradcam = GradCAM(model, layer_name=layer_name)
    n_classes = len(class_names)
    
    fig, axes = plt.subplots(n_classes, samples_per_class * 2,
                              figsize=(samples_per_class * 6, n_classes * 3))
    fig.suptitle('Grad-CAM Visualization', fontsize=18, fontweight='bold')
    
    labels = np.array(labels)
    
    for ci in range(n_classes):
        class_indices = np.where(labels == ci)[0]
        if len(class_indices) == 0:
            continue
        
        np.random.seed(42)
        selected = np.random.choice(class_indices, 
                                     size=min(samples_per_class, len(class_indices)),
                                     replace=False)
        
        for si, idx in enumerate(selected):
            # Load image
            img = cv2.imread(image_paths[idx])
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, img_size)
            img_normalized = img_resized.astype(np.float32) / 255.0
            img_batch = np.expand_dims(img_normalized, axis=0)
            
            # Compute Grad-CAM
            try:
                heatmap, pred_cls, conf = gradcam.compute_heatmap(img_batch)
                overlay = gradcam.overlay(img_resized, heatmap)
            except Exception as e:
                print(f"  ⚠️ GradCAM error for {image_paths[idx]}: {e}")
                continue
            
            # Original image
            ax_orig = axes[ci][si * 2]
            ax_orig.imshow(img_resized)
            ax_orig.set_title(f'True: {class_names[ci]}', fontsize=9)
            ax_orig.axis('off')
            
            # Grad-CAM overlay
            ax_cam = axes[ci][si * 2 + 1]
            ax_cam.imshow(overlay)
            pred_name = class_names[pred_cls]
            color = 'green' if pred_cls == ci else 'red'
            ax_cam.set_title(f'Pred: {pred_name} ({conf:.2f})',
                           fontsize=9, color=color, fontweight='bold')
            ax_cam.axis('off')
        
        # Label row
        axes[ci][0].set_ylabel(class_names[ci], fontsize=11, fontweight='bold',
                               rotation=0, labelpad=60)
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  ✅ Grad-CAM grid saved: {save_path}")
    plt.close()
