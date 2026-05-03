"""
app/models/emotion_model.py
============================
[M] Model Layer — ML inference logic.

Singleton EmotionPredictor: load model 1 lần, predict nhiều lần.
Tự động detect model tốt nhất trong outputs/models/.
"""

import os
import glob
import json
import logging
import numpy as np

logger = logging.getLogger(__name__)


class EmotionPredictor:
    """
    Singleton class để load và inference Keras emotion model.

    Tự động chọn model tốt nhất dựa trên kết quả đã lưu.
    Hỗ trợ cả baseline (100×100) và transfer learning (224×224).
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_dir=None, emotion_labels=None):
        if self._initialized:
            return
        self._initialized = True
        self.model = None
        self.model_name = None
        self.img_size = (100, 100)
        self.model_dir = model_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "outputs", "models"
        )
        self.emotion_labels = emotion_labels or [
            "Surprise", "Fear", "Disgust",
            "Happiness", "Sadness", "Anger", "Neutral"
        ]
        self._load_best_model()

    # ----------------------------------------------------------
    # Private helpers
    # ----------------------------------------------------------

    def _find_best_model_path(self):
        """Tìm model tốt nhất dựa trên accuracy trong file results."""
        eval_dir = os.path.join(
            os.path.dirname(self.model_dir), "evaluation"
        )
        best_path = None
        best_acc = -1

        result_files = glob.glob(os.path.join(eval_dir, "*_results.json"))
        for rf in result_files:
            try:
                with open(rf) as f:
                    data = json.load(f)
                model_name = data.get("model_name") or \
                    os.path.basename(rf).replace("_results.json", "")
                acc = data.get("accuracy", 0)
                keras_path = os.path.join(self.model_dir, f"{model_name}.keras")
                if os.path.exists(keras_path) and acc > best_acc:
                    best_acc = acc
                    best_path = keras_path
                    self.model_name = model_name
            except Exception:
                continue

        # Fallback: lấy file .keras đầu tiên tìm được
        if best_path is None:
            keras_files = glob.glob(os.path.join(self.model_dir, "*.keras"))
            if keras_files:
                # Ưu tiên mobilenetv2 > resnet > baseline
                for prefer in ["mobilenetv2", "resnet50", "baseline"]:
                    for kf in keras_files:
                        if prefer in kf.lower():
                            best_path = kf
                            break
                    if best_path:
                        break
                if best_path is None:
                    best_path = keras_files[0]
                self.model_name = os.path.basename(best_path).replace(".keras", "")

        return best_path

    def _load_best_model(self):
        """Load model vào memory."""
        try:
            import tensorflow as tf
            from utils.models import ConvLayer
            model_path = self._find_best_model_path()
            if model_path is None:
                logger.warning("Không tìm thấy model nào trong %s", self.model_dir)
                return

            logger.info("Loading model: %s", model_path)
            self.model = tf.keras.models.load_model(model_path, custom_objects={'ConvLayer': ConvLayer})

            # Xác định img_size từ model input shape
            input_shape = self.model.input_shape
            self.img_size = (input_shape[1], input_shape[2])
            logger.info(
                "Model loaded: %s | img_size=%s", self.model_name, self.img_size
            )
        except ImportError:
            logger.error("TensorFlow chưa được cài đặt.")
        except Exception as e:
            logger.error("Lỗi load model: %s", e)

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    @property
    def is_loaded(self):
        return self.model is not None

    def preprocess(self, img_array):
        """
        Tiền xử lý ảnh từ numpy array (BGR hoặc RGB).

        Args:
            img_array: numpy array shape (H, W, 3), uint8 hoặc float

        Returns:
            tensor: shape (1, H, W, 3), float32 normalized
        """
        import cv2
        from utils.data_loader import preprocess_raf_np
        # Resize
        img_resized = cv2.resize(img_array, (self.img_size[1], self.img_size[0]))
        # Preprocess numpy using RAF-DB mean/std
        img_normalized = preprocess_raf_np(img_resized)
        # Add batch dim
        return np.expand_dims(img_normalized, axis=0)

    def predict(self, img_array):
        """
        Dự đoán cảm xúc từ ảnh khuôn mặt đã crop.

        Args:
            img_array: numpy array (H, W, 3) — ảnh mặt đã crop, BGR hoặc RGB

        Returns:
            dict: {
                "emotion": str (label dự đoán),
                "confidence": float (0–1),
                "probabilities": dict {label: float},
                "success": bool
            }
        """
        if not self.is_loaded:
            return {
                "emotion": "Unknown",
                "confidence": 0.0,
                "probabilities": {},
                "success": False,
                "error": "Model chưa được load."
            }

        try:
            img_tensor = self.preprocess(img_array)
            proba = self.model.predict(img_tensor, verbose=0)[0]
            pred_idx = int(np.argmax(proba))
            confidence = float(proba[pred_idx])

            return {
                "emotion": self.emotion_labels[pred_idx],
                "confidence": confidence,
                "probabilities": {
                    label: float(proba[i])
                    for i, label in enumerate(self.emotion_labels)
                },
                "success": True,
            }
        except Exception as e:
            logger.error("Predict error: %s", e)
            return {
                "emotion": "Unknown",
                "confidence": 0.0,
                "probabilities": {},
                "success": False,
                "error": str(e)
            }

    def get_info(self):
        """Trả về thông tin model."""
        if not self.is_loaded:
            return {"loaded": False}
        total = self.model.count_params()
        return {
            "loaded": True,
            "model_name": self.model_name,
            "img_size": self.img_size,
            "num_classes": len(self.emotion_labels),
            "emotion_labels": self.emotion_labels,
            "total_params": total,
        }
