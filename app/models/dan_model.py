"""
app/models/dan_model.py
========================
DAN (Distract your Attention Network) predictor.
Gracefully degrades when PyTorch is not installed.
"""

import os
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import models, transforms
    import numpy as np
    from PIL import Image
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch/torchvision not installed — DAN model disabled.")


# ── Architecture (only defined when PyTorch is available) ───────────
if HAS_TORCH:

    class DANArchitecture(nn.Module):
        def __init__(self, num_class=7, num_head=4):
            super(DANArchitecture, self).__init__()
            resnet = models.resnet18(weights=None)
            self.features = nn.Sequential(*list(resnet.children())[:-2])
            self.num_head = num_head

            self.conv_att = nn.Conv2d(512, self.num_head, kernel_size=1)
            self.fc = nn.Linear(512, num_class)
            self.bn = nn.BatchNorm1d(num_class)

        def forward(self, x):
            x = self.features(x)

            att_map = self.conv_att(x)
            att_map = att_map.view(att_map.size(0), self.num_head, -1)
            att_map = F.softmax(att_map, dim=2)
            att_map = att_map.view(att_map.size(0), self.num_head, x.size(2), x.size(3))

            x_flat = x.view(x.size(0), 1, x.size(1), -1)
            att_flat = att_map.view(att_map.size(0), self.num_head, 1, -1)

            weighted_features = (x_flat * att_flat).sum(dim=-1)
            final_features = weighted_features.mean(dim=1)

            out = self.fc(final_features)
            out = self.bn(out)
            return out


# ── Predictor (always importable) ──────────────────────────────────
class DANPredictor:
    """Singleton predictor. Works as a no-op stub when PyTorch is missing."""

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
        self.model_name = "DAN (PyTorch)"
        self.img_size = (224, 224)

        self.emotion_labels = emotion_labels or [
            "Surprise", "Fear", "Disgust",
            "Happiness", "Sadness", "Anger", "Neutral"
        ]

        if not HAS_TORCH:
            logger.warning("DANPredictor created but PyTorch is not installed — model will not load.")
            return

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.model_dir = model_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "outputs", "models"
        )
        self._load_model()

    def _load_model(self):
        # Chi load original, bo finetuned
        model_path = os.path.join(self.model_dir, "best_dan_model.pth")
        if not os.path.exists(model_path):
            logger.error("Khong tim thay model DAN tai: %s", model_path)
            return

        try:
            self.model = DANArchitecture(num_class=len(self.emotion_labels), num_head=4)
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            logger.info("DAN model loaded successfully: %s", model_path)
        except Exception as e:
            logger.error("Loi load model DAN: %s", e)
            self.model = None

    @property
    def is_loaded(self):
        return self.model is not None

    def predict(self, img_array):
        """
        Dự đoán cảm xúc bằng mô hình DAN.
        Args:
            img_array: numpy array (H, W, 3) RGB
        Returns:
            dict: prediction results
        """
        if not self.is_loaded:
            return {
                "emotion": "Unknown",
                "confidence": 0.0,
                "probabilities": {},
                "success": False,
                "error": "DAN Model chưa được load."
            }

        try:
            if img_array.shape[-1] == 3:
                img_pil = Image.fromarray(img_array)
            else:
                img_pil = Image.fromarray(img_array).convert('RGB')

            input_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(input_tensor)
                probabilities = F.softmax(output, dim=1).cpu().numpy()[0]

            pred_idx = int(np.argmax(probabilities))
            confidence = float(probabilities[pred_idx])

            return {
                "emotion": self.emotion_labels[pred_idx],
                "confidence": confidence,
                "probabilities": {
                    label: float(probabilities[i])
                    for i, label in enumerate(self.emotion_labels)
                },
                "success": True,
            }
        except Exception as e:
            logger.error("DAN predict error: %s", e)
            return {
                "emotion": "Unknown",
                "confidence": 0.0,
                "probabilities": {},
                "success": False,
                "error": str(e)
            }

    def get_info(self):
        if not self.is_loaded:
            return {"loaded": False}
        total = sum(p.numel() for p in self.model.parameters())
        return {
            "loaded": True,
            "model_name": self.model_name,
            "img_size": self.img_size,
            "num_classes": len(self.emotion_labels),
            "emotion_labels": self.emotion_labels,
            "total_params": total,
        }
