import os
import collections
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from models.ir50 import Backbone
from models.mobilefacenet import MobileFaceNet
from models.hyp_crossvit import HyVisionTransformer

logger = logging.getLogger(__name__)


class SE_block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d), nn.Sigmoid())

    def forward(self, x):
        return x * self.fc(x)


class POSTER(nn.Module):
    def __init__(self, num_classes=7, depth=8):
        super().__init__()
        self.face_landback = MobileFaceNet([112, 112], 136)
        self.ir_back = Backbone(50, 0.0, "ir")
        self.ir_layer = nn.Linear(1024, 512)
        self.pyramid_fuse = HyVisionTransformer(
            in_chans=49, q_chanel=49, embed_dim=512,
            depth=depth, num_heads=8, mlp_ratio=2.0,
            drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1,
        )
        self.se_block = SE_block(512)
        self.dropout = nn.Dropout(0.3)
        self.head = nn.Linear(512, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x_face = F.interpolate(x, size=112)
        _, x_face = self.face_landback(x_face)
        x_face = x_face.view(B, -1, 49).transpose(1, 2)
        x_ir = self.ir_layer(self.ir_back(x))
        y = self.se_block(self.pyramid_fuse(x_ir, x_face))
        y = self.dropout(y)
        return self.head(y), y


def _load_weights_with_mapping(model, sd):
    md = model.state_dict()
    new = collections.OrderedDict()
    key_map = {}
    idx = 0
    for group, count in [("body1", 3), ("body2", 4), ("body3", 14)]:
        for i in range(count):
            key_map[f"{group}.{i}."] = f"body.{idx}."
            idx += 1
    for k, v in sd.items():
        k_clean = k.replace("module.", "")
        mapped_k = k_clean
        for old_prefix, new_prefix in key_map.items():
            if k_clean.startswith(old_prefix):
                mapped_k = k_clean.replace(old_prefix, new_prefix)
                break
        if mapped_k in md and md[mapped_k].size() == v.size():
            new[mapped_k] = v
    md.update(new)
    model.load_state_dict(md)
    return model


class POSTERPredictor:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_dir=None, checkpoint_name="poster_best.pth", emotion_labels=None):
        if self._initialized:
            return
        self._initialized = True
        self.model = None
        self.model_name = "POSTER (PyTorch)"
        self.img_size = (224, 224)
        self.emotion_labels = emotion_labels or [
            "Surprise", "Fear", "Disgust", "Happiness", "Sadness", "Anger", "Neutral"
        ]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.model_dir = model_dir or os.path.join(base_dir, "outputs", "models")
        # Ưu tiên finetuned model
        ft_path = os.path.join(self.model_dir, "poster_best_finetuned.pth")
        self.checkpoint_path = ft_path if os.path.exists(ft_path) else os.path.join(self.model_dir, checkpoint_name)
        self.pretrain_ir50 = os.path.join(base_dir, "pretrain", "ir50.pth")
        self.pretrain_mfn = os.path.join(base_dir, "pretrain", "mobilefacenet_model_best.pth.tar")
        self._load_model()

    def _load_model(self):
        if not os.path.exists(self.checkpoint_path):
            logger.error("POSTER checkpoint not found: %s", self.checkpoint_path)
            return
        try:
            self.model = POSTER(num_classes=len(self.emotion_labels), depth=8)
            ckpt = torch.load(self.checkpoint_path, map_location=self.device)
            sd = ckpt.get("state_dict", ckpt)
            try:
                self.model.load_state_dict(sd)
                logger.info("POSTER: direct load OK")
            except Exception:
                logger.info("POSTER: direct load failed, using key mapping...")
                self.model = _load_weights_with_mapping(self.model, sd)
            self.model.to(self.device)
            self.model.eval()
            logger.info("POSTER model loaded: %s", self.checkpoint_path)
        except Exception as e:
            logger.error("POSTER load error: %s", e)
            self.model = None

    @property
    def is_loaded(self):
        return self.model is not None

    def predict(self, img_array):
        if not self.is_loaded:
            return {"emotion": "Unknown", "confidence": 0.0, "probabilities": {}, "success": False,
                    "error": "POSTER model not loaded."}
        try:
            if img_array.ndim == 3 and img_array.shape[-1] == 3:
                img_pil = Image.fromarray(img_array)
            else:
                img_pil = Image.fromarray(img_array).convert("RGB")
            input_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits, _ = self.model(input_tensor)
                probabilities = F.softmax(logits, dim=1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probabilities))
            confidence = float(probabilities[pred_idx])
            return {
                "emotion": self.emotion_labels[pred_idx],
                "confidence": confidence,
                "probabilities": {label: float(probabilities[i]) for i, label in enumerate(self.emotion_labels)},
                "success": True,
            }
        except Exception as e:
            logger.error("POSTER predict error: %s", e)
            return {"emotion": "Unknown", "confidence": 0.0, "probabilities": {}, "success": False, "error": str(e)}

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
