"""
app/config.py
=============
Cấu hình ứng dụng Flask cho hệ thống FER.
"""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Config:
    """Cấu hình cơ bản."""
    SECRET_KEY = os.environ.get("SECRET_KEY", "fer-demo-secret-key-2024")
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max upload

    # Upload folder
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "app", "static", "uploads")
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp", "webp"}

    # Model settings
    MODEL_DIR = os.path.join(BASE_DIR, "outputs", "models")
    DEFAULT_MODEL_NAME = None      # None = auto-detect best
    IMAGE_SIZE_BASELINE = (100, 100)
    IMAGE_SIZE_TRANSFER = (224, 224)

    # Inference
    CONFIDENCE_THRESHOLD = 0.2    # Min confidence để hiển thị kết quả
    FACE_SCALE_FACTOR = 1.1
    FACE_MIN_NEIGHBORS = 4
    FACE_MIN_SIZE = (48, 48)

    # Emotion labels & colors
    EMOTION_LABELS = [
        "Surprise", "Fear", "Disgust",
        "Happiness", "Sadness", "Anger", "Neutral"
    ]
    EMOTION_COLORS = {
        "Surprise":  "#d97706", # Elegant Amber
        "Fear":      "#7c3aed", # Premium Purple
        "Disgust":   "#0f766e", # Deep Teal
        "Happiness": "#059669", # Emerald Green
        "Sadness":   "#2563eb", # Royal Blue
        "Anger":     "#dc2626", # Sophisticated Crimson
        "Neutral":   "#4b5563", # Refined Slate Grey
    }
    EMOTION_EMOJIS = {
        "Surprise":  "",
        "Fear":      "",
        "Disgust":   "",
        "Happiness": "",
        "Sadness":   "",
        "Anger":     "",
        "Neutral":   "",
    }


class DevelopmentConfig(Config):
    DEBUG = True
    TESTING = False


class ProductionConfig(Config):
    DEBUG = False
    TESTING = False


config = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "default": DevelopmentConfig,
}
