"""
app/controllers/main_controller.py
====================================
[C] Controller — Routes cho trang chính và navigation.

Blueprint: main
    GET  /          → index.html (webcam real-time)
    GET  /upload    → upload.html (upload ảnh)
    GET  /api/info  → JSON thông tin model
"""

from flask import Blueprint, render_template, jsonify, current_app
from app.models.emotion_model import EmotionPredictor

main_bp = Blueprint("main", __name__)


@main_bp.route("/")
def index():
    """Trang chính: real-time webcam FER."""
    predictor = EmotionPredictor()
    model_info = predictor.get_info()
    return render_template(
        "index.html",
        model_info=model_info,
        emotion_labels=current_app.config["EMOTION_LABELS"],
        emotion_colors=current_app.config["EMOTION_COLORS"],
        emotion_emojis=current_app.config["EMOTION_EMOJIS"],
    )


@main_bp.route("/upload")
def upload_page():
    """Trang upload ảnh để phân tích."""
    predictor = EmotionPredictor()
    model_info = predictor.get_info()
    return render_template(
        "upload.html",
        model_info=model_info,
        emotion_labels=current_app.config["EMOTION_LABELS"],
        emotion_colors=current_app.config["EMOTION_COLORS"],
        emotion_emojis=current_app.config["EMOTION_EMOJIS"],
    )


@main_bp.route("/api/info")
def api_info():
    """JSON endpoint trả về thông tin model đang chạy."""
    predictor = EmotionPredictor()
    return jsonify({
        "success": True,
        "model": predictor.get_info(),
        "emotion_labels": current_app.config["EMOTION_LABELS"],
        "emotion_colors": current_app.config["EMOTION_COLORS"],
        "emotion_emojis": current_app.config["EMOTION_EMOJIS"],
    })
