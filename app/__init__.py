"""
app/__init__.py
===============
App factory pattern cho Flask application.

Sử dụng:
    from app import create_app
    app = create_app("development")
    app.run()
"""

import os
import logging
from flask import Flask
from .config import config


def create_app(config_name="default"):
    """
    Factory function tạo Flask app instance.

    Args:
        config_name: "development" | "production" | "default"

    Returns:
        Flask app instance đã cấu hình đầy đủ
    """
    app = Flask(
        __name__,
        template_folder="views/templates",
        static_folder="static",
    )

    # Load config
    app.config.from_object(config[config_name])

    # Đảm bảo upload folder tồn tại
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    # Logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    # ---- Khởi tạo Model (1 lần khi start) ----
    with app.app_context():
        from .models.emotion_model import EmotionPredictor
        predictor = EmotionPredictor(
            model_dir=app.config["MODEL_DIR"],
            emotion_labels=app.config["EMOTION_LABELS"],
        )
        if predictor.is_loaded:
            logger.info("✅ EmotionPredictor loaded: %s | img_size=%s",
                        predictor.model_name, predictor.img_size)
        else:
            logger.warning("⚠️  EmotionPredictor: Không tìm thấy model. "
                           "Chạy 03_baseline_model.py trước để train model.")

    # ---- Register Blueprints ----
    from .controllers.main_controller import main_bp
    from .controllers.predict_controller import predict_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(predict_bp)

    logger.info("🚀 Flask app created [%s mode]", config_name)
    return app
