"""
app/controllers/predict_controller.py
=======================================
[C] Controller — API endpoints cho prediction.

Blueprint: predict
    POST /api/predict        → Nhận base64 webcam frame → JSON results
    POST /api/upload         → Nhận file upload → JSON results + annotated image
"""

import os
import uuid
import logging
from flask import (
    Blueprint, request, jsonify, current_app
)
from app.models.emotion_model import EmotionPredictor
from app.services.face_service import FaceDetectionService

logger = logging.getLogger(__name__)
predict_bp = Blueprint("predict", __name__)


def _allowed_file(filename):
    allowed = current_app.config.get("ALLOWED_EXTENSIONS", {"jpg", "jpeg", "png"})
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed


def _run_pipeline(img_rgb):
    """
    Pipeline chung: img_rgb → detect faces → predict mỗi mặt.

    Returns:
        list of face_results dicts
    """
    face_svc = FaceDetectionService(
        scale_factor=current_app.config.get("FACE_SCALE_FACTOR", 1.1),
        min_neighbors=current_app.config.get("FACE_MIN_NEIGHBORS", 4),
        min_size=tuple(current_app.config.get("FACE_MIN_SIZE", (48, 48))),
    )
    predictor = EmotionPredictor()
    emotion_colors = current_app.config["EMOTION_COLORS"]

    img_rgb_copy, faces = face_svc.detect_from_array(
        __import__("cv2").cvtColor(img_rgb, __import__("cv2").COLOR_RGB2BGR)
    )

    if not faces:
        return [], img_rgb, False

    predictions = []
    face_results = []
    for (x, y, w, h) in faces:
        face_crop = face_svc.crop_face(img_rgb, (x, y, w, h))
        pred = predictor.predict(face_crop)
        predictions.append(pred)
        face_results.append({
            "bbox": {"x": x, "y": y, "w": w, "h": h},
            "emotion": pred["emotion"],
            "confidence": pred["confidence"],
            "probabilities": pred["probabilities"],
        })

    # Vẽ annotations
    annotated = face_svc.draw_results(img_rgb, faces, predictions, emotion_colors)
    return face_results, annotated, True


# ----------------------------------------------------------
# Routes
# ----------------------------------------------------------

@predict_bp.route("/api/predict", methods=["POST"])
def predict_webcam():
    """
    Nhận frame base64 từ webcam (JSON body).

    Body JSON: { "image": "<base64 string>" }
    Response:  { "success": bool, "faces": [...], "face_count": int }
    """
    data = request.get_json(silent=True) or {}
    b64_image = data.get("image", "")

    if not b64_image:
        return jsonify({"success": False, "error": "Không có dữ liệu ảnh."}), 400

    face_svc = FaceDetectionService()
    img_rgb, faces = face_svc.detect_from_base64(b64_image)

    if img_rgb is None:
        return jsonify({"success": False, "error": "Không decode được ảnh."}), 400

    predictor = EmotionPredictor()
    emotion_colors = current_app.config["EMOTION_COLORS"]

    predictions = []
    face_results = []
    for (x, y, w, h) in faces:
        face_crop = face_svc.crop_face(img_rgb, (x, y, w, h))
        pred = predictor.predict(face_crop)
        predictions.append(pred)
        face_results.append({
            "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
            "emotion": pred["emotion"],
            "confidence": round(pred["confidence"], 4),
            "probabilities": {k: round(v, 4) for k, v in pred["probabilities"].items()},
        })

    # Annotated image để render lại phía client (optional)
    annotated_b64 = None
    if faces:
        annotated = face_svc.draw_results(img_rgb, faces, predictions, emotion_colors)
        annotated_b64 = face_svc.encode_to_base64(annotated)

    return jsonify({
        "success": True,
        "face_count": len(faces),
        "faces": face_results,
        "annotated_image": annotated_b64,
    })


@predict_bp.route("/api/upload", methods=["POST"])
def predict_upload():
    """
    Nhận file upload → detect faces → predict emotions.

    Form data: file = <image file>
    Response:  JSON { success, face_count, faces, annotated_image }
    """
    if "file" not in request.files:
        return jsonify({"success": False, "error": "Không có file được gửi lên."}), 400

    file = request.files["file"]
    if file.filename == "" or not _allowed_file(file.filename):
        return jsonify({
            "success": False,
            "error": "File không hợp lệ. Chỉ hỗ trợ PNG, JPG, JPEG, WEBP."
        }), 400

    # Lưu file tạm
    upload_folder = current_app.config["UPLOAD_FOLDER"]
    os.makedirs(upload_folder, exist_ok=True)
    ext = file.filename.rsplit(".", 1)[1].lower()
    tmp_name = f"{uuid.uuid4().hex}.{ext}"
    tmp_path = os.path.join(upload_folder, tmp_name)
    file.save(tmp_path)

    try:
        face_svc = FaceDetectionService(
            scale_factor=current_app.config.get("FACE_SCALE_FACTOR", 1.1),
            min_neighbors=current_app.config.get("FACE_MIN_NEIGHBORS", 4),
            min_size=tuple(current_app.config.get("FACE_MIN_SIZE", (48, 48))),
        )
        predictor = EmotionPredictor()
        emotion_colors = current_app.config["EMOTION_COLORS"]

        img_rgb, faces = face_svc.detect_from_file(tmp_path)
        if img_rgb is None:
            return jsonify({"success": False, "error": "Không đọc được file ảnh."}), 400

        predictions = []
        face_results = []
        for (x, y, w, h) in faces:
            face_crop = face_svc.crop_face(img_rgb, (x, y, w, h))
            pred = predictor.predict(face_crop)
            predictions.append(pred)
            face_results.append({
                "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
                "emotion": pred["emotion"],
                "confidence": round(pred["confidence"], 4),
                "probabilities": {k: round(v, 4) for k, v in pred["probabilities"].items()},
            })

        annotated_b64 = None
        if faces:
            annotated = face_svc.draw_results(img_rgb, faces, predictions, emotion_colors)
            annotated_b64 = face_svc.encode_to_base64(annotated)
        else:
            # Không detect được mặt → gửi lại ảnh gốc
            annotated_b64 = face_svc.encode_to_base64(img_rgb)

        return jsonify({
            "success": True,
            "face_count": len(faces),
            "faces": face_results,
            "annotated_image": annotated_b64,
        })

    except Exception as e:
        logger.error("Upload predict error: %s", e)
        return jsonify({"success": False, "error": str(e)}), 500

    finally:
        # Xóa file tạm
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
