import os
import uuid
import logging
from flask import Blueprint, request, jsonify, current_app
from app.services.prediction_service import PredictionService
from app.db import save_prediction, get_history, get_stats

logger = logging.getLogger(__name__)
predict_bp = Blueprint("predict", __name__)


def _allowed_file(filename):
    allowed = current_app.config.get("ALLOWED_EXTENSIONS", {"jpg", "jpeg", "png"})
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed


def _get_model_type():
    model_type = "keras"
    if request.is_json:
        model_type = request.get_json(silent=True).get("model_type", "keras")
    else:
        model_type = request.form.get("model_type", "keras")
    return model_type


@predict_bp.route("/api/predict", methods=["POST"])
def predict_webcam():
    data = request.get_json(silent=True) or {}
    b64_image = data.get("image", "")
    if not b64_image:
        return jsonify({"success": False, "error": "No image data."}), 400

    model_type = data.get("model_type", "keras")
    reset_smoother = data.get("reset_smoother", False)

    svc = PredictionService(model_type=model_type)
    face_results, annotated, error = svc.predict_from_base64(b64_image, reset_smoother=reset_smoother, realtime=True)

    if error:
        return jsonify({"success": False, "error": error}), 400

    return jsonify({
        "success": True,
        "face_count": len(face_results),
        "faces": face_results,
    })


@predict_bp.route("/api/upload", methods=["POST"])
def predict_upload():
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded."}), 400

    file = request.files["file"]
    if file.filename == "" or not _allowed_file(file.filename):
        return jsonify({
            "success": False,
            "error": "Invalid file. Supported: PNG, JPG, JPEG, WEBP."
        }), 400

    upload_folder = current_app.config["UPLOAD_FOLDER"]
    os.makedirs(upload_folder, exist_ok=True)
    ext = file.filename.rsplit(".", 1)[1].lower()
    tmp_name = f"{uuid.uuid4().hex}.{ext}"
    tmp_path = os.path.join(upload_folder, tmp_name)
    file.save(tmp_path)

    try:
        model_type = request.form.get("model_type", "keras")
        svc = PredictionService(model_type=model_type)
        face_results, annotated, error = svc.predict_from_file(tmp_path)

        if error:
            return jsonify({"success": False, "error": error}), 400

        annotated_b64 = None
        if annotated is not None:
            annotated_b64 = svc.encode_annotated(annotated)
        else:
            img_rgb, _ = svc.face_svc.detect_from_file(tmp_path)
            if img_rgb is not None:
                annotated_b64 = svc.encode_annotated(img_rgb)

        return jsonify({
            "success": True,
            "face_count": len(face_results),
            "faces": face_results,
            "annotated_image": annotated_b64,
        })

    except Exception as e:
        logger.error("Upload predict error: %s", e)
        return jsonify({"success": False, "error": str(e)}), 500

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@predict_bp.route("/api/save-result", methods=["POST"])
def save_result():
    data = request.get_json(silent=True) or {}
    emotion = data.get("emotion", "Unknown")
    confidence = data.get("confidence", 0.0)
    probabilities = data.get("probabilities", {})
    model_type = data.get("model_type", "keras")
    face_count = data.get("face_count", 1)
    source = data.get("source", "webcam")

    if confidence < 0.3:
        return jsonify({"success": False, "error": "Confidence too low, not saved."})

    save_prediction(emotion, confidence, probabilities, model_type, face_count, source)
    return jsonify({"success": True})


@predict_bp.route("/api/history", methods=["GET"])
def predict_history():
    return jsonify({"success": True, "history": get_history()})


@predict_bp.route("/api/stats", methods=["GET"])
def predict_stats():
    return jsonify({"success": True, **get_stats()})
