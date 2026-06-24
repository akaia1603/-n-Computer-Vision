import logging
from flask import current_app
from app.services.face_service import FaceDetectionService
from app.services.emotion_smoother import EmotionSmoother
from app.models.emotion_model import EmotionPredictor
from app.models.dan_model import DANPredictor
from app.models.poster_model import POSTERPredictor

logger = logging.getLogger(__name__)

_PREDICTORS = {
    "keras": EmotionPredictor,
    "dan": DANPredictor,
    "poster": POSTERPredictor,
}


def get_predictor(model_type):
    cls = _PREDICTORS.get(model_type)
    if cls is None:
        cls = EmotionPredictor
    return cls()


class PredictionService:
    _last_model_type = None

    def __init__(self, model_type="keras"):
        self.model_type = model_type
        self.predictor = get_predictor(model_type)
        self.face_svc = FaceDetectionService(
            scale_factor=current_app.config.get("FACE_SCALE_FACTOR", 1.1),
            min_neighbors=current_app.config.get("FACE_MIN_NEIGHBORS", 4),
            min_size=tuple(current_app.config.get("FACE_MIN_SIZE", (48, 48))),
        )
        self.smoother = EmotionSmoother()
        if PredictionService._last_model_type is not None and PredictionService._last_model_type != model_type:
            self.smoother.reset()
        PredictionService._last_model_type = model_type

    def predict_from_base64(self, b64_image, reset_smoother=False, realtime=True):
        if reset_smoother:
            self.smoother.reset()
        img_rgb, faces = self.face_svc.detect_from_base64(b64_image)
        if img_rgb is None:
            return None, [], "Could not decode image"
        return self._predict_faces(img_rgb, faces, realtime=realtime)

    def predict_from_file(self, file_path):
        img_rgb, faces = self.face_svc.detect_from_file(file_path)
        if img_rgb is None:
            return None, [], "Could not read image"
        return self._predict_faces(img_rgb, faces, realtime=False)

    def predict_from_array(self, img_bgr):
        img_rgb, faces = self.face_svc.detect_from_array(img_bgr)
        return self._predict_faces(img_rgb, faces, realtime=True)

    def _predict_faces(self, img_rgb, faces, realtime=True):
        face_results = []

        for (x, y, w, h) in faces:
            face_crop = self.face_svc.crop_face(img_rgb, (x, y, w, h))
            pred = self.predictor.predict(face_crop)
            face_results.append({
                "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
                "emotion": pred["emotion"],
                "confidence": round(pred["confidence"], 4),
                "probabilities": {k: round(v, 4) for k, v in pred["probabilities"].items()},
            })

        if realtime:
            return face_results, None, None

        emotion_colors = current_app.config["EMOTION_COLORS"]
        smoothed_results = []
        conf_threshold = current_app.config.get("CONFIDENCE_THRESHOLD", 0.2)
        for fr in face_results:
            smoothed = self.smoother.smooth(fr["probabilities"], fr["emotion"], fr["confidence"])
            is_low = smoothed["confidence"] < conf_threshold
            smoothed_results.append({
                "bbox": fr["bbox"],
                "emotion": "Analyzing..." if is_low else smoothed["emotion"],
                "confidence": round(smoothed["confidence"], 4),
                "probabilities": {k: round(v, 4) for k, v in smoothed["probabilities"].items()},
                "low_confidence": is_low,
            })

        annotated = None
        if faces:
            annotated = self.face_svc.draw_results(img_rgb, faces, [fr for fr in face_results], emotion_colors)

        return smoothed_results, annotated, None

    def encode_annotated(self, annotated):
        return self.face_svc.encode_to_base64(annotated)
