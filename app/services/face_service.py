"""
app/services/face_service.py
=============================
[S] Service Layer — Face detection & preprocessing pipeline.

Sử dụng YOLOv8 cho face detection (với fallback về OpenCV Haar Cascade nếu chưa train).
Nhận ảnh → detect faces → crop từng khuôn mặt → trả về list results.
"""

import os
import logging
import base64
import numpy as np
import cv2

logger = logging.getLogger(__name__)

# Đường dẫn Haar Cascade
_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


class FaceDetectionService:
    """
    Service detect khuôn mặt và chuẩn bị input cho model.

    Hỗ trợ:
        - detect_from_array(): nhận numpy array
        - detect_from_base64(): nhận chuỗi base64 (webcam frame)
        - detect_from_file(): nhận đường dẫn file ảnh
    """

    def __init__(self, model_path="outputs/models/yolo_face.pt", scale_factor=1.1, min_neighbors=4, min_size=(48, 48)):
        self.yolo_model = None
        self.face_cascade = None
        
        try:
            if os.path.exists(model_path):
                from ultralytics import YOLO
                self.yolo_model = YOLO(model_path)
                logger.info(f"Đã load thành công YOLO face model từ: {model_path}")
            else:
                logger.warning(f"Không tìm thấy YOLO model tại {model_path}. Fallback về Haar Cascade.")
                self.face_cascade = cv2.CascadeClassifier(_CASCADE_PATH)
                if self.face_cascade.empty():
                    logger.error("Không load được Haar Cascade từ: %s", _CASCADE_PATH)
        except Exception as e:
            logger.error(f"Lỗi khởi tạo YOLO model: {e}. Fallback về Haar Cascade.")
            self.face_cascade = cv2.CascadeClassifier(_CASCADE_PATH)

        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def detect_from_base64(self, b64_string):
        """
        Decode base64 image → detect faces.

        Args:
            b64_string: chuỗi base64 (có hoặc không có data URI prefix)

        Returns:
            (img_rgb, faces) hoặc (None, []) nếu lỗi
        """
        try:
            # Strip data URI prefix nếu có
            if "," in b64_string:
                b64_string = b64_string.split(",", 1)[1]
            img_bytes = base64.b64decode(b64_string)
            arr = np.frombuffer(img_bytes, dtype=np.uint8)
            img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img_bgr is None:
                return None, []
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            faces = self._detect(img_bgr)
            return img_rgb, faces
        except Exception as e:
            logger.error("decode base64 error: %s", e)
            return None, []

    def detect_from_array(self, img_bgr):
        """
        Detect từ numpy array BGR.

        Returns:
            (img_rgb, faces)
        """
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        faces = self._detect(img_bgr)
        return img_rgb, faces

    def detect_from_file(self, file_path):
        """
        Detect từ đường dẫn file ảnh.

        Returns:
            (img_rgb, faces) hoặc (None, []) nếu lỗi
        """
        img_bgr = cv2.imread(file_path)
        if img_bgr is None:
            logger.error("Không đọc được ảnh: %s", file_path)
            return None, []
        return self.detect_from_array(img_bgr)

    def crop_face(self, img_rgb, face_box, padding=0.15):
        """
        Crop vùng mặt (với padding) từ ảnh RGB.

        Args:
            img_rgb:  numpy array (H, W, 3)
            face_box: (x, y, w, h) từ OpenCV
            padding:  tỉ lệ padding xung quanh mặt

        Returns:
            face_img: numpy array (H', W', 3) RGB
        """
        x, y, w, h = face_box
        h_img, w_img = img_rgb.shape[:2]
        pad_x = int(w * padding)
        pad_y = int(h * padding)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w_img, x + w + pad_x)
        y2 = min(h_img, y + h + pad_y)
        return img_rgb[y1:y2, x1:x2]

    def draw_results(self, img_rgb, faces, predictions, emotion_colors=None):
        """
        Vẽ bounding box + emotion label lên ảnh.

        Args:
            img_rgb:      numpy array RGB
            faces:        list of (x, y, w, h)
            predictions:  list of dicts (output của EmotionPredictor.predict)
            emotion_colors: dict {emotion: hex_color}

        Returns:
            img_annotated: numpy array RGB
        """
        img_out = img_rgb.copy()
        default_colors = {
            "Surprise": (255, 217, 61), "Fear": (255, 107, 107),
            "Disgust": (107, 203, 119), "Happiness": (77, 150, 255),
            "Sadness": (155, 89, 182), "Anger": (231, 76, 60),
            "Neutral": (149, 165, 166)
        }

        for (x, y, w, h), pred in zip(faces, predictions):
            emotion = pred.get("emotion", "Unknown")
            confidence = pred.get("confidence", 0)

            # Màu sắc
            if emotion_colors and emotion in emotion_colors:
                hex_c = emotion_colors[emotion].lstrip("#")
                color = tuple(int(hex_c[i:i+2], 16) for i in (0, 2, 4))
            else:
                color = default_colors.get(emotion, (200, 200, 200))

            # Bounding box
            cv2.rectangle(img_out, (x, y), (x + w, y + h), color, 2)

            # Label background
            label = f"{emotion} {confidence:.0%}"
            (lw, lh), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2
            )
            ly = max(y - 10, lh + 5)
            cv2.rectangle(
                img_out,
                (x, ly - lh - baseline - 4),
                (x + lw + 4, ly + baseline - 4),
                color, -1
            )
            cv2.putText(
                img_out, label,
                (x + 2, ly - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (255, 255, 255) if sum(color) < 380 else (0, 0, 0),
                2, cv2.LINE_AA
            )

        return img_out

    def encode_to_base64(self, img_rgb, quality=85):
        """Encode numpy RGB array → base64 JPEG string."""
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"

    # ----------------------------------------------------------
    # Private
    # ----------------------------------------------------------

    def _detect(self, img_bgr):
        """Detect faces trong ảnh BGR. Trả về list (x, y, w, h)."""
        if self.yolo_model is not None:
            # Dùng YOLOv8
            results = self.yolo_model(img_bgr, verbose=False)
            faces = []
            if len(results) > 0:
                boxes = results[0].boxes
                for box in boxes:
                    # Chuyển (x1, y1, x2, y2) sang (x, y, w, h)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    w = int(x2 - x1)
                    h = int(y2 - y1)
                    faces.append([int(x1), int(y1), w, h])
            
            # Sắp xếp theo diện tích mặt (lớn nhất trước)
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            return faces
        else:
            # Fallback dùng Haar Cascade
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors,
                minSize=self.min_size,
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            if len(faces) == 0:
                return []
            # Sắp xếp theo diện tích mặt (lớn nhất trước)
            faces = sorted(faces.tolist(), key=lambda f: f[2] * f[3], reverse=True)
            return faces
