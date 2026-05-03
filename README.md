# Facial Emotion Recognition — Real-time Demo

> **Nhận dạng cảm xúc khuôn mặt theo thời gian thực** sử dụng deep learning trên dataset **RAF-DB**.

![Python](https://img.shields.io/badge/Python-3.11-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange) ![Flask](https://img.shields.io/badge/Flask-MVC-green) ![OpenCV](https://img.shields.io/badge/OpenCV-4.12-red)

---

## 🎯 Mục tiêu

Xây dựng hệ thống nhận dạng **7 cảm xúc khuôn mặt** (Surprise, Fear, Disgust, Happiness, Sadness, Anger, Neutral) với:
- ✅ Độ chính xác ≥ **80%** trên test set
- ✅ Tốc độ xử lý ≥ **15 FPS** trong inference
- ✅ Pipeline hoàn chỉnh: EDA → Training → Evaluation → Web Demo

---

## 📁 Cấu trúc dự án

```
Computer Vision Demo/
├── 01_problem_statement.md       ✅ Phát biểu bài toán
├── 02_eda_and_preprocessing.py   ✅ EDA & Preprocessing
├── 02_eda_and_preprocessing.md   ✅ Tài liệu EDA
├── 03_baseline_model.py          ✅ Baseline CNN training
├── 03_baseline_model.md          ✅ Tài liệu baseline
├── 04_model_experiments.py       ✅ ResNet50 + MobileNetV2+CBAM
├── 04_model_experiments.md       ✅ Tài liệu thực nghiệm
├── 05_evaluation_analysis.py     ✅ Evaluation + Grad-CAM
├── 05_evaluation_analysis.md     ✅ Tài liệu đánh giá
├── 06_web_demo.md                ✅ Tài liệu web demo
├── app/                          ✅ Flask MVC Web Demo
│   ├── __init__.py               App factory
│   ├── config.py                 Cấu hình
│   ├── run.py                    Entry point
│   ├── controllers/              Route handlers (Blueprint)
│   ├── models/                   ML inference logic
│   ├── services/                 Face detection service
│   ├── views/templates/          HTML templates
│   └── static/                   CSS + JS
├── utils/                        Shared utilities
│   ├── data_loader.py            Data loading & augmentation
│   ├── models.py                 Model architectures
│   ├── training.py               Training pipeline
│   ├── evaluation.py             Metrics & visualization
│   └── gradcam.py                Grad-CAM implementation
├── data/                         RAF-DB dataset
│   ├── train/                    12,271 ảnh training
│   └── test/                     3,068 ảnh test
├── outputs/
│   ├── eda/                      EDA charts
│   ├── models/                   Saved .keras models
│   ├── training_logs/            Loss/accuracy curves
│   ├── evaluation/               Confusion matrix, Grad-CAM
│   └── experiments/              Model comparison results
└── requirements.txt
```

---

## 🚀 Bắt đầu nhanh

### 1. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 2. Chạy EDA (xem phân tích dataset)

```bash
python 02_eda_and_preprocessing.py
```

### 3. Train Baseline CNN (~30-60 phút CPU)

```bash
python 03_baseline_model.py
```

### 4. Train Transfer Learning Models (~60-120 phút CPU)

```bash
python 04_model_experiments.py
```

### 5. Đánh giá toàn diện + Grad-CAM

```bash
python 05_evaluation_analysis.py
```

### 6. Chạy Web Demo

```bash
python app/run.py
```

Mở trình duyệt: **http://127.0.0.1:5000**

---

## 🏗️ Kiến trúc Models

| Model | Params | Input | Đặc điểm |
|-------|--------|-------|-----------|
| **Baseline CNN** | ~500K | 100×100 | 4 Conv blocks, nhanh, nhẹ |
| **ResNet50** | ~23M | 224×224 | Transfer learning ImageNet, skip connections |
| **MobileNetV2+CBAM** | ~3.4M | 224×224 | Attention mechanism, balance accuracy/speed |

### CBAM Attention Module

```
Feature Map [H, W, C]
    ├── Channel Attention → "Kênh nào quan trọng?"
    └── Spatial Attention → "Vùng nào quan trọng?"
            → Tập trung vào mắt, miệng, lông mày
```

---

## 🌐 Flask MVC Architecture

```
Request → Controller → Service/Model → Controller → Response
                ↓              ↓
           (Route)    (FaceDetection)  (EmotionPredictor)
                               ↓
                          View (Template)
```

**API Endpoints:**

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| `GET`  | `/` | Real-time webcam page |
| `GET`  | `/upload` | Upload ảnh page |
| `POST` | `/api/predict` | Predict từ webcam frame (base64) |
| `POST` | `/api/upload` | Predict từ file upload |
| `GET`  | `/api/info` | Thông tin model |

---

## 📊 Dataset: RAF-DB

- **Tổng**: 15,339 ảnh khuôn mặt
- **Train**: 12,271 ảnh | **Test**: 3,068 ảnh
- **7 classes**: Surprise, Fear, Disgust, Happiness, Sadness, Anger, Neutral
- **Vấn đề**: Dataset mất cân bằng → xử lý bằng class weights + augmentation

---

## 🔧 Môi trường

- Python 3.11.5
- TensorFlow 2.21.0 + Keras 3.13.2 (CPU only)
- OpenCV 4.12
- scikit-learn 1.7
- Flask 3.x

> ⚠️ **Không có GPU**: Training trên CPU — dùng EarlyStopping + ReduceLROnPlateau để tối ưu thời gian.

---

## 📈 Pipeline hoàn chỉnh

```
RAF-DB Dataset
    │
    ▼ 02_eda_and_preprocessing.py
EDA: Phân bố, chất lượng ảnh, augmentation preview
    │
    ▼ 03_baseline_model.py
Baseline CNN (100×100) → outputs/models/baseline_cnn.keras
    │
    ▼ 04_model_experiments.py
ResNet50 + MobileNetV2+CBAM → outputs/models/
    │
    ▼ 05_evaluation_analysis.py
Confusion Matrix + ROC-AUC + Grad-CAM + Error Analysis
    │
    ▼ app/run.py
Flask Web Demo: Real-time webcam + Image upload
```

---

*Computer Vision Demo — Nhận dạng Cảm xúc Khuôn mặt theo Thời gian thực*
