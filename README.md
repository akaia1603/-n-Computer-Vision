# Facial Emotion Recognition — Real-time Demo

> **Nhận dạng cảm xúc khuôn mặt theo thời gian thực** sử dụng deep learning trên dataset **RAF-DB**.

![Python](https://img.shields.io/badge/Python-3.11-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red) ![Flask](https://img.shields.io/badge/Flask-MVC-green) ![OpenCV](https://img.shields.io/badge/OpenCV-4.12-red)

## Mục tiêu

Xây dựng hệ thống nhận dạng **7 cảm xúc khuôn mặt** (Surprise, Fear, Disgust, Happiness, Sadness, Anger, Neutral) với 3 kiến trúc:
- **Baseline CNN** (TensorFlow) — ~77% test accuracy
- **DAN** — Distract Your Attention Network (PyTorch, Colab)
- **POSTER** — Pyramid fusiOn transformer for facial ExPRession (PyTorch, ~91% val accuracy)

## Cấu trúc dự án

```
Computer Vision Demo/
├── 01_preprocessing.ipynb           Preview augmentation pipeline
├── 02_eda.ipynb                     Exploratory Data Analysis
├── 03_Train_CNN_model.ipynb         Train Baseline CNN (TensorFlow)
├── 04_train_dan.ipynb               Train DAN on RAF-DB (PyTorch, Colab)
├── 05_fine_tune_dan_colab.ipynb     Pre-train FER2013 → Fine-tune RAF-DB
├── 06_poster_train.ipynb            Train POSTER model (PyTorch)
├── utils/                           Shared utilities
│   ├── data_loader.py               Data loading & augmentation (TF)
│   ├── models.py                    CNN architectures (ConvLayer, baseline)
│   ├── training.py                  Training pipeline & callbacks
│   ├── evaluation.py                Metrics, confusion matrix, ROC
│   └── gradcam.py                   Grad-CAM visualization
├── models/                          (THIẾU) Source code từ repo POSTER
│   ├── ir50.py
│   ├── mobilefacenet.py
│   └── hyp_crossvit.py
├── pretrain/                        (THIẾU) Pretrained weights cho POSTER
│   ├── ir50.pth
│   └── mobilefacenet_model_best.pth.tar
├── data/
│   ├── DATASET/train/               12,271 ảnh training (7 class folders)
│   ├── DATASET/test/                3,068 ảnh test
│   ├── train_labels.csv
│   └── test_labels.csv
├── modelcheckpoints/                Checkpoints từ CNN training
├── checkpoints/                     Checkpoints từ POSTER training
├── outputs/
│   ├── eda/                         EDA charts
│   ├── models/                      Saved .keras models
│   ├── evaluation/                  Confusion matrix, ROC
│   └── training_logs/
├── log/                             CSV logs từ CNN training
├── app/                             Flask Web Demo
│   ├── config.py                    Cấu hình
│   ├── run.py                       Entry point
│   ├── controllers/                 Route handlers
│   ├── models/                      ML inference logic
│   ├── services/                    Face detection service
│   ├── views/                       HTML templates
│   └── static/                      CSS + JS
├── requirements.txt
└── README.md
```

## Cách chạy

### 1. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### 2. EDA
Mở `02_eda.ipynb` trong Jupyter.

### 3. Train Baseline CNN
Mở `03_Train_CNN_model.ipynb` và Run All.

### 4. Train DAN (trên Colab)
Upload `04_train_dan.ipynb` lên Google Colab, mount Drive, chạy.

### 5. Train POSTER
1. Clone model source từ [POSTER repo](https://github.com/zengqunzhao/Poster)
2. Copy `models/*.py` vào `models/`
3. Download pretrained weights vào `pretrain/`
4. Mở `06_poster_train.ipynb` và Run All

### 6. Chạy Web Demo
```bash
python app/run.py
```
Mở trình duyệt: **http://127.0.0.1:5000**

## Models

| Model | Framework | Params | Test Acc | Ghi chú |
|-------|-----------|--------|----------|---------|
| Baseline CNN | TensorFlow | ~670K | 76.89% | 3 ConvBlocks + GAP |
| DAN | PyTorch | ~11M | — | ResNet18 + 4-head attention |
| POSTER | PyTorch | ~58M | 91.13% | IR-50 + MobileFaceNet + HyViT |

## Dataset: RAF-DB

- **15,339** ảnh khuôn mặt in-the-wild
- **Train**: 12,271 | **Test**: 3,068
- **7 classes**: Surprise, Fear, Disgust, Happiness, Sadness, Anger, Neutral
- Mất cân bằng nặng (Fear chỉ 281 ảnh, Happiness có 4,772)

## Thiếu sót cần bổ sung

- [ ] Clone `models/ir50.py`, `models/mobilefacenet.py`, `models/hyp_crossvit.py` từ POSTER repo
- [ ] Download `pretrain/ir50.pth`, `pretrain/mobilefacenet_model_best.pth.tar`
- [ ] Dataset FER2013 cho fine-tune notebook 5

## Môi trường

- Python 3.11
- TensorFlow 2.21 + Keras 3.13
- PyTorch 2.0+
- OpenCV 4.12
- Flask 3.x
- scikit-learn, seaborn

---

*Computer Vision Demo — Nhận dạng Cảm xúc Khuôn mặt*
