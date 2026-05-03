"""
utils package
=============
Các module tiện ích dùng chung cho toàn bộ dự án FER.

Modules:
    - data_loader: Load dữ liệu, augmentation, preprocessing
    - models: Định nghĩa kiến trúc CNN, ResNet50, MobileNetV2+Attention
    - training: Pipeline huấn luyện với callbacks
    - evaluation: Metrics, confusion matrix, classification report
    - gradcam: Grad-CAM visualization
"""

from . import data_loader
from . import models
from . import training
from . import evaluation
from . import gradcam
