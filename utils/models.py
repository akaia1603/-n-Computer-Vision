"""
utils/models.py
===============
Định nghĩa các kiến trúc mô hình cho bài toán nhận dạng cảm xúc khuôn mặt.

Chỉ duy trì kiến trúc:
    1. Baseline CNN: Mạng CNN tùy chỉnh (được cập nhật theo class ConvLayer)

Sử dụng:
    >>> from utils.models import build_baseline_cnn
    >>> model = build_baseline_cnn()
    >>> model.summary()

Yêu cầu: tensorflow, keras
"""

import tensorflow as tf
from tensorflow import keras
from keras import layers, models, regularizers


# ============================================================
# 1. BASELINE CNN — Mạng CNN tùy chỉnh
# ============================================================

class ConvLayer(layers.Layer):
    """
    A 2D convolutional layer followed by Batch Normalization and ReLU activation.
    """
    def __init__(self, kernel_num, kernel_size, strides=1, padding='same', kernel_initializer='he_normal', name=None, **kwargs):
        super(ConvLayer, self).__init__(name=name, **kwargs)
        self.kernel_num = kernel_num
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.kernel_initializer = kernel_initializer
        
        self.conv = layers.Conv2D(kernel_num, kernel_size=kernel_size, strides=strides, padding=padding, kernel_initializer=kernel_initializer, activation=None)
        self.batchnorm = layers.BatchNormalization()
        self.activation = layers.Activation("relu")
    
    def call(self, inputs):
        x = self.conv(inputs)
        x = self.batchnorm(x)
        x = self.activation(x)
        return x

    def get_config(self):
        config = super(ConvLayer, self).get_config()
        config.update({
            "kernel_num": self.kernel_num,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "kernel_initializer": self.kernel_initializer,
        })
        return config

def get_base_model(image_shape):
    """
    Create base model for facial emotion recognition.
    """
    model = keras.Sequential()
    model.add(layers.InputLayer(input_shape=image_shape))

    # block 1 - 64 filters (2 times)
    model.add(ConvLayer(64, kernel_size=(3,3), padding="same", kernel_initializer="he_normal", name="block1_conv1"))
    model.add(ConvLayer(64, kernel_size=(3,3), padding="same", kernel_initializer="he_normal", name="block1_conv2"))
    model.add(layers.MaxPool2D(3,3, name="maxpool_1"))

    # block 2 - 96 filters (3 times)
    model.add(ConvLayer(96, kernel_size=(3,3), padding='same', kernel_initializer='he_normal', name="block2_conv1"))
    model.add(ConvLayer(96, kernel_size=(3,3), padding='same', kernel_initializer='he_normal', name="block2_conv2"))
    model.add(ConvLayer(96, kernel_size=(3,3), padding='same', kernel_initializer='he_normal', name="block2_conv3"))
    model.add(layers.MaxPool2D(3,3, name="maxpool_2"))

    # block 3 - 128 filters (3 times)
    model.add(ConvLayer(128, kernel_size=(3,3), padding='same', kernel_initializer='he_normal', name="block3_conv1"))
    model.add(ConvLayer(128, kernel_size=(3,3), padding='same', kernel_initializer='he_normal', name="block3_conv2"))
    model.add(ConvLayer(128, kernel_size=(3,3), padding='same', kernel_initializer='he_normal', name="block3_conv3"))

    # global average pooling layer
    model.add(layers.GlobalAveragePooling2D(name='GAP'))
    return model

def build_baseline_cnn(input_shape=(100, 100, 3), num_classes=7, dropout_rate=0.25):
    model = get_base_model(input_shape)
    if dropout_rate > 0:
        model.add(layers.Dropout(dropout_rate, name="dropout"))
    model.add(layers.Dense(num_classes, activation='softmax', name="predictions"))
    return model




# ============================================================
# TIỆN ÍCH
# ============================================================

def get_model_summary(model):
    """In tóm tắt model và trả về string."""
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    return "\n".join(stringlist)


def count_parameters(model):
    """Đếm total và trainable parameters."""
    total = model.count_params()
    trainable = sum(
        tf.keras.backend.count_params(w) for w in model.trainable_weights
    )
    non_trainable = total - trainable
    return {
        "total": total,
        "trainable": trainable,
        "non_trainable": non_trainable
    }


if __name__ == "__main__":
    print("=" * 60)
    print("Testing model architectures...")
    print("=" * 60)
    
    # Test 1: Baseline CNN
    print("\n1. Baseline CNN:")
    model1 = build_baseline_cnn()
    params1 = count_parameters(model1)
    print(f"   Total params: {params1['total']:,}")
    print(f"   Trainable: {params1['trainable']:,}")
    
    print("\n All models built successfully!")
