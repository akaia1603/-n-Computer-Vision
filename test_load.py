import sys
import logging
import traceback
import tensorflow as tf

logging.basicConfig(level=logging.INFO)

print("Testing model load...")
try:
    from app.models.emotion_model import EmotionPredictor
    predictor = EmotionPredictor()
    print("Is loaded:", predictor.is_loaded)
    if not predictor.is_loaded:
        print("Model failed to load, checking directly...")
        from utils.models import ConvLayer
        model = tf.keras.models.load_model('outputs/models/baseline_cnn.keras', custom_objects={'ConvLayer': ConvLayer})
        print("Model loaded directly!")
except Exception as e:
    print("ERROR OCCURRED:")
    traceback.print_exc()
