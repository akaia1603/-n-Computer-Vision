import logging

logger = logging.getLogger(__name__)


class EmotionSmoother:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, alpha=0.35, confidence_threshold=0.4, hold_frames=3):
        if self._initialized:
            return
        self._initialized = True
        self.alpha = alpha
        self.confidence_threshold = confidence_threshold
        self.hold_frames = hold_frames
        self.reset()

    def reset(self):
        self.ema_probs = None
        self.last_emotion = None
        self.hold_counter = 0

    def smooth(self, probs, top_emotion=None, top_confidence=None):
        if self.ema_probs is None:
            self.ema_probs = dict(probs)
            if top_emotion is None:
                top_emotion = max(probs, key=probs.get)
            self.last_emotion = top_emotion
            self.hold_counter = 0
            return {
                "emotion": top_emotion,
                "confidence": top_confidence or probs[top_emotion],
                "probabilities": dict(probs),
            }

        for k in probs:
            prev = self.ema_probs.get(k, 0.0)
            self.ema_probs[k] = self.alpha * probs[k] + (1 - self.alpha) * prev

        smoothed_top = max(self.ema_probs, key=self.ema_probs.get)
        smoothed_conf = self.ema_probs[smoothed_top]

        if smoothed_top != self.last_emotion:
            if smoothed_conf > self.confidence_threshold and self.hold_counter >= self.hold_frames:
                self.last_emotion = smoothed_top
                self.hold_counter = 0
            else:
                smoothed_top = self.last_emotion
                smoothed_conf = self.ema_probs[smoothed_top]
                self.hold_counter += 1
        else:
            self.hold_counter = 0

        return {
            "emotion": smoothed_top,
            "confidence": smoothed_conf,
            "probabilities": dict(self.ema_probs),
        }
