import cv2
import numpy as np
from tensorflow import keras
from collections import deque
import time

class WebcamDigitRecognizer:
    def __init__(self, model_path):
        self.model = keras.models.load_model(model_path)
        print("[INFO] Model loaded")

        self.cap = None
        self.running = False

        self.pred_history = deque(maxlen=20)
        self.conf_history = deque(maxlen=20)

        self.last_time = time.time()
        self.fps = 0

    def preprocess(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11, 2
        )

        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        contours = [c for c in contours if cv2.contourArea(c) > 500]
        if not contours:
            return None, None

        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)

        digit = thresh[y:y+h, x:x+w]

        size = max(w, h)
        square = np.zeros((size, size), dtype=np.uint8)
        square[(size-h)//2:(size-h)//2+h,
               (size-w)//2:(size-w)//2+w] = digit

        digit_28 = cv2.resize(square, (28, 28))
        digit_28 = digit_28.astype("float32") / 255.0

        return digit_28, (x, y, w, h)

    def predict(self, digit_img):
        img = digit_img.reshape(1, 28, 28, 1)
        preds = self.model.predict(img, verbose=0)

        digit = int(np.argmax(preds))
        conf = float(np.max(preds) * 100)

        if conf > 60:
            self.pred_history.append(digit)
            self.conf_history.append(conf)

        if not self.pred_history:
            return None, None

        stable_digit = max(set(self.pred_history), key=self.pred_history.count)
        stable_conf = max(self.conf_history)

        return stable_digit, stable_conf

    def calculate_fps(self):
        now = time.time()
        self.fps = 1 / (now - self.last_time)
        self.last_time = now
        return int(self.fps)
