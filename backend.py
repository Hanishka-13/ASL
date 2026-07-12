import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pyttsx3
import os
from collections import deque
from datetime import datetime

# ======================================================
# Config
# ======================================================

CONFIDENCE_THRESHOLD = 0.70
IGNORED_LABELS = {"nothing"}
WINDOW_SIZE = 6
STABILITY_RATIO = 0.6

MODEL_PATH = "models/sign_model.keras"
LABELS_PATH = "models/labels.txt"
SAVE_PATH = "downloads/translation.txt"


# ======================================================
# Model + labels
# ======================================================

def load_model_and_labels():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH, "r") as f:
        labels = [line.strip() for line in f]
    return model, labels


# ======================================================
# MediaPipe Hands
# ======================================================

def load_hands():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )
    mp_draw = mp.solutions.drawing_utils
    return mp_hands, hands, mp_draw


# ======================================================
# Text-to-Speech
# ======================================================
# NOTE: pyttsx3 engines can only safely runAndWait() once per
# instance. Reusing a single cached engine across Streamlit
# reruns causes "run loop already started". So we spin up a
# fresh engine every time speak() is called instead of caching it.

def speak(text):
    if not text.strip():
        return
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)
    engine.setProperty("volume", 1.0)
    engine.say(text)
    engine.runAndWait()
    engine.stop()


# ======================================================
# Landmark extraction
# ======================================================

def extract_landmarks(results):
    """
    Extract 126 features (2 hands x 21 landmarks x 3 coordinates).
    Uses RAW MediaPipe landmark values directly (no centering, no
    scaling) to match how dataset/landmarks.csv was generated during
    training.
    """
    features = []

    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks[:2]:
            for lm in hand.landmark:
                features.extend([lm.x, lm.y, lm.z])

    while len(features) < 126:
        features.append(0.0)

    return np.array(features, dtype=np.float32)


# ======================================================
# Prediction
# ======================================================

def predict_frame(frame, hands, model, labels, mp_draw, mp_hands, draw_landmarks=True):
    """
    Runs one frame through MediaPipe + the model.
    Returns: annotated_frame, prediction (str), confidence (float), results
    """
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    features = extract_landmarks(results)

    prediction = "-"
    confidence = 0.0

    if np.any(features):
        input_data = np.expand_dims(features, axis=0)
        output = model.predict(input_data, verbose=0)
        index = np.argmax(output)
        confidence = float(output[0][index])
        raw_label = labels[index]

        if confidence >= CONFIDENCE_THRESHOLD and raw_label not in IGNORED_LABELS:
            prediction = raw_label

    if draw_landmarks and results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

    return frame, prediction, confidence, results


def get_stable_prediction(buffer):
    counts = {}
    for label in buffer:
        if label == "-":
            continue
        counts[label] = counts.get(label, 0) + 1

    if not counts:
        return None

    best_label = max(counts, key=counts.get)
    best_count = counts[best_label]

    if best_count / len(buffer) >= STABILITY_RATIO:
        return best_label

    return None


def apply_prediction(sentence, predicted_label):
    """
    Applies a stable prediction to the sentence list.
    - 'del'   -> deletes the last character
    - 'space' -> inserts an actual space character
    - anything else -> appended as a normal letter
    """
    if predicted_label == "del":
        if sentence:
            sentence.pop()

    elif predicted_label == "space":
        sentence.append(" ")

    else:
        sentence.append(predicted_label)

    return sentence


def new_buffer():
    return deque(maxlen=WINDOW_SIZE)


# ======================================================
# Save to file (+ create folder if missing)
# ======================================================

def save_to_file(text):
    if not text.strip():
        return None

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(SAVE_PATH, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {text}\n")

    return timestamp