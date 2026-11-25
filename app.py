"""
app.py
-------
Real-time sign language detection using your webcam.
Loads the trained model from model.pkl, tracks your hand with MediaPipe,
and displays the predicted sign live on screen.

HOW TO USE:
    python app.py

Press 'q' to quit.
"""

import cv2
import mediapipe as mp
import numpy as np
import pickle
import os

MODEL_PATH = "model.pkl"

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"{MODEL_PATH} not found. Run collect_data.py then train_model.py first."
        )
    with open(MODEL_PATH, "rb") as f:
        data = pickle.load(f)
    return data["model"], data["encoder"]


def landmarks_to_features(hand_landmarks):
    row = []
    for lm in hand_landmarks.landmark:
        row.extend([lm.x, lm.y, lm.z])
    return np.array(row).reshape(1, -1)


def main():
    model, encoder = load_model()

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    CONFIDENCE_THRESHOLD = 0.6

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        prediction_text = "No hand detected"
        color = (0, 0, 255)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                features = landmarks_to_features(hand_landmarks)
                probs = model.predict_proba(features)[0]
                best_idx = np.argmax(probs)
                confidence = probs[best_idx]

                if confidence >= CONFIDENCE_THRESHOLD:
                    label = encoder.inverse_transform([best_idx])[0]
                    prediction_text = f"{label} ({confidence * 100:.0f}%)"
                    color = (0, 255, 0)
                else:
                    prediction_text = f"Uncertain ({confidence * 100:.0f}%)"
                    color = (0, 165, 255)

        cv2.rectangle(frame, (0, 0), (frame.shape[1], 60), (0, 0, 0), -1)
        cv2.putText(frame, prediction_text, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        cv2.imshow("Sign Language Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
