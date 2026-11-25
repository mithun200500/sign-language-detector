"""
collect_data.py
----------------
Collects hand landmark data using the modern MediaPipe Tasks API.

HOW TO USE:
1. Run:
       python collect_data.py

2. Enter a label, for example:
       A

3. Press:
       s = start collecting
       p = pause collecting
       n = enter a new label
       q = quit and save

Data is saved to:
    data/landmarks.csv
"""

import cv2
import mediapipe as mp
import csv
import os
import urllib.request


# ==================================================
# PATHS
# ==================================================

DATA_DIR = "data"
CSV_PATH = os.path.join(DATA_DIR, "landmarks.csv")

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "hand_landmarker.task")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ==================================================
# DOWNLOAD MEDIAPIPE HAND MODEL
# ==================================================

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/"
    "hand_landmarker.task"
)


def download_model():

    if os.path.exists(MODEL_PATH):
        return

    print("MediaPipe hand model not found.")
    print("Downloading hand_landmarker.task...")

    try:
        urllib.request.urlretrieve(
            MODEL_URL,
            MODEL_PATH
        )

        print("Model downloaded successfully.\n")

    except Exception as e:

        print("ERROR: Could not download MediaPipe model.")
        print(e)

        raise


download_model()


# ==================================================
# MEDIAPIPE HAND LANDMARKER
# ==================================================

BaseOptions = mp.tasks.BaseOptions

HandLandmarker = mp.tasks.vision.HandLandmarker

HandLandmarkerOptions = (
    mp.tasks.vision.HandLandmarkerOptions
)

RunningMode = mp.tasks.vision.RunningMode


options = HandLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),

    running_mode=RunningMode.VIDEO,

    num_hands=1,

    min_hand_detection_confidence=0.7,

    min_hand_presence_confidence=0.5,

    min_tracking_confidence=0.5
)


# ==================================================
# HAND CONNECTIONS
# ==================================================

HAND_CONNECTIONS = [

    # Thumb
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),

    # Index finger
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),

    # Middle finger
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),

    # Ring finger
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),

    # Little finger
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),

    # Palm
    (0, 17)
]


# ==================================================
# CONVERT LANDMARKS TO CSV ROW
# ==================================================

def landmarks_to_row(hand_landmarks, label):

    row = []

    for landmark in hand_landmarks:

        row.extend([
            landmark.x,
            landmark.y,
            landmark.z
        ])

    row.append(label)

    return row


# ==================================================
# CREATE CSV HEADER
# ==================================================

def ensure_header(csv_path):

    if not os.path.exists(csv_path):

        header = []

        for i in range(21):

            header.extend([
                f"x{i}",
                f"y{i}",
                f"z{i}"
            ])

        header.append("label")

        with open(
            csv_path,
            "w",
            newline=""
        ) as f:

            writer = csv.writer(f)

            writer.writerow(header)


# ==================================================
# DRAW LANDMARKS
# ==================================================

def draw_landmarks(frame, hand_landmarks):

    height, width, _ = frame.shape

    points = []

    # Convert normalized coordinates
    # to screen coordinates

    for landmark in hand_landmarks:

        x = int(landmark.x * width)

        y = int(landmark.y * height)

        points.append((x, y))

        # Draw point

        cv2.circle(
            frame,
            (x, y),
            5,
            (0, 255, 0),
            -1
        )

    # Draw connections

    for start, end in HAND_CONNECTIONS:

        if (
            start < len(points)
            and end < len(points)
        ):

            cv2.line(
                frame,
                points[start],
                points[end],
                (255, 0, 0),
                2
            )


# ==================================================
# MAIN
# ==================================================

def main():

    # Create CSV if it doesn't exist

    ensure_header(CSV_PATH)


    # Open webcam

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        print("ERROR: Could not open webcam.")

        return


    # Ask for label

    label = input(
        "Enter the label for the sign "
        "(example: A): "
    ).strip()


    if not label:

        print("ERROR: Label cannot be empty.")

        cap.release()

        return


    collecting = False

    samples_collected = 0

    print("\nControls:")

    print("  s -> start/resume collecting")

    print("  p -> pause collecting")

    print("  n -> enter a NEW label")

    print("  q -> quit and save\n")


    # Timestamp for VIDEO mode

    timestamp_ms = 0


    # Create MediaPipe detector

    with HandLandmarker.create_from_options(
        options
    ) as landmarker:


        # Open CSV

        with open(
            CSV_PATH,
            "a",
            newline=""
        ) as f:


            writer = csv.writer(f)


            # Main webcam loop

            while True:


                success, frame = cap.read()


                if not success:

                    print(
                        "ERROR: Could not read webcam frame."
                    )

                    break


                # Mirror webcam

                frame = cv2.flip(
                    frame,
                    1
                )


                # Convert BGR → RGB

                rgb_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )


                # Convert to MediaPipe image

                mp_image = mp.Image(

                    image_format=(
                        mp.ImageFormat.SRGB
                    ),

                    data=rgb_frame
                )


                # Detect hand

                result = landmarker.detect_for_video(

                    mp_image,

                    timestamp_ms
                )


                # Increase timestamp

                timestamp_ms += 33


                # Check for detected hands

                if result.hand_landmarks:


                    for hand_landmarks in (
                        result.hand_landmarks
                    ):


                        # Draw landmarks

                        draw_landmarks(

                            frame,

                            hand_landmarks
                        )


                        # Save data

                        if collecting:


                            row = landmarks_to_row(

                                hand_landmarks,

                                label
                            )


                            writer.writerow(row)

                            f.flush()


                            samples_collected += 1


                # Status

                if collecting:

                    status = "COLLECTING"

                    status_text = (
                        0,
                        255,
                        0
                    )

                else:

                    status = "PAUSED"

                    status_text = (
                        0,
                        0,
                        255
                    )


                # Label text

                cv2.putText(

                    frame,

                    f"Label: {label}",

                    (10, 30),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.8,

                    (255, 255, 255),

                    2
                )


                # Status text

                cv2.putText(

                    frame,

                    f"Status: {status}",

                    (10, 65),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.8,

                    status_text,

                    2
                )


                # Sample counter

                cv2.putText(

                    frame,

                    f"Samples: {samples_collected}",

                    (10, 100),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.7,

                    (255, 255, 0),

                    2
                )


                # Controls

                cv2.putText(

                    frame,

                    "S=Start  P=Pause  N=New Label  Q=Quit",

                    (10, 135),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.55,

                    (255, 255, 255),

                    2
                )


                # Show webcam

                cv2.imshow(

                    "Sign Language - Data Collection",

                    frame
                )


                # Keyboard

                key = cv2.waitKey(1) & 0xFF


                # Start

                if key == ord("s"):

                    collecting = True

                    print(
                        f"Collecting samples for: {label}"
                    )


                # Pause

                elif key == ord("p"):

                    collecting = False

                    print(
                        "Collection paused."
                    )


                # New label

                elif key == ord("n"):

                    collecting = False

                    cv2.destroyAllWindows()


                    new_label = input(
                        "Enter the NEW label: "
                    ).strip()


                    if new_label:

                        label = new_label

                        samples_collected = 0

                        print(
                            f"New label: {label}"
                        )

                    else:

                        print(
                            "Invalid label."
                        )


                # Quit

                elif key == ord("q"):

                    break


    # Cleanup

    cap.release()

    cv2.destroyAllWindows()


    print("\n--------------------------------")

    print("Data collection finished.")

    print(
        f"Data saved to: {CSV_PATH}"
    )

    print(
        f"Samples collected: {samples_collected}"
    )

    print("--------------------------------")


# ==================================================
# START PROGRAM
# ==================================================

if __name__ == "__main__":

    main()