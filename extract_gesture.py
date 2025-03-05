from collections import defaultdict
import cv2
import mediapipe as mp
import numpy as np
from process_gesture import find_apex
import json

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


def get_hands_detection_model():
    """Return a new instance of a mediapipe Hands detection model."""
    return mp_hands.Hands()


def get_frame_timestamp_in_seconds(cap):
    """Get capture frame timestamp and convert from seconds to milliseconds."""
    return cap.get(cv2.CAP_PROP_POS_MSEC) / 1000


def convert_frame_to_rgb(frame):
    """Convert a video frame to RGB format."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def detect_hands(rgb_image, model):
    """Detect hands in an image."""
    return model.process(rgb_image)
    

def detect_hand_gestures_in_video(video_path, hands_detection_model, min_frame_appearances=10):
    """Detect video frames containing hands and return index of hand gestures with associated hand landmarks and timing information."""
    # Open video file and load as video capture object
    cap = cv2.VideoCapture(video_path)

    # Iterate over video frames and collect hand gesture landmarks
    gestures = defaultdict(lambda: [])
    gesture_index = 0
    hand_gesture_detected = False
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert the frame to RGB format
        frame_rgb = convert_frame_to_rgb(frame)
        
        # Process the frame to detect hands
        results = detect_hands(frame_rgb, hands_detection_model)
        
        # Check if hands were detected
        if results.multi_hand_landmarks:
            # Increment new gesture
            if not hand_gesture_detected:
                gesture_index += 1
            hand_gesture_detected = True

            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on the frame
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Save the hand landmarks and timestamp
                hand_shape = [np.array([lm.x, lm.y, lm.z]) for lm in hand_landmarks.landmark]  # Extract hand shape as (x, y, z)
                gestures[gesture_index].append({
                    "timestamp": get_frame_timestamp_in_seconds(cap),
                    "hand_shape": hand_shape
                })
        else:
            hand_gesture_detected = False

    # Filter out gestures appearing in minimum N frames (potential false positives)
    gestures = {idx: entry for idx, entry in gestures.items() if len(entry) >= min_frame_appearances}

    return gestures


def detect_gesture_apices(gestures: dict) -> dict:
    """Return a dictionary of gesture indices with their apex timestamps according to 3 criteria:
    - minimum speed (sudden stop or change in direction),
    - maximum acceleration
    - maximum extension from start
    """
    gesture_apices = defaultdict(lambda: {})
    for idx, gesture_data in gestures.items():
        timestamps = np.array([entry['timestamp'] for entry in gesture_data])
        hand_shapes = np.array([entry['hand_shape'] for entry in gesture_data])
        min_speed_time, max_accel_time, max_extension_time = find_apex(hand_shapes, timestamps)
        gesture_apices[idx]["min_speed_timestamp"] = min_speed_time
        gesture_apices[idx]["max_acceleration_timestamp"] = max_accel_time
        gesture_apices[idx]["max_extension_timestamp"] = max_extension_time
    return gesture_apices

hands_detection_model = get_hands_detection_model()
gestures = detect_hand_gestures_in_video("data/video.mp4", hands_detection_model)
gesture_apices = detect_gesture_apices(gestures)