import cv2
import mediapipe as mp
import os
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
    

def detect_hands_in_video(video_path, hands_detection_model):
    """Detect video frames containing hands and return list of hand landmarks with timing information."""
    # Open video file and load as video capture object
    cap = cv2.VideoCapture(video_path)

    # Iterate over video frames and collect hand landmarks
    landmarks_data = []
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
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on the frame
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Save the hand landmarks and timestamp
                hand_shape = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]  # Extract hand shape as (x, y, z)
                landmarks_data.append({
                    "timestamp": get_frame_timestamp_in_seconds(cap),
                    "hand_shape": hand_shape
                })
    return landmarks_data
    

hands_detection_model = get_hands_detection_model()
detect_hands_in_video("data/video.mp4", hands_detection_model)