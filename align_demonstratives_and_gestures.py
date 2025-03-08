import json
from statistics import mean

from extract_gesture import (detect_gesture_apices,
                             detect_hand_gestures_in_video,
                             get_hands_detection_model)
from extract_speech import speech_to_text

# Perform speech-to-text conversion
asr_results = speech_to_text("data/audio.wav", model_path="models/de/vosk-model-de-0.21", chunk_duration_seconds=0.1)

# Write transcription to file
with open("data/transcription.json", "w") as f:
    json.dump(asr_results, f, indent=4)

# Initialize hands detection model with minimum dection confidence    
hands_detection_model = get_hands_detection_model(min_detection_confidence=0.75)

# Detect hand gestures and find apices of each
gestures = detect_hand_gestures_in_video("data/video.mp4", hands_detection_model)
gesture_apices = detect_gesture_apices(gestures)

# Get averaged timestamp of each gesture's apex candidates
apex_timestamps = [mean(gesture_apex.values()) for gesture_apex in gesture_apices.values()]