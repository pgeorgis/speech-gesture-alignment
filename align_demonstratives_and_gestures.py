import json
import os
from statistics import mean

from constants import ASR_MODEL_PATH, AUDIO_PATH, TRANCRIPT_PATH, VIDEO_PATH
from extract_gesture import (detect_gesture_apices,
                             detect_hand_gestures_in_video,
                             get_hands_detection_model)
from extract_speech import speech_to_text
from process_video import extract_frames_by_timestamp

# Perform speech-to-text conversion
asr_results = speech_to_text(AUDIO_PATH, model_path=ASR_MODEL_PATH, chunk_duration_seconds=0.1)

# Write transcription to file
with open(TRANCRIPT_PATH, "w") as f:
    json.dump(asr_results, f, indent=4)

# Initialize hands detection model with minimum dection confidence    
hands_detection_model = get_hands_detection_model(min_detection_confidence=0.75)

# Detect hand gestures and find apices of each
gestures = detect_hand_gestures_in_video(VIDEO_PATH, hands_detection_model)
gesture_apices = detect_gesture_apices(gestures)

# Get averaged timestamp of each gesture's apex candidates
apex_timestamps = [mean(gesture_apex.values()) for gesture_apex in gesture_apices.values()]

# Save video frames from apex timestamps
extract_frames_by_timestamp(
    video_path=VIDEO_PATH,
    timestamps=apex_timestamps,
    output_folder=os.path.join("data", "video_frames")
)
