import json
import os
from statistics import mean
from typing import Callable

from constants import (ASR_MODEL_PATH, AUDIO_PATH, DEMONSTRATIVE_PRONOUNS,
                       TRANCRIPT_PATH, VIDEO_PATH)
from extract_gesture import (detect_gesture_apices,
                             detect_hand_gestures_in_video,
                             get_hands_detection_model)
from extract_speech import speech_to_text
from process_video import extract_frames_by_timestamp

def get_word_onsets_from_asr_results(asr_results: dict, filter_func: Callable) -> list:
    """Get word onset timings for selected words in ASR results."""
    filtered_word_timings = []
    for entry in asr_results["result"]:
        if filter_func(entry.get("word", "")):
            filtered_word_timings.append((entry.get("word"), entry.get("start")))
    return filtered_word_timings


def find_nearest_gesture_to_words(word_onsets: list, gesture_apices: dict) -> dict:
    """Find nearest gesture (by apex timestamp) to each word's onset."""
    nearest_gestures = {}
    for word, onset in word_onsets:
        nearest_gesture = min(gesture_apices.keys(), key=lambda x: abs(onset - gesture_apices[x]))
        gesture_apex = gesture_apices[nearest_gesture]
        nearest_gestures[(word, onset)] = (nearest_gesture, gesture_apex)
    return nearest_gestures

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
#apex_timestamps = {idx: mean(gesture_apex.values()) for idx, gesture_apex in gesture_apices.items()}
# Get timestamp of each gesture's apex candidates
apex_timestamps = {
    (idx, apex_type): apex_timestamp
    for idx, gesture in gesture_apices.items()
    for (apex_type, apex_timestamp) in gesture.items()
}

# Get timestamps of demonstrative pronoun onsets from ASR results
demonstrative_onsets = get_word_onsets_from_asr_results(
    asr_results,
    filter_func=lambda x: any(demonstr_regex.match(x) for demonstr_regex in DEMONSTRATIVE_PRONOUNS)
)

# Find nearest gesture to each pronoun's onset
nearest_gestures_to_demonstratives = find_nearest_gesture_to_words(demonstrative_onsets, apex_timestamps)

# Save video frames from nearest gesture apex timestamps
nearest_gesture_timestamps = [
    apex_timestamp
    for _, apex_timestamp in nearest_gestures_to_demonstratives.values()
]
extract_frames_by_timestamp(
    video_path=VIDEO_PATH,
    timestamps=nearest_gesture_timestamps,
    output_folder=os.path.join("data", "video_frames")
)
