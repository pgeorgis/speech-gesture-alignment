import json
import os
from statistics import mean
from typing import Callable

from constants import (ASR_MODEL_PATH, ASR_TIMED_RESULTS_KEY, AUDIO_PATH,
                       DEMONSTRATIVE_PRONOUNS, TOKEN_KEY, TOKEN_ONSET_KEY,
                       TRANCRIPT_PATH, VIDEO_PATH, DEMONSTRATIVES_SUBTITLES_FILE_PATH, FULL_SUBTITLES_FILE_PATH)
from extract_gesture import (detect_gesture_apices,
                             detect_hand_gestures_in_video,
                             get_hands_detection_model)
from extract_speech import speech_to_text
from process_video import extract_frames_by_timestamp
from subtitles import json_to_srt, write_srt, add_subtitles_to_video


def get_word_timings_from_asr_results(asr_results: dict, filter_func: Callable) -> list:
    """Get word timings for selected words in ASR results."""
    filtered_word_timings = []
    for entry in asr_results[ASR_TIMED_RESULTS_KEY]:
        if filter_func(entry.get(TOKEN_KEY, "")):
            filtered_word_timings.append(entry)
    return filtered_word_timings


def find_nearest_gesture_to_words(word_timings: list, gesture_apices: dict, key=TOKEN_ONSET_KEY) -> dict:
    """Find nearest gesture (by apex timestamp) to each word's onset."""
    nearest_gestures = {}
    for entry in word_timings:
        word_time = entry[key]
        word = entry[TOKEN_KEY]
        nearest_gesture = min(gesture_apices.keys(), key=lambda x: abs(word_time - gesture_apices[x]))
        gesture_apex = gesture_apices[nearest_gesture]
        nearest_gestures[(word, word_time)] = (nearest_gesture, gesture_apex)
    return nearest_gestures


def is_demonstrative(word):
    """Check if a word matches a demonstrative pronoun regex."""
    if any(demonstr_regex.match(word) for demonstr_regex in DEMONSTRATIVE_PRONOUNS):
        return True
    return False


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
demonstrative_timings = get_word_timings_from_asr_results(
    asr_results,
    filter_func=is_demonstrative,
)

# Find nearest gesture to each pronoun's onset
nearest_gestures_to_demonstratives = find_nearest_gesture_to_words(
    demonstrative_timings,
    apex_timestamps,
    key=TOKEN_ONSET_KEY,
)

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

# Create subtitle file with demonstratives only
# Add subtitles to video for demonstratives only
demonstratives_str = json_to_srt(demonstrative_timings)
write_srt(demonstratives_str, DEMONSTRATIVES_SUBTITLES_FILE_PATH)
add_subtitles_to_video(VIDEO_PATH, DEMONSTRATIVES_SUBTITLES_FILE_PATH, subtitle_language="de", soft_subtitle=True)
# Create video copy with full subtitles also (from corrected transcription)
with open(os.path.join("data", "corrected_transcription.json"), "r") as f:
    corrected_asr_results = json.load(f)
full_subtitles = json_to_srt(corrected_asr_results[ASR_TIMED_RESULTS_KEY])
write_srt(full_subtitles, FULL_SUBTITLES_FILE_PATH)
add_subtitles_to_video(VIDEO_PATH, FULL_SUBTITLES_FILE_PATH, subtitle_language="de", soft_subtitle=True)