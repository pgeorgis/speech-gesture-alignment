import json
import logging
import os
from collections import defaultdict
from statistics import mean
from typing import Callable

from constants import (ASR_MODEL_PATH, ASR_TIMED_RESULTS_KEY, AUDIO_PATH,
                       DEMONSTRATIVE_PRONOUNS,
                       DEMONSTRATIVES_SUBTITLES_FILE_PATH,
                       FULL_SUBTITLES_FILE_PATH, GESTURES_JSON, TOKEN_KEY,
                       TOKEN_ONSET_KEY, TRANCRIPT_PATH, VIDEO_PATH)
from extract_gesture import (GestureDetector, combine_overlapping_gestures,
                             detect_gesture_apices)
from extract_speech import speech_to_text
from process_video import extract_frames_by_timestamp
from subtitles import add_subtitles_to_video, json_to_srt, write_srt

logger = logging.getLogger(__name__)

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


# Get timestamps of demonstrative pronoun onsets from ASR results
demonstrative_timings = get_word_timings_from_asr_results(
    asr_results,
    filter_func=is_demonstrative,
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


# Detect hand gestures within range of demonstratives and find apices of each
max_seconds_bounds = 0.5
gesture_detector = GestureDetector(VIDEO_PATH)
gesture_events = gesture_detector.search_for_gestures_near_specific_words(
    word_timings=demonstrative_timings,
    max_window=0.5,
    combine_overlapping=True,
)
logger.info(f"Found {len(gesture_events)} gesture events within bounds of demonstratives")
gesture_apices = detect_gesture_apices(gesture_events)

# Get averaged timestamp of each gesture's apex candidates
apex_timestamps = {idx: mean(gesture_apex.values()) for idx, gesture_apex in gesture_apices.items()}

# # Save video frames from nearest gesture apex timestamps
extract_frames_by_timestamp(
    video_path=VIDEO_PATH,
    timestamps=apex_timestamps.values(),
    output_folder=os.path.join("data", "video_frames")
)

