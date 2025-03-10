import json
import logging
from typing import Callable

import pandas as pd

from constants import (ALIGNED_GESTURES_TSV, ASR_MODEL_PATH,
                       ASR_TIMED_RESULTS_KEY, AUDIO_PATH,
                       CORRECTED_TRANCRIPT_PATH, DEMONSTRATIVE_POS,
                       DEMONSTRATIVES_SUBTITLES_FILE_PATH,
                       FULL_SUBTITLES_FILE_PATH,
                       GESTURE_ALIGNMENT_DENSITY_PLOT, GESTURES_JSON,
                       POS_TAG_KEY, TOKEN_ONSET_KEY, TRANCRIPT_PATH,
                       VIDEO_FRAMES_OUTDIR, VIDEO_PATH)
from extract_gesture import GestureDetector
from extract_speech import speech_to_text
from gesture_apex import detect_gesture_apices
from plot_gesture_alignment import create_gesture_word_alignment_density_plot
from pos_tagging import pos_tag_asr_results
from process_video import extract_frames_by_timestamp
from subtitles import add_subtitles_to_video, json_to_srt, write_srt

logger = logging.getLogger(__name__)

def get_word_timings_from_asr_results(asr_results: dict, filter_func: Callable) -> list:
    """Get word timings for selected words in ASR results."""
    filtered_word_timings = []
    for entry in asr_results[ASR_TIMED_RESULTS_KEY]:
        if filter_func(entry):
            filtered_word_timings.append(entry)
    return filtered_word_timings


def find_nearest_gesture_to_words(word_timings: list,
                                  gesture_apices: dict,
                                  key=TOKEN_ONSET_KEY,
                                  max_offset_from_word: float | None = None,
                                  ) -> list:
    """Find nearest gesture (by apex timestamp) to each word's onset."""
    nearest_gestures = []
    for entry in word_timings:
        entry_copy = entry.copy()
        word_time = entry_copy[key]
        nearest_gesture = min(gesture_apices.keys(), key=lambda x: abs(word_time - gesture_apices[x]))
        nearest_gesture_apex = gesture_apices[nearest_gesture]
        nearest_gesture_offset = word_time - nearest_gesture_apex
        if max_offset_from_word and abs(nearest_gesture_offset) > max_offset_from_word:
            entry_copy["nearest_gesture"] = None
            entry_copy["nearest_gesture_apex"] = None
            entry_copy["nearest_gesture_offset"] = None
        else:
            entry_copy["nearest_gesture"] = nearest_gesture
            entry_copy["nearest_gesture_apex"] = nearest_gesture_apex
            entry_copy["nearest_gesture_offset"] = nearest_gesture_offset
        nearest_gestures.append(entry_copy)
    
    return nearest_gestures


def is_demonstrative(word_entry):
    """Check if a word is a demonstrative by POS tag."""
    pos_tag = word_entry.get(POS_TAG_KEY, "")
    if pos_tag in DEMONSTRATIVE_POS:
        return True
    return False


# Perform speech-to-text conversion
asr_results = speech_to_text(AUDIO_PATH, model_path=ASR_MODEL_PATH, chunk_duration_seconds=0.1)
# Perform POS tagging on ASR results
asr_results = pos_tag_asr_results(asr_results, language="de")
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
with open(CORRECTED_TRANCRIPT_PATH, "r") as f:
    corrected_asr_results = json.load(f)
full_subtitles = json_to_srt(corrected_asr_results[ASR_TIMED_RESULTS_KEY])
write_srt(full_subtitles, FULL_SUBTITLES_FILE_PATH)
add_subtitles_to_video(VIDEO_PATH, FULL_SUBTITLES_FILE_PATH, subtitle_language="de", soft_subtitle=True)


# Detect hand gestures within range of demonstratives and find apices of each
max_seconds_bounds = 0.5
gesture_detector = GestureDetector(VIDEO_PATH)
gesture_detector.dump_gesture_events(GESTURES_JSON)
logger.info(f"Wrote full gesture json to {GESTURES_JSON}")
gesture_events = gesture_detector.search_for_gestures_near_specific_words(
    word_timings=demonstrative_timings,
    max_window=max_seconds_bounds,
    combine_overlapping=True,
)
logger.info(f"Found {len(gesture_events)} gesture events within bounds of demonstratives")
# Get averaged timestamp of each gesture's apex candidates
gesture_apices = detect_gesture_apices(gesture_events, average=True)

# Save video frames from nearest gesture apex timestamps
extract_frames_by_timestamp(
    video_path=VIDEO_PATH,
    timestamps=gesture_apices.values(),
    output_folder=VIDEO_FRAMES_OUTDIR
)

# Find nearest gesture apex to each word of interest
nearest_gestures_to_demonstratives = find_nearest_gesture_to_words(
    demonstrative_timings,
    gesture_apices,
    key=TOKEN_ONSET_KEY,
    max_offset_from_word=0.75,
)
# Assemble aligned demonstratives and gestures into dataframe
aligned_word_gesture_df = pd.DataFrame(nearest_gestures_to_demonstratives)
# Write to TSV file
aligned_word_gesture_df.to_csv(ALIGNED_GESTURES_TSV, index=False, sep="\t")
create_gesture_word_alignment_density_plot(
    aligned_word_gesture_df,
    offset_key="nearest_gesture_offset",
    title="Alignment of gesture apices to demonstratives",
    outfile=GESTURE_ALIGNMENT_DENSITY_PLOT,
)
