import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s: %(message)s')

# PATHS TO INPUT DATA AND PROCESSED OUTPUTS
ASR_MODEL_PATH = os.path.join("asr_models", "de", "vosk-model-de-0.21")  # TODO model needs to be added to git or downloaded
AUDIO_PATH = os.path.join("data", "input", "audio.wav")
TRANCRIPT_PATH = os.path.join("data", "transcription", "transcription.json")
CORRECTED_TRANCRIPT_PATH = os.path.join("data", "transcription", "corrected_transcription.json")
VIDEO_PATH = os.path.join("data", "input", "video.mp4")
VIDEO_FRAMES_OUTDIR = os.path.join("data", "video_frames")
DEMONSTRATIVES_SUBTITLES_FILE_PATH = os.path.join("data", "subtitles", "demonstratives.srt")
FULL_SUBTITLES_FILE_PATH = os.path.join("data", "subtitles", "full_subtitles.srt")
GESTURES_JSON = os.path.join("data", "gestures", "gestures.json")
ALIGNED_GESTURES_TSV = os.path.join("data", "gestures", "aligned_gestures.tsv")
GESTURE_ALIGNMENT_DENSITY_PLOT = os.path.join("data", "gestures", "gesture_alignment_density_plot.png")

# KEYS IN ASR RESULTS
ASR_TIMED_RESULTS_KEY = "result"
ASR_TEXT_KEY = "text"
TOKEN_KEY = "word"
TOKEN_ONSET_KEY = "start"
TOKEN_END_KEY = "end"
POS_TAG_KEY = "pos_tag"

# GESTURE KEYS
NEAREST_GESTURE_KEY = "nearest_gesture"
NEAREST_GESTURE_APEX_KEY = "nearest_gesture_apex"
NEAREST_GESTURE_OFFSET_KEY = "nearest_gesture_offset"

# DEMONSTRATIVE PRONOUNS TO FILTER IN TRANSCRIPTION
DEMONSTRATIVE_POS = {
    "PDAT",  # demonstrative determiner
    "PDS",   # demonstrative pronoun
    "ART",   # article (common mislabel for demonstratives like <das>, <dem>, etc.)
}
