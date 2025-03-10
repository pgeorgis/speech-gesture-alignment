import logging
import os
import re

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

# KEYS IN ASR RESULTS
ASR_TIMED_RESULTS_KEY = "result"
TOKEN_KEY = "word"
TOKEN_ONSET_KEY = "start"
TOKEN_END_KEY = "end"

# DEMONSTRATIVE PRONOUNS TO FILTER IN TRANSCRIPTION
DEMONSTRATIVE_PRONOUNS = {
    r"das$",
    r"dem$",
    r"da(hinte[nr]|rüber)?$",
    r"dies(e([mnrs])?)?$",
}
DEMONSTRATIVE_PRONOUNS = {re.compile(pattern, re.IGNORECASE) for pattern in DEMONSTRATIVE_PRONOUNS}
