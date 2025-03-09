import logging
import os
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s: %(message)s')

# PATHS TO INPUT DATA AND PROCESSED OUTPUTS
ASR_MODEL_PATH = os.path.join("models", "de", "vosk-model-de-0.21")  # TODO model needs to be added to git or downloaded
AUDIO_PATH = os.path.join("data", "audio.wav")
TRANCRIPT_PATH = os.path.join("data", "transcription.json")
VIDEO_PATH = os.path.join("data", "video.mp4")
DEMONSTRATIVES_SUBTITLES_FILE_PATH = os.path.join("data", "demonstratives.srt")
FULL_SUBTITLES_FILE_PATH = os.path.join("data", "full_subtitles.srt")

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
