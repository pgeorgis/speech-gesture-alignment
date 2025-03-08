import logging
import os
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s: %(message)s')

DEMONSTRATIVE_PRONOUNS = {
    r"das$",
    #r"da(hinte[nr]|rüber)$",
    r"dies(e([mnrs])?)?$",
}
DEMONSTRATIVE_PRONOUNS = {re.compile(pattern, re.IGNORECASE) for pattern in DEMONSTRATIVE_PRONOUNS}

ASR_MODEL_PATH = os.path.join("models", "de", "vosk-model-de-0.21")  # TODO model needs to be added to git or downloaded
AUDIO_PATH = os.path.join("data", "audio.wav")
TRANCRIPT_PATH = os.path.join("data", "transcription.json")
VIDEO_PATH = os.path.join("data", "video.mp4")
