import json
import logging
import wave

from pydub import AudioSegment
from vosk import KaldiRecognizer, Model

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def convert_to_mono(wav_file: str, output_file: str):
    """Convert stereo audio to mono using pydub."""
    audio = AudioSegment.from_wav(wav_file)
    mono_audio = audio.set_channels(1)
    mono_audio.export(output_file, format="wav")


def speech_to_text(wav_file: str, model_path: str = "model", chunk_duration_seconds: float = 0.25):
    """Extract speech with word timings using Vosk speech-to-text model."""
    # Check sampling rate and whether audio is mono/stereo
    with wave.open(wav_file, "rb") as wf:
        if wf.getnchannels() != 1:
            # If not mono, convert to mono audio in new wav file, then extract from that file
            logger.info("Converting audio to mono")
            mono_wave_file = wav_file.replace(".wav", "_mono.wav")
            convert_to_mono(wav_file, output_file=mono_wave_file)
            return speech_to_text(wav_file=mono_wave_file, model_path=model_path)
        sample_rate = wf.getframerate()

    frames_per_chunk = int(sample_rate * chunk_duration_seconds)
    model = Model(model_path)
    recognizer = KaldiRecognizer(model, sample_rate)
    recognizer.SetWords(True)
    with wave.open(wav_file, "rb") as wf:
        while True:
            data = wf.readframes(frames_per_chunk)
            if len(data) == 0:
                break
            recognizer.AcceptWaveform(data)

    result = json.loads(recognizer.FinalResult())
    return result
