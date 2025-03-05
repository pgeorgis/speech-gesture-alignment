import speech_recognition as sr

def speech_to_text(wav_file: str, language: str = 'en-us'):
    """Convert speech from a .wav file to text."""
    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_file) as source:
        audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data, language=language)
    return text
