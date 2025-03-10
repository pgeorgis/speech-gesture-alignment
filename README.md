# Speech-Gesture Alignment

## Description
This project performs automatic speech recognition and hand gesture detection, and searches for nearest gesture apices within bounds of words of interest.

## Installation
### Prerequisites
- Python 3
- Set up virtual environment (see below)
- vosk ASR model must be downloaded from https://alphacephei.com/vosk/models into `asr_models` directory
e.g. `asr_models/de/vosk-model-de-0.21`

### Setup Instructions
Create virtual environment using `setup.sh`:
```
./setup.sh
```

Ensure the virtual environment is activated:
```
source .venv/bin/activate
```

## Usage
Run the alignment script with the following command:
```
python3 align_demonstratives_and_gestures.py
```
