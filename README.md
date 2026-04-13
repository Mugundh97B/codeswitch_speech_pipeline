# Code-Switched Speech Processing Pipeline

##  Overview

This project implements an **end-to-end speech processing pipeline** for **code-switched (Hindi–English) lecture audio**.
The system performs transcription, language identification, phoneme conversion, translation, prosody alignment, and spoof detection.

---

## Features

* **Speech-to-Text (STT)** using Whisper
* **Frame-level Language Identification (LID)** (Hindi vs English)
* **IPA Conversion** for phonetic representation
* **Translation to Tamil** using custom dictionary
* **Prosody Extraction** (Pitch F0 + Energy)
* **Dynamic Time Warping (DTW)** for prosody alignment
* **Speaker Embedding Extraction** (SpeechBrain)
* **Speech Synthesis (Tamil Output)**
* **Spoof Detection** (Real vs Synthetic Speech Classification)

---

##  Pipeline Architecture

```text
Input Audio (Code-Switched Lecture)
        ↓
Speech-to-Text (Whisper)
        ↓
Frame-level LID (MFCC + Neural Network)
        ↓
Constrained Decoding (Custom Vocabulary)
        ↓
IPA Conversion
        ↓
Translation (Tamil)
        ↓
Prosody Extraction (F0 + Energy)
        ↓
DTW Alignment
        ↓
Speaker Embedding (SpeechBrain)
        ↓
Speech Synthesis
        ↓
Spoof Detection (Classifier)
```

---

## Project Structure

```
codeswitch_speech_pipeline/
.
├── README.md
├── data
│   ├── processed
│   │   ├── lecture_10min.wav
│   │   └── lecture_clean.wav
│   └── raw
│       └── lecture.wav
├── notebooks
├── original_segment.wav
├── outputs
│   └── lid_model.pth
├── report
│   └── images
│       
├── requirements.txt
├── speech_understanding.ipynb
├── src
│   ├── lid
│   │   ├── auto_label.py
│   │   ├── create_labels.py
│   │   ├── feature_extraction.py
│   │   ├── model.py
│   │   ├── predict.py
│   │   └── timeline.py
│   ├── spoof
│   ├── stt
│   │   ├── constrained_decode.py
│   │   ├── denoise.py
│   │   ├── download_audio.py
│   │   ├── transcribe.py
│   │   └── vocab.txt
│   ├── translation
│   │   ├── __pycache__
│   │   │   └── tamil_dict.cpython-310.pyc
│   │   ├── ipa_converter.py
│   │   ├── tamil_dict.py
│   │   └── translator.py
│   └── tts
├── student_voice_ref.wav
├── synthetic_tamil.wav
└── transcript.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

##  Usage (Colab Recommended)

1. Clone the repository:

```bash
git clone https://github.com/Mugundh97B/codeswitch_speech_pipeline.git
cd codeswitch_speech_pipeline
```

2. Run in **Google Colab** for full pipeline execution:

* Upload required audio files:

  * `original_segment.wav`
  * `student_voice_ref.wav`

3. Execute steps:

* STT → LID → IPA → Translation → Prosody → DTW → Spoof Detection

---

## Results

* Accurate transcription of code-switched speech
*  Frame-level language switching detection
*  Tamil translation with phonetic awareness
*  Prosody preserved using DTW
*  Successful classification of real vs synthetic speech

---

## Key Techniques Used

* MFCC Feature Extraction
* Neural Network (PyTorch) for LID
* Whisper for STT
* IPA Conversion (eng_to_ipa)
* Dynamic Time Warping (DTW)
* Speaker Embeddings (SpeechBrain)
* Logistic Regression for Spoof Detection

---





