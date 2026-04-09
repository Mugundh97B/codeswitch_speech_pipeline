import whisper

def transcribe_audio(audio_path):
    model = whisper.load_model("base")  # small model

    result = model.transcribe(audio_path)

    return result


if __name__ == "__main__":
    result = transcribe_audio("data/processed/lecture_clean.wav")

    for segment in result["segments"][:5]:
        print(segment)