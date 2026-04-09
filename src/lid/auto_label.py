import numpy as np
import whisper

def generate_labels(audio_path, frame_duration=0.5):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)

    num_frames = int(result["segments"][-1]["end"] / frame_duration) + 1
    labels = np.zeros(num_frames)

    for segment in result["segments"]:
        start = int(segment["start"] / frame_duration)
        end = int(segment["end"] / frame_duration)

        text = segment["text"]

        # Simple rule: if English letters present → English
        if any(c.isalpha() and c.isascii() for c in text):
            labels[start:end] = 1  # English
        else:
            labels[start:end] = 0  # Hindi

    return labels


if __name__ == "__main__":
    labels = generate_labels("data/processed/lecture_clean.wav")

    print("Total labels:", len(labels))
    print("Sample:", labels[:20])