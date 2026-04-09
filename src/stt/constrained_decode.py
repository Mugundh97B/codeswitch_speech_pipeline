import whisper


def load_vocab(vocab_path):
    with open(vocab_path, "r") as f:
        vocab = [line.strip().lower() for line in f.readlines()]
    return vocab


def boost_text(text, vocab):
    words = text.split()
    boosted_words = []

    for word in words:
        clean_word = word.lower()

        # If similar to vocab word → replace
        for v in vocab:
            if v.startswith(clean_word[:4]):  # simple heuristic
                clean_word = v

        boosted_words.append(clean_word)

    return " ".join(boosted_words)


def constrained_transcription(audio_path, vocab_path):
    model = whisper.load_model("base")

    result = model.transcribe(audio_path)

    vocab = load_vocab(vocab_path)

    boosted_segments = []

    for seg in result["segments"]:
        original_text = seg["text"]
        boosted_text = boost_text(original_text, vocab)

        boosted_segments.append({
            "start": seg["start"],
            "end": seg["end"],
            "original": original_text,
            "boosted": boosted_text
        })

    return boosted_segments


if __name__ == "__main__":
    segments = constrained_transcription(
        "data/processed/lecture_clean.wav",
        "src/stt/vocab.txt"
    )

    for s in segments[:5]:
        print("\n---")
        print("Original:", s["original"])
        print("Boosted :", s["boosted"])