import librosa
import numpy as np

def extract_mfcc_frames(audio_path, frame_size=0.5):
    y, sr = librosa.load(audio_path, sr=16000)

    frame_length = int(frame_size * sr)
    frames = []

    for i in range(0, len(y), frame_length):
        frame = y[i:i+frame_length]

        if len(frame) < frame_length:
            continue

        mfcc = librosa.feature.mfcc(y=frame, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)

        frames.append(mfcc_mean)

    return np.array(frames)


if __name__ == "__main__":
    features = extract_mfcc_frames("data/processed/lecture_clean.wav")
    print("Shape:", features.shape)