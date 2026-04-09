import numpy as np
import librosa
import soundfile as sf

def spectral_subtraction(input_path, output_path):
    # Load audio
    y, sr = librosa.load(input_path, sr=None)

    # Estimate noise from first second
    noise_sample = y[:sr]

    # FFT
    Y = np.fft.rfft(y)
    noise_fft = np.fft.rfft(noise_sample)

    # Magnitude
    Y_mag = np.abs(Y)
    noise_mag = np.abs(noise_fft)

    # remove noise
    clean_mag = Y_mag - np.mean(noise_mag)
    clean_mag = np.maximum(clean_mag, 0)

    # Rsignal reconstruction
    phase = np.angle(Y)
    clean_fft = clean_mag * np.exp(1j * phase)
    clean_signal = np.fft.irfft(clean_fft)

    # Normalize
    clean_signal = clean_signal / np.max(np.abs(clean_signal))

    # Save
    sf.write(output_path, clean_signal, sr)


if __name__ == "__main__":
    spectral_subtraction(
        "data/processed/lecture_10min.wav",
        "data/processed/lecture_clean.wav"
    )