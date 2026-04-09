import numpy as np

def create_dummy_labels(num_frames):
    """
    First half → Hindi (0)
    Second half → English (1)
    """
    labels = np.zeros(num_frames)

    split = num_frames // 2
    labels[split:] = 1

    return labels


if __name__ == "__main__":
    num_frames = 1200  # from previous step
    labels = create_dummy_labels(num_frames)

    print("Labels shape:", labels.shape)
    print("Sample labels:", labels[:20])