from predict import predict

def generate_timeline(frame_duration=0.5):
    preds = predict()

    timeline = []

    for i, label in enumerate(preds):
        start_time = i * frame_duration
        end_time = start_time + frame_duration

        language = "Hindi" if label == 0 else "English"

        timeline.append((start_time, end_time, language))

    return timeline


if __name__ == "__main__":
    timeline = generate_timeline()

    print("First 10 timeline entries:\n")
    for entry in timeline[:10]:
        print(entry)