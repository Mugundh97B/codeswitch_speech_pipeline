import torch
from model import LIDModel
from feature_extraction import extract_mfcc_frames

def predict():
    # Load features
    X = extract_mfcc_frames("data/processed/lecture_clean.wav")
    X = torch.tensor(X, dtype=torch.float32)

    # Load model
    model = LIDModel()
    model.load_state_dict(torch.load("outputs/lid_model.pth", map_location=torch.device('cpu')))
    model.eval()

    # Forward pass
    with torch.no_grad():
        outputs = model(X)
        predictions = torch.argmax(outputs, dim=1)

    return predictions.numpy()


if __name__ == "__main__":
    preds = predict()

    print("First 20 predictions:")
    print(preds[:20])