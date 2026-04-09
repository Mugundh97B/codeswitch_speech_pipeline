import torch
import torch.nn as nn
from feature_extraction import extract_mfcc_frames
from auto_label import generate_labels


class LIDModel(nn.Module):
    def __init__(self):
        super(LIDModel, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(13, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )

    def forward(self, x):
        return self.model(x)


def train():
    # Load features
    X = extract_mfcc_frames("data/processed/lecture_clean.wav")

    # Generate labels (semi-automatic)
    y = generate_labels("data/processed/lecture_clean.wav")

    # 🔥 IMPORTANT: match lengths
    min_len = min(len(X), len(y))
    X = X[:min_len]
    y = y[:min_len]

    # Convert to tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    # Model
    model = LIDModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(10):
        outputs = model(X)
        loss = criterion(outputs, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    return model


if __name__ == "__main__":
    model = train()

    # Save model
    torch.save(model.state_dict(), "outputs/lid_model.pth")
    print("Model saved!")