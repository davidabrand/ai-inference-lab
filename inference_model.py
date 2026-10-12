import torch
import torch.nn as nn


class TinyClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 3)
        )

    def forward(self, x):

        return self.network(x)


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


model = TinyClassifier().to(device)

model.eval()


labels = [
    "greeting",
    "network_issue",
    "technical_issue"
]


def run_model(features: list[float]):

    tensor = torch.tensor(
        [features],
        dtype=torch.float32,
        device=device
    )

    with torch.inference_mode():

        output = model(tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )

        prediction_index = torch.argmax(
            probabilities,
            dim=1
        ).item()

    return {
        "label": labels[prediction_index],
        "probabilities": probabilities
            .cpu()
            .tolist()[0],
        "device": str(device)
    }
