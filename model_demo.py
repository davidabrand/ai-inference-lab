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


input_tensor = torch.tensor(
    [[1.0, 0.5, 0.2, 0.9]],
    device=device
)


with torch.inference_mode():

    output = model(input_tensor)


print("Device:")
print(device)

print()

print("Input:")
print(input_tensor)

print()

print("Raw output:")
print(output)

prediction = torch.argmax(output, dim=1)

print("Prediction index:")
print(prediction.item())

labels = [
    "greeting",
    "network_issue",
    "technical_issue"
]

label = labels[prediction.item()]

print("Prediction:")
print(label)

probabilities = torch.softmax(output, dim=1)

print(probabilities)
