import torch


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

x = torch.tensor([
    1.0,
    2.0,
    3.0
])

print("Originally:")
print(x.device)

x = x.to(device)

print("After moving:")
print(x.device)
