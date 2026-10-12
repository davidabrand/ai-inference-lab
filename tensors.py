import torch


vector = torch.tensor([1, 2, 3])

matrix = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
])


print("Vector:")
print(vector)
print("Shape:", vector.shape)

print()

print("Matrix:")
print(matrix)
print("Shape:", matrix.shape)

import torch


a = torch.tensor([
    [1.0, 2.0],
    [3.0, 4.0]
])

b = torch.tensor([
    [5.0, 6.0],
    [7.0, 8.0]
])


print("Addition:")
print(a + b)

print()

print("Element multiplication:")
print(a * b)

print()

print("Matrix multiplication:")
print(a @ b)
