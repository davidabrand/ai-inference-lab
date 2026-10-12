from time import perf_counter
import torch

from hf_model import tokenizer, model, device

text = (
    "Artificial intelligence inference engineering "
    "is interesting because GPUs can efficiently "
    "process large tensor operations."
)

runs = 100


def synchronize():
    if device.type == "cuda":
        torch.cuda.synchronize()


# -----------------------------
# Full pipeline warm-up
# -----------------------------

for _ in range(20):

    inputs = tokenizer(
        text,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )

        prediction = torch.argmax(
            probabilities,
            dim=-1
        )

synchronize()


tokenization_times = []
device_times = []
model_times = []
output_times = []


# -----------------------------
# Benchmark
# -----------------------------
encoded = tokenizer(text)
token_count = len(encoded["input_ids"])

print(f"Input tokens: {token_count}")

for _ in range(runs):

    # Tokenization
    synchronize()

    start = perf_counter()

    inputs = tokenizer(
        text,
        return_tensors="pt"
    )

    tokenization_times.append(
        (perf_counter() - start) * 1000
    )


    # CPU -> GPU
    synchronize()

    start = perf_counter()

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    synchronize()

    device_times.append(
        (perf_counter() - start) * 1000
    )


    # Model inference
    synchronize()

    start = perf_counter()

    with torch.inference_mode():
        outputs = model(**inputs)

    synchronize()

    model_times.append(
        (perf_counter() - start) * 1000
    )


    # Output processing
    synchronize()

    start = perf_counter()

    probabilities = torch.softmax(
        outputs.logits,
        dim=-1
    )

    prediction = torch.argmax(
        probabilities,
        dim=-1
    )

    synchronize()

    output_times.append(
        (perf_counter() - start) * 1000
    )

def average(values):
    return sum(values) / len(values)


print()
print("=== Average Component Latency ===")
print(f"Device: {device}")
print(f"Runs: {runs}")
print()

print(
    f"Tokenization:      "
    f"{average(tokenization_times):.3f} ms"
)

print(
    f"Device movement:   "
    f"{average(device_times):.3f} ms"
)

print(
    f"Model inference:   "
    f"{average(model_times):.3f} ms"
)

print(
    f"Output processing: "
    f"{average(output_times):.3f} ms"
)
