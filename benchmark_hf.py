from time import perf_counter

from hf_model import predict_text


text = (
    "Artificial intelligence "
    "inference engineering is interesting."
)


runs = 100


# Warm-up

for _ in range(5):
    predict_text(text)


latencies = []


total_start = perf_counter()


for _ in range(runs):

    start = perf_counter()

    predict_text(text)

    latency_ms = (
        perf_counter() - start
    ) * 1000

    latencies.append(
        latency_ms
    )


total_elapsed = (
    perf_counter() -
    total_start
)


latencies.sort()


def percentile(
    values,
    percentile_value
):

    index = int(
        len(values)
        * percentile_value
    )

    index = min(
        index,
        len(values) - 1
    )

    return values[index]


average = (
    sum(latencies)
    / len(latencies)
)


p50 = percentile(
    latencies,
    0.50
)

p95 = percentile(
    latencies,
    0.95
)

p99 = percentile(
    latencies,
    0.99
)


throughput = (
    runs
    / total_elapsed
)


print()
print(
    "=== Hugging Face Benchmark ==="
)

print(
    f"Runs: {runs}"
)

print(
    f"Average latency: "
    f"{average:.2f} ms"
)

print(
    f"Minimum latency: "
    f"{min(latencies):.2f} ms"
)

print(
    f"Maximum latency: "
    f"{max(latencies):.2f} ms"
)

print(
    f"P50 latency: "
    f"{p50:.2f} ms"
)

print(
    f"P95 latency: "
    f"{p95:.2f} ms"
)

print(
    f"P99 latency: "
    f"{p99:.2f} ms"
)

print(
    f"Throughput: "
    f"{throughput:.2f} "
    "inferences/sec"
)
