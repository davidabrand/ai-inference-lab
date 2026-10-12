from inference_model import run_model
from time import perf_counter


features = [
    1.0,
    0.5,
    0.2,
    0.9
]

runs = 10000


# -------------------------
# Warm-up
# -------------------------

for _ in range(10):
    run_model(features)


# -------------------------
# Benchmark
# -------------------------

latencies = []

total_start = perf_counter()

for _ in range(runs):

    start = perf_counter()

    run_model(features)

    end = perf_counter()

    latency_ms = (end - start) * 1000

    latencies.append(latency_ms)


total_elapsed = perf_counter() - total_start


# -------------------------
# Statistics
# -------------------------

latencies.sort()


def percentile(data, percent):

    index = int(
        len(data) * percent
    )

    return data[index]


p50 = percentile(latencies, 0.50)

p95 = percentile(latencies, 0.95)

p99 = percentile(latencies, 0.99)

average_latency = sum(latencies) / len(latencies)

minimum_latency = min(latencies)

maximum_latency = max(latencies)

throughput = runs / total_elapsed


print()
print("=== Inference Benchmark ===")

print(f"Runs: {runs}")

print(
    f"Total time: "
    f"{total_elapsed:.4f} seconds"
)

print(
    f"Average latency: "
    f"{average_latency:.4f} ms"
)

print(
    f"Minimum latency: "
    f"{minimum_latency:.4f} ms"
)

print(
    f"Maximum latency: "
    f"{maximum_latency:.4f} ms"
)

print(
    f"Throughput: "
    f"{throughput:.2f} inferences/sec"
)

print(
    f"P50 latency: "
    f"{p50:.4f} ms"
)

print(
    f"P95 latency: "
    f"{p95:.4f} ms"
)

print(
    f"P99 latency: "
    f"{p99:.4f} ms"
)
