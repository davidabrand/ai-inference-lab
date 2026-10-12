import requests
from time import perf_counter


URL = "http://127.0.0.1:8000/infer"


requests_to_test = [
    "Hello",
    "My wifi stopped working",
    "The computer is broken",
    "Tell me something interesting"
] * 20


for text in requests_to_test:

    start = perf_counter()

    response = requests.post(
        URL,
        json={"text": text}
    )

    total_latency_ms = (perf_counter() - start) * 1000

    data = response.json()

    print("-" * 40)

    print(f"Input: {text}")
    print(f"Prediction: {data['prediction']}")

    print(
        f"Server processing: "
        f"{data['latency_ms']} ms"
    )

    print(
        f"Total request latency: "
        f"{total_latency_ms:.2f} ms"
    )
