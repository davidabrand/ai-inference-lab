from hf_model import predict_text
from fastapi import FastAPI
from pydantic import BaseModel, Field
from time import perf_counter
import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


app = FastAPI(
    title="AI Inference Lab",
    version="0.2.0"
)

class BatchInferenceRequest(BaseModel):
    texts: list[str]

class InferenceRequest(BaseModel):
    text: str = Field(min_length=1)

class HFInferenceRequest(BaseModel):
    text: str = Field(min_length=1)

@app.post("/hf-infer")
def hf_infer(
    request: HFInferenceRequest
):

    start = perf_counter()


    result = predict_text(
        request.text
    )


    latency_ms = (
        perf_counter() - start
    ) * 1000


    return {
        "input": request.text,
        "prediction": result["label"],
        "confidence": round(
            result["confidence"],
            4
        ),
        "device": result["device"],
        "latency_ms": round(
            latency_ms,
            2
        )
    }


def classify_text(text: str) -> str:

    text = text.lower()

    if "internet" in text or "wifi" in text:
        return "network_issue"

    if "error" in text or "broken" in text:
        return "technical_issue"

    if "hello" in text or "hi" in text:
        return "greeting"

    return "general"


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "service": "ai-inference-lab",
        "version": "0.2.0"
    }


@app.get("/model-info")
def model_info():

    return {
        "name": "rule-based-v1",
        "version": "1.0",
        "hardware": "CPU",
        "type": "simulation"
    }


@app.post("/infer")
def infer(request: InferenceRequest):

    logger.info("Inference request received")

    start = perf_counter()

    prediction = classify_text(request.text)

    latency_ms = (perf_counter() - start) * 1000

    logger.info(
        "Prediction=%s latency_ms=%.4f",
        prediction,
        latency_ms
    )

    return {
        "input": request.text,
        "prediction": prediction,
        "model": "rule-based-v1",
        "latency_ms": round(latency_ms, 4)
    }
@app.post("/batch-infer")
def batch_infer(request: BatchInferenceRequest):

    start = perf_counter()

    results = []

    for text in request.texts:
        prediction = classify_text(text)

        results.append({
            "input": text,
            "prediction": prediction
        })

    latency_ms = (perf_counter() - start) * 1000

    return {
        "count": len(results),
        "results": results,
        "latency_ms": round(latency_ms, 4)
    }

from inference_model import run_model

class TensorInferenceRequest(BaseModel):
    features: list[float]

@app.post("/torch-infer")
def torch_infer(request: TensorInferenceRequest):

    if len(request.features) != 4:
        return {
            "error": "Exactly 4 features are required"
        }

    start = perf_counter()

    result = run_model(request.features)

    latency_ms = (
        perf_counter() - start
    ) * 1000

    return {
        "prediction": result["label"],
        "probabilities": result["probabilities"],
        "device": result["device"],
        "latency_ms": round(latency_ms, 4)
    }
