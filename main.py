from fastapi import FastAPI
from pydantic import BaseModel, Field
from time import perf_counter

app = FastAPI(title="AI Inference Lab")


class InferenceRequest(BaseModel):
    text: str = Field(min_length=1)


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "ai-inference-lab"
    }

@app.get("/model-info")
def modelinfo():
    return{
        "name": "rule-based-v1",
        "version": "1.0",
        "hardware": "CPU",
        "type": "simulation"
    }
@app.post("/infer")
def infer(request: InferenceRequest):
    start = perf_counter()

    # Temporary rule-based inference simulation
    text = request.text.lower()

    if "error" in text or "broken" in text:
        prediction = "technical_issue"
    elif "hello" in text or "hi" in text:
        prediction = "greeting"
    elif "internet" in text:
        prediction = "network_issue"
    else:
        prediction = "general"

    latency_ms = (perf_counter() - start) * 1000

    return {
        "input": request.text,
        "prediction": prediction,
        "model": "rule-based-v1",
        "latency_ms": round(latency_ms, 4)
    }