import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


MODEL_NAME = (
    "distilbert-base-uncased-finetuned-sst-2-english"
)


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print(f"Loading model on: {device}")


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME
)


model = model.to(device)

model.eval()


def predict_text(text: str):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True
    )


    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }


    with torch.inference_mode():

        outputs = model(**inputs)


    logits = outputs.logits


    probabilities = torch.softmax(
        logits,
        dim=-1
    )


    prediction_index = torch.argmax(
        probabilities,
        dim=-1
    ).item()


    label = model.config.id2label[
        prediction_index
    ]


    confidence = probabilities[
        0,
        prediction_index
    ].item()


    return {
        "label": label,
        "confidence": confidence,
        "device": str(device)
        }
