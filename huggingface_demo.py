from transformers import pipeline


classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)


text = "I really enjoyed learning about PyTorch today."

result = classifier(text)

print(result)
