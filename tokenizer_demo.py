from transformers import AutoTokenizer


MODEL_NAME = (
    "distilbert-base-uncased-finetuned-sst-2-english"
)


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


text = "AI inference engineering is interesting."


tokens = tokenizer.tokenize(text)

token_ids = tokenizer.encode(text)


print("Original text:")
print(text)

print()

print("Tokens:")
print(tokens)

print()

print("Token IDs:")
print(token_ids)

print()

print("Number of tokens:")
print(len(token_ids))

inputs = tokenizer(
    text,
    return_tensors="pt"
)


print(inputs)
