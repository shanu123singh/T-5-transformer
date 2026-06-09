from transformers import T5Tokenizer, T5ForConditionalGeneration

print("Loading tokenizer...")
tokenizer = T5Tokenizer.from_pretrained("t5-small")
print("Tokenizer loaded")

print("Loading model...")
model = T5ForConditionalGeneration.from_pretrained("t5-small")
print("Model loaded")