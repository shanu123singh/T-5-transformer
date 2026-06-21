from transformers import T5Tokenizer, T5ForConditionalGeneration
import os

MODEL_PATH = "soil_model_v2"

if os.path.exists(MODEL_PATH):
    print("Loading trained model...")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
else:
    print("Loading base T5 model...")
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")