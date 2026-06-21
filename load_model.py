import os
from transformers import T5Tokenizer, T5ForConditionalGeneration

def load_model():

    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "soil_model_v2")

    print("MODEL PATH:", model_path)

    # check model exists
    if os.path.exists(model_path) and len(os.listdir(model_path)) > 0:

        print("✅ Loading trained model")

        tokenizer = T5Tokenizer.from_pretrained(model_path)
        model = T5ForConditionalGeneration.from_pretrained(model_path)

    else:

        print("⚠️ Loading fallback model (t5-small)")

        tokenizer = T5Tokenizer.from_pretrained("t5-small")
        model = T5ForConditionalGeneration.from_pretrained("t5-small")

    return tokenizer, model