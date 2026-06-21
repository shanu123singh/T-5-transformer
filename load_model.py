import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "soil_model_v2")

    print("Checking model path:", model_path)

    # Check if custom model exists
    if os.path.exists(model_path) and os.listdir(model_path):
        print("✅ Loading custom model: soil_model_v2")

        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

    else:
        print("⚠️ soil_model_v2 not found or empty. Loading fallback model (t5-small).")

        model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
        tokenizer = AutoTokenizer.from_pretrained("t5-small")

    return model, tokenizer