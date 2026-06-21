from load_model import model, tokenizer

def predict(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True
    )

    output = model.generate(
        inputs["input_ids"],
        max_length=80,
        num_beams=5
    )

    return tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )