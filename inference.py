def predict(model, tokenizer, text):
    inputs = tokenizer.encode(text, return_tensors="pt")

    outputs = model.generate(
        inputs,
        max_length=50,
        num_beams=4,
        early_stopping=True
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)