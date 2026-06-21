def predict(model, tokenizer, text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=10,
        num_beams=7,
        repetition_penalty=2.0,
        early_stopping=True
    )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    result = result.strip().lower()

    # cleanup garbage output
    if "not_duplicate" in result or result == "":
        return "unable to predict (model issue)"

    return result