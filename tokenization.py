from load_model import tokenizer

def encode(input_text, target_text):

    input_encoding = tokenizer(
        input_text,
        truncation=True,
        padding="max_length",
        max_length=64,
        return_tensors="pt"
    )

    target_encoding = tokenizer(
        target_text,
        truncation=True,
        padding="max_length",
        max_length=32,
        return_tensors="pt"
    )

    return input_encoding, target_encoding