from Dataset_builder import dataset
from load_model import model, tokenizer
from tokenization import encode

import torch
from torch.optim import AdamW
import os


torch.set_num_threads(8)


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using Device:", device)

model.to(device)


dataset = dataset[:100]


print("Dataset Size:", len(dataset))


optimizer = AdamW(
    model.parameters(),
    lr=5e-5
)


model.train()

epochs = 3

for epoch in range(epochs):

    total_loss = 0

    for idx, data in enumerate(dataset):

        input_text = data["input"]
        target_text = data["target"]

        inputs, targets = encode(
            input_text,
            target_text
        )

        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        labels = targets["input_ids"].to(device)

        labels = labels.clone()
        labels[labels == tokenizer.pad_token_id] = -100

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        if idx % 10 == 0:
            print(
                f"Sample {idx}/{len(dataset)} "
                f"Loss: {loss.item():.4f}"
            )

    avg_loss = total_loss / len(dataset)

    print(
        f"\nEpoch {epoch+1} Completed"
    )

    print(
        f"Average Loss: {avg_loss:.4f}"
    )


import os
import shutil

# Save model
save_path = "soil_model_v2"

# Agar folder pehle se exist karta hai to delete kar do
if os.path.exists(save_path):
    shutil.rmtree(save_path)

os.makedirs(save_path, exist_ok=True)

# Save model (.bin format me)
model.save_pretrained(
    save_path,
    safe_serialization=False
)

tokenizer.save_pretrained(save_path)

print(f"Model Saved Successfully at {save_path}")