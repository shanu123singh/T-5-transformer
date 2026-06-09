import pandas as pd
from input_formate import create_input
from target import create_target

print("Dataset Builder Started")

df = pd.read_csv("Crop_recommendation (2).csv")

print("CSV Loaded")
print("Rows:", len(df))

dataset = []

for i, (_, row) in enumerate(df.iterrows()):

    if i % 100 == 0:
        print(f"Processing Row {i}")

    input_text = create_input(
        row,
        "which crop should i grow?"
    )

    target_text = create_target(row)

    dataset.append({
        "input": input_text,
        "target": target_text
    })

print("Dataset Built Successfully")
print("Dataset Size:", len(dataset))