from Dataset_builder import dataset
from inference import predict

print("Target:")
print(dataset[0]["target"])

print("\nPrediction:")
print(predict(dataset[0]["input"]))