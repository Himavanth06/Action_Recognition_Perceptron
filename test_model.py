import pickle
import cv2
import numpy as np

with open("action_model.pkl", "rb") as f:
    model, labels = pickle.load(f)

inv_labels = {v: k for k, v in labels.items()}

print("\nHuman Action Recognition System")
print("Type 'exit' to quit.")

while True:
    img_path = input("\nEnter image path: ")

    if img_path.lower() == "exit":
        break

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print("Image not found. Try again.")
        continue

    img = cv2.resize(img, (64, 64))
    img = img.flatten().reshape(1, -1) / 255.0

    pred = model.predict(img)[0]
    print("Predicted Action:", inv_labels[pred])
