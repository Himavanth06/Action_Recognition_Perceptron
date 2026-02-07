from utils import load_images
from model import Perceptron
import pickle

print("[INFO] Loading training images...")
X_train, y_train, labels = load_images("dataset/train")

print("[INFO] Training perceptron model...")
model = Perceptron(lr=0.01, epochs=40)
model.fit(X_train, y_train)

print("[INFO] Saving trained model...")
with open("action_model.pkl", "wb") as f:
    pickle.dump((model, labels), f)

print("Model trained and saved successfully.")
