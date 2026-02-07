import cv2
import os
import numpy as np

def load_images(folder, img_size=64):
    X, y, labels = [], [], {}
    label_id = 0

    for category in os.listdir(folder):
        path = os.path.join(folder, category)

        if category not in labels:
            labels[category] = label_id
            label_id += 1

        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if image is None:
                continue

            image = cv2.resize(image, (img_size, img_size))
            X.append(image.flatten() / 255.0)
            y.append(labels[category])

    return np.array(X), np.array(y), labels
