import numpy as np

class Perceptron:
    def __init__(self, lr=0.01, epochs=40):
        self.lr = lr
        self.epochs = epochs

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.W = np.zeros((len(self.classes), X.shape[1]))
        self.b = np.zeros(len(self.classes))

        for epoch in range(self.epochs):
            errors = 0
            for xi, yi in zip(X, y):
                scores = np.dot(self.W, xi) + self.b
                pred = np.argmax(scores)

                if pred != yi:
                    self.W[yi] += self.lr * xi
                    self.W[pred] -= self.lr * xi
                    self.b[yi] += self.lr
                    self.b[pred] -= self.lr
                    errors += 1

            print(f"Epoch {epoch+1}/{self.epochs} | Errors: {errors}")

    def predict(self, X):
        scores = np.dot(X, self.W.T) + self.b
        return np.argmax(scores, axis=1)
