import numpy as np
#import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tkinter as tk
from tkinter import Canvas, messagebox
from PIL import Image, ImageDraw

# Load the MNIST dataset
def load_data():
    mnist = fetch_openml('mnist_784', version=1)
    X = mnist.data
    y = mnist.target.astype(np.int8)
    return X, y

# Preprocess the data
def preprocess_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, scaler

# Train the model
def train_model(X_train, y_train):
    model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=20, alpha=1e-4,
                          solver='adam', verbose=10, random_state=42)
    model.fit(X_train, y_train)
    return model

# GUI to draw and predict digits
class DigitPredictor(tk.Tk):
    def __init__(self, model, scaler):
        super().__init__()
        self.title("Digit Predictor")
        self.model = model
        self.scaler = scaler

        self.canvas = Canvas(self, width=200, height=200, bg='white')
        self.canvas.pack()

        self.button_predict = tk.Button(self, text='Predict', command=self.predict_digit)
        self.button_predict.pack()

        self.button_clear = tk.Button(self, text='Clear', command=self.clear_canvas)
        self.button_clear.pack()

        self.canvas.bind("<B1-Motion>", self.draw)

        self.image = Image.new('L', (200, 200), 255)
        self.draw = ImageDraw.Draw(self.image)

    def draw(self, event):
        x, y = event.x, event.y
        self.canvas.create_oval(x-8, y-8, x+8, y+8, fill='black', width=20)
        self.draw.ellipse([x-8, y-8, x+8, y+8], fill='black')

    def clear_canvas(self):
        self.canvas.delete('all')
        self.image = Image.new('L', (200, 200), 255)
        self.draw = ImageDraw.Draw(self.image)

    def predict_digit(self):
        img = self.image.resize((28, 28)).convert('L')
        img = np.array(img)
        img = 255 - img  # Invert the image colors
        img = img.flatten().astype(np.float32) / 255.0
        img = self.scaler.transform([img])

        prediction = self.model.predict(img)
        digit = prediction[0]
        messagebox.showinfo("Prediction", f"The drawn digit is: {digit}")

if __name__ == "__main__":
    # Load and preprocess the data
    X, y = load_data()
    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)

    # Train the model
    model = train_model(X_train, y_train)

    # Create the GUI and run the application
    app = DigitPredictor(model, scaler)
    app.mainloop()
