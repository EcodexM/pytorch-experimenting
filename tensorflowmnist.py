import tensorflow as tf
import numpy as np
import tkinter as tk
from tkinter import Canvas, Label
from PIL import Image, ImageDraw

# Load the MNIST dataset
def load_data():
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train, X_test = X_train / 255.0, X_test / 255.0
    return X_train, X_test, y_train, y_test

# Build the model
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Train the model
def train_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))
    return model

# Save the trained model
def save_model(model, path='mnist_model.h5'):
    model.save(path)

# Load the trained model
def load_model(path='mnist_model.h5'):
    return tf.keras.models.load_model(path)

# GUI to draw and predict digits
class DigitPredictor(tk.Tk):
    def __init__(self, model):
        super().__init__()
        self.title("Digit Predictor")
        self.model = model

        self.canvas = Canvas(self, width=200, height=200, bg='white')
        self.canvas.pack()

        self.label = Label(self, text="Draw a digit and see the prediction below", font=("Helvetica", 14))
        self.label.pack()

        self.result_label = Label(self, text="", font=("Helvetica", 32))
        self.result_label.pack()

        self.button_predict = tk.Button(self, text='Predict', command=self.predict_digit)
        self.button_predict.pack()

        self.button_clear = tk.Button(self, text='Clear', command=self.clear_canvas)
        self.button_clear.pack()

        self.button_train = tk.Button(self, text='Train Model', command=self.train_and_save_model)
        self.button_train.pack()

        self.button_load = tk.Button(self, text='Load Model', command=self.load_model_from_file)
        self.button_load.pack()

        self.canvas.bind("<B1-Motion>", self.draw)

        self.image = Image.new('L', (200, 200), 255)
        self.draw = ImageDraw.Draw(self.image)

    def draw(self, event):
        x, y = event.x, event.y
        self.canvas.create_oval(x-8, y-8, x+8, y+8, fill='black', width=20)
        self.draw.ellipse([x-8, y-8, x+8, y+8], fill='black')
        self.predict_digit()

    def clear_canvas(self):
        self.canvas.delete('all')
        self.image = Image.new('L', (200, 200), 255)
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="")

    def predict_digit(self):
        img = self.image.resize((28, 28)).convert('L')
        img = np.array(img)
        img = 255 - img  # Invert the image colors
        img = img.reshape((1, 28, 28))
        img = img / 255.0

        prediction = self.model.predict(img)
        digit = np.argmax(prediction)
        self.result_label.config(text=str(digit))

    def train_and_save_model(self):
        X_train, X_test, y_train, y_test = load_data()
        self.model = build_model()
        self.model = train_model(self.model, X_train, y_train, X_test, y_test)
        save_model(self.model)
        self.result_label.config(text="Model trained and saved!")

    def load_model_from_file(self):
        self.model = load_model()
        self.result_label.config(text="Model loaded!")

if __name__ == "__main__":
    # Load and preprocess the data
    X_train, X_test, y_train, y_test = load_data()

    # Build the model
    model = build_model()

    # Create the GUI and run the application
    app = DigitPredictor(model)
    app.mainloop()
