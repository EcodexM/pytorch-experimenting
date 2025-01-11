import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import tkinter as tk
from tkinter import Canvas, messagebox
import numpy as np
from PIL import Image, ImageDraw

# Define the neural network model
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.layer1 = nn.Linear(28*28, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128,64)
        self.layer4 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x

# Train the model on the MNIST dataset
def train_model():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    model = NeuralNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 7
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    return model

# Save and load the trained model
def save_model(model, path='mnist_model.pth'):
    torch.save(model.state_dict(), path)

def load_model(path='mnist_model.pth'):
    model = NeuralNet()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

# GUI to draw and predict digits
class DigitPredictor(tk.Tk):
    def __init__(self, model):
        super().__init__()
        self.title("Digit Predictor")
        self.model = model

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
        self.canvas.create_oval(x-8, y-8, x+8, y+8, fill='black', width=2)
        self.draw.ellipse([x-8, y-8, x+8, y+8], fill='black')

    def clear_canvas(self):
        self.canvas.delete('all')
        self.image = Image.new('L', (200, 200), 255)
        self.draw = ImageDraw.Draw(self.image)

    def predict_digit(self):
        img = self.image.resize((28, 28))
        img = np.array(img)
        img = img / 255.0
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        output = self.model(img)
        _, predicted = torch.max(output.data, 1)
        digit = predicted.item()
        messagebox.showinfo("Prediction", f"The drawn digit is: {digit}")

if __name__ == "__main__":
    # Uncomment the following lines to train and save the model
    model = train_model()
    save_model(model)

    # Load the trained model
    model = load_model()

    # Create the GUI and run the application
    app = DigitPredictor(model)
    app.mainloop()
