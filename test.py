import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import os
import matplotlib.pyplot as plt

# Define the network architecture
# Define the network architecture
class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.dropout = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Load the pre-trained model
model = MyNetwork()
model.load_state_dict(torch.load("mnist_model.pth"))
model.eval()

# Load and preprocess the handwritten digit images
def preprocess_image(image_path):
    # Open image
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize((28, 28), Image.LANCZOS)

    # Convert image to tensor without resizing or normalization
    img = transforms.ToTensor()(img)
    img = (img - 0.5) / 0.5

    return img

# Path to the folder containing handwritten digit images
folder_path = "data/sample"

# List of image paths for the handwritten digits
image_paths = [os.path.join(folder_path, f"digit_{i}.jpeg") for i in range(10)]

# Classify each handwritten digit and display the result
plt.figure(figsize=(10, 8))
for i, image_path in enumerate(image_paths, 1):
    # Preprocess the image
    img = preprocess_image(image_path)

    # Run the model
    with torch.no_grad():
        output = model(img.unsqueeze(0))

    # Get the predicted label
    predicted_label = torch.argmax(output).item()

    # Display the handwritten digit and its classified result
    plt.subplot(2, 5, i)
    plt.imshow(img.squeeze(), cmap='gray')
    plt.title(f"Predicted: {predicted_label}")
    plt.axis('off')

    # Print the predicted and correct labels
    print(f"Image {i}: Predicted Label: {predicted_label}")

plt.tight_layout()
plt.show()
