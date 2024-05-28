import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import cv2
import numpy as np

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
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(-1, 320)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return nn.functional.log_softmax(x, dim=1)

# Load the pre-trained model
model = MyNetwork()
model.load_state_dict(torch.load("mnist_model.pth"))

# Get the weights of the first layer
with torch.no_grad():
    weights = model.conv1.weight.data

# Load the first training example image
image_path = "data/sample/digit_0.jpeg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check if the image was loaded successfully
if image is None:
    print(f"Failed to load image from '{image_path}'. Please check the file path and format.")
    exit()

# Ensure the image is in float32 data type
image = image.astype(np.float32)

# Normalize the image to range [0, 1]
image /= 255.0

# Plot the original image
plt.figure(figsize=(8, 6))
plt.subplot(3, 4, 1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.axis('off')

# Apply each filter and plot the result
for i in range(10):
    filter = weights[i, 0].cpu().numpy()
    filtered_image = cv2.filter2D(image, -1, filter)
    plt.subplot(3, 4, i + 2)
    plt.imshow(filtered_image, cmap='gray')
    plt.title(f"Filter {i+1}")
    plt.axis('off')

plt.tight_layout()
plt.show()
