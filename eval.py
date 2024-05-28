import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
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

# Load the test dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)

# Plot the first 9 digits of the test set
fig, axs = plt.subplots(3, 3, figsize=(8, 8))
for i in range(9):
    image, label = test_set[i]
    axs[i//3, i%3].imshow(image.squeeze().numpy(), cmap='gray')
    axs[i//3, i%3].set_title(f"Label: {label}")
    axs[i//3, i%3].axis('off')

# Run the model on the first 10 examples in the test set
print("Predictions for the first 10 examples in the test set:")
for i, (image, label) in enumerate(test_loader):
    if i >= 10:
        break
    output = model(image)
    probabilities = torch.exp(output)
    predicted_label = torch.argmax(probabilities, dim=1).item()
    print(f"Example {i+1}:")
    print("Network Output Values:", [f"{prob:.2f}" for prob in probabilities.squeeze().tolist()])
    print("Predicted Label Index:", predicted_label)
    print("Correct Label:", label.item())

plt.tight_layout()
plt.show()
