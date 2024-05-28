import torch
import torch.nn as nn
import matplotlib.pyplot as plt

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

# Print the model architecture
print(model)

# Get the weights of the first layer
weights = model.conv1.weight.data

# Print the shape of the weights
print("Shape of the first layer weights:", weights.shape)

# Visualize the ten filters
plt.figure(figsize=(10, 8))
for i in range(10):
    plt.subplot(3, 4, i + 1)
    plt.imshow(weights[i, 0].cpu(), cmap='gray')
    plt.title(f"Filter {i+1}")
    plt.axis('off')

plt.tight_layout()
plt.show()
