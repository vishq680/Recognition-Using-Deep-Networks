import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt

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

# Load the pre-trained MNIST model
mnist_model = MyNetwork()
mnist_model.load_state_dict(torch.load("mnist_model.pth"))

# Freeze the network weights
for param in mnist_model.parameters():
    param.requires_grad = False

# Replace the last layer with a new Linear layer for three Greek letters
mnist_model.fc2 = nn.Linear(50, 3)  # Assuming the output size of the second last layer is 50 for the MNIST model

# Define the transformation for Greek letters dataset
class GreekTransform:
    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale(x)
        x = torchvision.transforms.functional.affine(x, 0, (0,0), 36/128, 0)
        x = torchvision.transforms.functional.center_crop(x, (28, 28))
        return torchvision.transforms.functional.invert(x)

# Define the path to the directory containing the three folders alpha, beta, and gamma
training_set_path = "data/greek_train"

# Create a DataLoader for the Greek dataset
greek_train = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder(training_set_path,
                                      transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                                                GreekTransform(),
                                                                                torchvision.transforms.Normalize((0.1307,), (0.3081,))])),
    batch_size=5,
    shuffle=True
)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mnist_model.parameters(), lr=0.001)

# Train the network
num_epochs = 10  # You may need to adjust this based on the convergence behavior
train_losses = []
for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in greek_train:
        optimizer.zero_grad()
        outputs = mnist_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    epoch_loss = running_loss / len(greek_train)
    train_losses.append(epoch_loss)
    print(f"Epoch {epoch+1}, Loss: {epoch_loss}")

# Plot the training error
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Error')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Error')
plt.legend()
plt.grid(True)
plt.show()

# Evaluate the network on the Greek dataset
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in greek_train:
        outputs = mnist_model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy on Greek dataset: {100 * correct / total}%")
