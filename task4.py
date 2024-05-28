import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
import time
import matplotlib.pyplot as plt



def plot_results(results):
    for num_conv_layers in num_conv_layers_options:
        for num_hidden_nodes in num_hidden_nodes_options:
            plt.figure(figsize=(10, 5))

            accuracies = []
            training_times = []

            for num_epochs in num_epochs_options:
                for result in results:
                    if result[:3] == (num_conv_layers, num_hidden_nodes, num_epochs):
                        accuracies.append(result[3])
                        training_times.append(result[4])

            plt.subplot(1, 2, 1)
            plt.plot(num_epochs_options, accuracies, marker='o')
            plt.title(f'Accuracy vs. Number of Epochs (Conv Layers={num_conv_layers}, Hidden Nodes={num_hidden_nodes})')
            plt.xlabel('Number of Epochs')
            plt.ylabel('Accuracy (%)')

            plt.subplot(1, 2, 2)
            plt.plot(num_epochs_options, training_times, marker='o')
            plt.title(f'Training Time vs. Number of Epochs (Conv Layers={num_conv_layers}, Hidden Nodes={num_hidden_nodes})')
            plt.xlabel('Number of Epochs')
            plt.ylabel('Training Time (s)')

            plt.tight_layout()
            plt.show()


# Define the network architecture
class MyNetwork(nn.Module):
    def __init__(self, num_conv_layers, num_hidden_nodes):
        super(MyNetwork, self).__init__()
        self.num_conv_layers = num_conv_layers
        self.num_hidden_nodes = num_hidden_nodes
        
        # Define the convolutional layers
        self.conv_layers = nn.ModuleList()
        in_channels = 1  # Input channels for the first convolutional layer
        for _ in range(num_conv_layers):
            self.conv_layers.append(nn.Conv2d(in_channels, 10, kernel_size=3))
            in_channels = 10  # Update input channels for subsequent layers
        
        # Calculate the size of the input to the fully connected layers
        self.calculate_fc_input_size()
        
        # Define the fully connected layers
        self.fc1 = nn.Linear(self.fc_input_size, num_hidden_nodes)
        self.fc2 = nn.Linear(num_hidden_nodes, 10)  # Output layer
        
    def calculate_fc_input_size(self):
        # Dummy input to calculate the size after passing through convolutional layers
        x = torch.randn(1, 1, 28, 28)
        for conv_layer in self.conv_layers:
            x = F.relu(F.max_pool2d(conv_layer(x), 2))
        self.fc_input_size = x.view(-1).size(0)
        
    def forward(self, x):
        # Forward pass through convolutional layers
        for conv_layer in self.conv_layers:
            x = F.relu(F.max_pool2d(conv_layer(x), 2))
        
        # Flatten the output for fully connected layers
        x = x.view(-1, self.fc_input_size)
        
        # Forward pass through fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train_model(model, train_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        start_time = time.time()  # Start time for the epoch
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
        end_time = time.time()  # End time for the epoch
        training_time = end_time - start_time  # Calculate training time for the epoch
        print(f'Epoch [{epoch + 1}/{num_epochs}], Training Time: {training_time:.2f}s')
    return training_time


def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def main():
    # Load data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)

    # Define search parameters
    num_conv_layers_options = [2, 3]
    num_hidden_nodes_options = [50, 100]
    num_epochs_options = [5, 10, 15]

    # Evaluate network variations
    results = []
    for num_conv_layers in num_conv_layers_options:
        for num_hidden_nodes in num_hidden_nodes_options:
            for num_epochs in num_epochs_options:
                model = MyNetwork(num_conv_layers, num_hidden_nodes)
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=0.001)
                training_time = train_model(model, train_loader, criterion, optimizer, num_epochs)
                accuracy = evaluate_model(model, test_loader)
                results.append((num_conv_layers, num_hidden_nodes, num_epochs, accuracy, training_time))

    # Print results
    print("Num Conv Layers | Num Hidden Nodes | Num Epochs | Accuracy (%) | Training Time (s)")
    for result in results:
        print("{:<15} | {:<16} | {:<11} | {:<13.2f} | {:.2f}".format(*result))
        
    plot_results(results)


if __name__ == "__main__":
    main()
