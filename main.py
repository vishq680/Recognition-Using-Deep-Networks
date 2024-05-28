import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F

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

def evaluate_model(model, criterion, data_loader):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    average_loss = total_loss / len(data_loader)

    return accuracy, average_loss

def train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, num_epochs=5):
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    train_examples_seen = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_accuracy = 100 * correct / total
        train_accuracies.append(train_accuracy)

        # Calculate the total number of training examples seen after each epoch
        total_examples_seen = (epoch + 1) * len(train_loader.dataset)
        train_examples_seen.append(total_examples_seen)

        # Append the average loss after each epoch
        train_losses.append(running_loss / len(train_loader))

        # Evaluate the model on the test set
        test_accuracy, test_loss = evaluate_model(model, criterion, test_loader)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        print(f'Epoch {epoch + 1}/{num_epochs}, '
              f'Train Loss: {running_loss / len(train_loader):.4f}, '
              f'Train Accuracy: {train_accuracy:.2f}%, '
              f'Test Loss: {test_loss:.4f}, '
              f'Test Accuracy: {test_accuracy:.2f}%')

    # Plot training and testing errors
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Error', color='blue')
    plt.plot(range(1, num_epochs + 1), test_losses, 'o', label='Testing Error', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Negative Log Likelihood Loss')
    plt.title('Training and Testing Negative Log Likelihood Losses')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot training and testing accuracies
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_accuracies, label='Training Accuracy', color='blue')
    plt.plot(range(1, num_epochs + 1), test_accuracies, label='Testing Accuracy', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Testing Accuracies')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot number of training examples seen vs negative log likelihood loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_examples_seen, train_losses, label='Training Loss', color='blue')
    plt.plot(train_examples_seen, test_losses, 'o', label='Testing Loss', color='red')
    plt.xlabel('Number of Training Examples Seen')
    plt.ylabel('Negative Log Likelihood Loss')
    plt.title('Number of Training Examples Seen vs Negative Log Likelihood Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)
    print("Model saved successfully.")

def load_data(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def plot_examples(test_loader):
    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()

def main():
    # Load data
    train_loader, test_loader = load_data()

    # Plot examples
    plot_examples(test_loader)

    # Initialize network
    model = MyNetwork()

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train and evaluate the network
    train_and_evaluate(model, train_loader, test_loader, criterion, optimizer)

    # Save the trained model
    save_model(model, "mnist_model.pth")

if __name__ == "__main__":
    main()
