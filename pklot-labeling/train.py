import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets
from torch.utils.data import random_split
from torchvision.models.resnet import ResNet50_Weights
from datetime import datetime

def train(dataset, ratio, num_epochs, num_classes):
    """ Train a model using the given dataset
    @param dataset: The dataset to train the model on
    @param ratio: The ratio of data to be used for training
    @param num_epochs: The number of epochs to train the model
    @param num_classes: The number of output classes
    """

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Split the dataset into training and validation sets
    train_ratio = ratio  # Percentage of data to be used for training (80% in this example)
    val_ratio = 1 - train_ratio  # Remaining percentage for validation

    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = dataset_size - train_size

    # Split the dataset into training and validation sets
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=200, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=150, shuffle=False)

    # Load the pre-trained ResNet50 model
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    num_features = model.fc.in_features

    # Replace the last fully connected layer for fine-tuning
    model.fc = nn.Linear(num_features, num_classes)  # num_classes is the number of output classes

    # use DataParallel to train on multiple GPUs
    model = nn.DataParallel(model)

    # Move the model to the device
    model = model.to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # best validation loss
    valid_losses_min = np.Inf

    # Training loop
    for epoch in range(num_epochs):

        # Initialize the running loss and correct predictions
        train_loss = 0.0
        train_correct = 0.0

        # Set the model to training mode
        model.train()

        # image index
        ii =0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            _, predictions = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_correct += torch.sum(predictions == labels).item()
            now = datetime.now()
            current_time = now.strftime("%Y-%m-%d %H:%M:%S")
            print("{} train {}: {}/{}: {}".format(current_time, epoch, ii, train_size, train_loss))
            ii += 1

        # Calculate the average loss and accuracy
        train_loss /= len(train_loader.dataset)
        train_accuracy = train_correct / len(train_loader.dataset)

        # Initialize the validation loss and correct predictions
        val_loss = 0.0
        val_correct = 0.0

        # Set the model to evaluation mode
        model.eval()

        ii = 0

        # Validation loop
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, predictions = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                val_correct += torch.sum(predictions == labels).item()
                ii += 1
                now = datetime.now()
                current_time = now.strftime("%Y-%m-%d %H:%M:%S")
                print("{} val {}: {}/{}: {}".format(current_time, epoch, ii, val_size, val_loss))

        # Calculate the average loss and accuracy
        val_loss /= len(val_loader.dataset)
        val_accuracy = val_correct / len(val_loader.dataset)

        # Save the model with the best validation loss
        if val_loss <= valid_losses_min:
            print("best loss found")
            valid_losses_min = val_loss
            torch.save(model.state_dict(), "resnet50-best.pth")
        else:
            torch.save(model.state_dict(), "resnet50-" + str(epoch) + ".pth")

        print(f"Epoch {epoch+1}/{num_epochs}: "
            f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), "resnet50.pth")
