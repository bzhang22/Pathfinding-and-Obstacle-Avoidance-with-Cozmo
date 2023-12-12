import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch.optim as optim

class CustomImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        # Specify dtype=float for the label columns if they are numeric
        self.img_labels = pd.read_csv(csv_file, dtype={"label_column_1": float, "label_column_2": float, "label_column_3": float})
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path).convert('RGB')
        # Ensure the labels are extracted as a numeric array
        label = self.img_labels.iloc[idx, 1:].values.astype(np.float32)
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label)

# Example usage

# Example usage
transform = transforms.Compose([
    transforms.Resize((320, 240)),
    transforms.ToTensor(),
])
dataset = CustomImageDataset(csv_file='training_data_cube_2.csv', img_dir='./cubedata_2', transform=transform)



# Calculate lengths of splits
total_size = len(dataset)
train_size = int(0.8 * total_size)
test_size = total_size - train_size

# Split the dataset
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create data loaders for empty and testing sets
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)




class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.fc1 = nn.Linear(80 * 60 * 32, 128)  # Updated input size

        # Output layers
        self.fc_classification = nn.Linear(128, 1)  # For binary classification
        self.fc_regression = nn.Linear(128, 1)  # For regression (left distance)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))

        # Classification output (is_there_an_obstacle)
        out_classification = torch.sigmoid(self.fc_classification(x))

        # Regression output (left_distance)
        out_regression = self.fc_regression(x)

        return out_classification, out_regression


model = CustomCNN()









# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Assuming binary cross-entropy loss for classification and mean squared error loss for regression
classification_loss_fn = nn.BCELoss()
regression_loss_fn = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(200):  # Number of epochs
    # Training phase
    model.train()  # Set the model to training mode
    running_train_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()

        # Split labels for classification and regression
        labels_classification = labels[:, 0].unsqueeze(1)  # Reshape for BCELoss
        labels_regression = labels[:, 1].unsqueeze(1)

        # Forward pass
        outputs_classification, outputs_regression = model(inputs)

        # Calculate losses for both tasks
        loss_classification = classification_loss_fn(outputs_classification, labels_classification)
        loss_regression = regression_loss_fn(outputs_regression, labels_regression)

        # Combine losses
        loss = loss_classification + loss_regression

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item()

    # Calculate average training loss
    avg_train_loss = running_train_loss / len(train_loader)
    print(f'Epoch {epoch + 1}, Training loss: {avg_train_loss:.3f}')

    # Testing phase
    model.eval()  # Set the model to evaluation mode
    running_test_loss = 0.0
    with torch.no_grad():  # No gradient calculation for testing phase
        for inputs, labels in test_loader:
            # Split labels for classification and regression
            labels_classification = labels[:, 0].unsqueeze(1)
            labels_regression = labels[:, 1].unsqueeze(1)

            # Forward pass
            outputs_classification, outputs_regression = model(inputs)

            # Calculate losses for both tasks
            loss_classification = classification_loss_fn(outputs_classification, labels_classification)
            loss_regression = regression_loss_fn(outputs_regression, labels_regression)

            # Combine losses
            loss = loss_classification + loss_regression

            running_test_loss += loss.item()

        # Calculate average testing loss
        avg_test_loss = running_test_loss / len(test_loader)
        print(f'Epoch {epoch + 1}, Testing loss: {avg_test_loss:.3f}')

print('Finished Training')

model_path = os.path.join('', 'ml_models/cozmo_model_test.pt')
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")


import numpy as np
from sklearn.metrics import confusion_matrix, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming your model and test_loader are defined and loaded as per your setup
model.eval()  # Set the model to evaluation mode

# Variables to accumulate predictions and labels
all_classification_preds = []
all_classification_labels = []
all_regression_preds = []
all_regression_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        # Split labels for classification and regression
        labels_classification = labels[:, 0].unsqueeze(1)  # Assuming first column is classification
        labels_regression = labels[:, 1].unsqueeze(1)  # Assuming second column is regression

        # Forward pass
        outputs_classification, outputs_regression = model(inputs)

        # Accumulate predictions and labels for evaluation
        all_classification_preds.extend(outputs_classification.view(-1).cpu().numpy())
        all_classification_labels.extend(labels_classification.view(-1).cpu().numpy())
        all_regression_preds.extend(outputs_regression.view(-1).cpu().numpy())
        all_regression_labels.extend(labels_regression.view(-1).cpu().numpy())

# Process classification predictions for confusion matrix
threshold = 0.5
binary_classification_preds = [1 if pred > threshold else 0 for pred in all_classification_preds]

# Calculate and print Confusion Matrix
conf_matrix = confusion_matrix(all_classification_labels, binary_classification_preds)
print("Confusion Matrix:\n", conf_matrix)

# Plotting Confusion Matrix
plt.figure(figsize=(8, 6))

sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues',
            xticklabels=['Not Obstacle', 'Obstacle'],
            yticklabels=['Not Obstacle', 'Obstacle'],
            annot_kws={"size": 16})  # Increase font size for the annotations

plt.xlabel('Predicted', fontsize=14)
plt.ylabel('Actual', fontsize=14)
plt.title('Confusion Matrix', fontsize=16)
plt.show()

