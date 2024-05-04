import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import DataLoader, TensorDataset


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, learning_rate, dropout_rate=0.2):
        """
        Initializes the Multi-Layer Perceptron model.
        :param : number of input features
        :param hidden_size: ordered LIST of the number of neurons in each hidden layer
        :param output_size: number of classes to classify
        :param learning_rate: FLOAT of the learning rate
        """
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size[0])
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.fc3 = nn.Linear(hidden_size[1], output_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.cost = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x

features = pd.read_csv('train_features.csv')
X = features.drop(columns=['Genre']).values
y = features['Genre'].values

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

scaler = StandardScaler()
X_train, X_test_val, y_train, y_test_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
X_train = scaler.fit_transform(X_train)
X_test_val = scaler.transform(X_test_val)
X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=0.5, random_state=42,
                                                stratify=y_test_val)

train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Initialize MLP model
input_size = X_train.shape[1]
hidden_size = [256,256]
output_size = len(np.unique(y_train))  # Number of classes

model = MLP(input_size, hidden_size, output_size, 0.001)
print(model)

# Define optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0.0
    correct_train = 0
    total_train = 0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = model.cost(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total_train += targets.size(0)
        correct_train += (predicted == targets).sum().item()

    train_accuracy = correct_train / total_train
    print(
        f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss / len(train_loader):.4f}, Train Accuracy: {100 * train_accuracy:.2f}%")

    # Validation
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            loss = model.cost(outputs, targets)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_val += targets.size(0)
            correct_val += (predicted == targets).sum().item()

    val_accuracy = correct_val / total_val
    print(
        f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss / len(val_loader):.4f}, Validation Accuracy: {100 * val_accuracy:.2f}%")
