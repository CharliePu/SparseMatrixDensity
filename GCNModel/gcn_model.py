import sys
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, LayerNorm, global_mean_pool
from torch_geometric.data import DataLoader
from tqdm import tqdm

import matplotlib.pyplot as plt

from dataset import SparseMatrixDataset

# Define the GCN network architecture
class GCN_NET(nn.Module):
    def __init__(self, features, classes):
        super(GCN_NET, self).__init__()
        self.n1conv1 = GCNConv(features, 256)
        self.n1ln1 = LayerNorm(256)
        self.n1drop1 = nn.Dropout(0.1)
        self.n1conv2 = GCNConv(256, 256)
        self.n1ln2 = LayerNorm(256)
        self.n1drop2 = nn.Dropout(0.1)

        self.n2conv1 = GCNConv(features, 256)
        self.n2ln1 = LayerNorm(256)
        self.n2drop1 = nn.Dropout(0.1)
        self.n2conv2 = GCNConv(256, 256)
        self.n2ln2 = LayerNorm(256)
        self.n2drop2 = nn.Dropout(0.1)

        self.linear1 = nn.Linear(512, 128)
        self.linear2 = nn.Linear(128, classes) # class should be one for regression


    def forward(self, data):
        # network 1
        n1 = data["m1"]
        n1_x, n1_edge_index = n1.x, n1.edge_index

        n1_x = self.n1conv1(n1_x, n1_edge_index)
        n1_x = self.n1ln1(n1_x)
        n1_x = F.relu(n1_x)
        n1_x = self.n1drop1(n1_x)

        n1_x = self.n1conv2(n1_x, n1_edge_index)
        n1_x = self.n1ln2(n1_x)
        n1_x = F.relu(n1_x)
        n1_x = self.n1drop2(n1_x)

        n1_x = global_mean_pool(n1_x, n1.batch)

        # network 2
        n2 = data["m2"]
        n2_x, n2_edge_index = n2.x, n2.edge_index

        n2_x = self.n2conv1(n2_x, n2_edge_index)
        n2_x = self.n2ln1(n2_x)
        n2_x = F.relu(n2_x)
        n2_x = self.n2drop1(n2_x)

        n2_x = self.n2conv2(n2_x, n2_edge_index)
        n2_x = self.n2ln2(n2_x)
        n2_x = F.relu(n2_x)
        n2_x = self.n2drop2(n2_x)

        n2_x = global_mean_pool(n2_x, n2.batch)

        # merge networks
        x = torch.cat([n1_x, n2_x], dim=1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x

# Function to calculate model loss
def calculate_loss(model, loader, device):
    model.eval()
    total_samples, total_loss = 0, 0

    with torch.no_grad():
        for batch in train_loader:
            batch["m1"].to(device)
            batch["m2"].to(device)
            out = model(batch)
            loss = F.mse_loss(out, batch["prod_nnz_density"].to(device))

            total_samples += batch["prod_nnz_density"].size(dim=0)
            total_loss += loss.item()

    return total_loss / len(loader)

if __name__ == '__main__':
    # Load dataset
    dataset = SparseMatrixDataset(root="./dataset", name="test")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dataset info: Features={dataset.num_node_features}, Classes={dataset.num_classes}")

    # Initialize model and data loaders
    model = GCN_NET(dataset.num_node_features, dataset.num_classes).to(device)

    # Split dataset into training, validation, and test sets
    torch.manual_seed(123456789)
    num_graphs = len(dataset)
    num_training, num_validation = int(num_graphs * 0.7), int(num_graphs * 0.15)
    num_test = num_graphs - num_training - num_validation
    training_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(dataset, [num_training, num_validation, num_test])

    train_loader = DataLoader(training_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(validation_dataset, batch_size=2)
    test_loader = DataLoader(test_dataset, batch_size=2)

    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    num_epochs = 1
    train_loss_values, val_loss_values = [], []
    best_validation_loss = float('inf')

    # Training loop
    for epoch in tqdm(range(num_epochs), desc='Training', ncols=100):
        total_loss, total_samples = 0, 0
        model.train()
        
        for batch in train_loader:
            optimizer.zero_grad()

            batch["m1"].to(device)
            batch["m2"].to(device)
            out = model(batch)
            loss = F.mse_loss(out, batch["prod_nnz_density"].to(device))
            
            total_samples += batch["prod_nnz_density"].size(dim=0)
            total_loss += loss.item()
            
            loss.backward()
            optimizer.step()

        # Record and print epoch results
        avg_loss = total_loss / len(train_loader)
        train_loss_values.append(avg_loss)

        val_loss = calculate_loss(model, val_loader, device)
        val_loss_values.append(val_loss)

        if val_loss < best_validation_loss:
            best_validation_loss = val_loss
            best_model = model.state_dict()

        tqdm.write(f'Epoch: {epoch + 1}, Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}')

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # Save best model & last model
    if not os.path.exists(f"./models/{dataset.dataset_name}"):
        os.makedirs(f"./models/{dataset.dataset_name}")
    torch.save(best_model, f"./models/{dataset.dataset_name}/{timestamp}_best_model.pth")
    torch.save(model.state_dict(), f"./models/{dataset.dataset_name}/{timestamp}_last_model.pth")

    # Final evaluation on test set
    test_loss = calculate_loss(model, test_loader, device)
    print(f'Test Loss: {test_loss:.4f}')

    # Save loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss_values, label='Training Loss')
    plt.plot(val_loss_values, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curve')
    plt.legend()

    path = f"./imgs/loss/{dataset.dataset_name}"

    if not os.path.exists(path):
        os.makedirs(path)
    
    plt.savefig(f"{path}/{timestamp}_{test_loss:.4f}_loss_curve.png")
    plt.close()