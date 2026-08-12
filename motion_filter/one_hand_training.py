import os
import glob
import csv
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from tqdm import tqdm

# ==========================
# CONFIG
# ==========================
DATA_DIR = "../data/one_hand"
MODEL_PATH = "one_hand_motion_model.pth"
META_PATH = "one_hand_motion_meta.json"
BATCH_SIZE = 32
EPOCHS = 30
VAL_SPLIT = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================
# DATASET
# ==========================
class LandmarkDataset(Dataset):
    def __init__(self, csv_dir):
        self.samples = []
        self.labels = []
        self.signs = []

        csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
        for file in csv_files:
            sign_name = os.path.splitext(os.path.basename(file))[0]
            if sign_name not in self.signs:
                self.signs.append(sign_name)

            with open(file, "r") as f:
                reader = csv.reader(f)
                header = next(reader)  # skip header
                for row in reader:
                    features = list(map(float, row[2:]))  # skip frame, hand
                    self.samples.append(features)
                    self.labels.append(self.signs.index(sign_name))

        self.samples = np.array(self.samples, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)

        print(f"📊 Loaded {len(self.samples)} samples from {len(self.signs)} classes: {self.signs}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx]), torch.tensor(self.labels[idx])

# ==========================
# MODEL
# ==========================
class LandmarkMLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# ==========================
# TRAIN FUNCTION
# ==========================
def train():
    dataset = LandmarkDataset(DATA_DIR)
    num_classes = len(dataset.signs)
    input_size = dataset.samples.shape[1]

    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

    model = LandmarkMLP(input_size=input_size, num_classes=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(EPOCHS):
        # Train
        model.train()
        running_loss, correct, total = 0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

        train_acc = 100. * correct / total

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = outputs.max(1)
                val_correct += preds.eq(labels).sum().item()
                val_total += labels.size(0)

        val_acc = 100. * val_correct / val_total
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {running_loss/len(train_loader):.4f}, "
              f"Acc: {train_acc:.2f}% | Val Loss: {val_loss/len(val_loader):.4f}, Acc: {val_acc:.2f}%")

    # Save model + meta
    torch.save(model.state_dict(), MODEL_PATH)
    meta = {"classes": dataset.signs, "input_size": input_size}
    with open(META_PATH, "w") as f:
        json.dump(meta, f)
    print(f"✅ Model saved to {MODEL_PATH}")
    print(f"✅ Meta saved to {META_PATH}")

if __name__ == "__main__":
    train()

