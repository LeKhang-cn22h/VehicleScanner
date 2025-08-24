import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pandas as pd

# -------------------------
# Cấu hình thiết bị (GPU/CPU)
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Thiết bị:", device)

# -------------------------
# Danh sách ký tự & ánh xạ index
# -------------------------
CLASSES = "0123456789ABCDEFGHKLMNPRSTUVXYZ"
char_to_idx = {c: i for i, c in enumerate(CLASSES)}
IMG_SIZE = 64

# -------------------------
# Dataset Class
# -------------------------
class LicensePlateDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        for label in os.listdir(root_dir):
            label_path = os.path.join(root_dir, label)
            if not os.path.isdir(label_path):
                continue
            for img_name in os.listdir(label_path):
                img_path = os.path.join(label_path, img_name)
                self.samples.append((img_path, char_to_idx[label]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)  # (1, 64, 64)
        return torch.tensor(img, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# -------------------------
# Tạo DataLoader
# -------------------------
train_dataset = LicensePlateDataset("datasetkytu/train")
test_dataset = LicensePlateDataset("datasetkytu/test")

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Số ảnh train: {len(train_dataset)}, test: {len(test_dataset)}")

# -------------------------
# Định nghĩa CNN Model
# -------------------------
class CNNModel(nn.Module):
    def __init__(self, num_classes=len(CLASSES)):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc1 = nn.Linear(128 * 8 * 8, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.relu(self.conv3(x))
        x = self.pool(x)  # còn 8x8
        x = x.view(-1, 128 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# -------------------------
# Khởi tạo model, loss, optimizer
# -------------------------
model = CNNModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -------------------------
# Lưu số liệu huấn luyện
# -------------------------
train_losses, train_accuracies = [], []
test_losses, test_accuracies = [], []

# -------------------------
# Vòng lặp huấn luyện
# -------------------------
epochs = 100
for epoch in range(epochs):
    # ----- Train -----
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total
    train_losses.append(avg_train_loss)
    train_accuracies.append(train_acc)

    # ----- Test -----
    model.eval()
    running_loss_test = 0.0
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss_test += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

    avg_test_loss = running_loss_test / len(test_loader)
    test_acc = 100 * correct_test / total_test
    test_losses.append(avg_test_loss)
    test_accuracies.append(test_acc)

    print(f"Epoch [{epoch+1}/{epochs}] "
          f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
          f"Test Loss: {avg_test_loss:.4f}, Test Acc: {test_acc:.2f}%")

# -------------------------
# Lưu mô hình
# -------------------------
torch.save(model.state_dict(), "cnn_bienso_model.pth")
print("✅ Mô hình đã được lưu tại cnn_bienso_model.pth")

# -------------------------
# Xuất số liệu ra CSV
# -------------------------
df = pd.DataFrame({
    "Epoch": list(range(1, epochs+1)),
    "Train_Loss": train_losses,
    "Train_Acc": train_accuracies,
    "Test_Loss": test_losses,
    "Test_Acc": test_accuracies
})
df.to_csv("training_metrics.csv", index=False)
print("✅ Lưu số liệu huấn luyện ra training_metrics.csv")

# -------------------------
# Vẽ biểu đồ Loss & Accuracy
# -------------------------
plt.figure(figsize=(12,5))

# Loss
plt.subplot(1,2,1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.title('Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Accuracy
plt.subplot(1,2,2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.title('Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("training_charts.png", dpi=300)
plt.show()
print("✅ Lưu biểu đồ huấn luyện ra training_charts.png")
