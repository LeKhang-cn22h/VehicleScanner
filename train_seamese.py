import os
import random
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from PIL import ImageFile
from torchvision.models import resnet18

ImageFile.LOAD_TRUNCATED_IMAGES = True
# === Hyperparameters ===
BATCH_SIZE = 4
LR = 0.001
EPOCHS = 300
IMG_SIZE = (32, 128)
DATASET_DIR = 'Dataset'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Transform ảnh ===
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# === Dataset Siamese ===
class SiameseDataset(Dataset):
    def __init__(self, dataset_dir, transform=None):
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.folders = os.listdir(dataset_dir)
        self.pairs = self.create_pairs()

    def create_pairs(self):
        pairs = []

        # Xác định thư mục cùng xe và khác xe
        same_vehicle_folders = [f"xe{i}" for i in range(1, 101)]
        different_vehicle_folders = [f"xe{i}" for i in range(101, 201)]

        # === Cặp ảnh cùng xe (label = 0)
        for folder in same_vehicle_folders:
            folder_path = os.path.join(self.dataset_dir, folder)
            if not os.path.exists(folder_path): continue
            images = os.listdir(folder_path)
            if len(images) >= 2:
                # Tạo nhiều cặp cùng xe nếu có nhiều ảnh
                for i in range(len(images)):
                    for j in range(i + 1, len(images)):
                        img1 = os.path.join(folder_path, images[i])
                        img2 = os.path.join(folder_path, images[j])
                        pairs.append((img1, img2, 0))

        # === Cặp ảnh khác xe (label = 1)
        for _ in range(len(pairs)):
            folder1, folder2 = random.sample(different_vehicle_folders, 2)
            folder1_path = os.path.join(self.dataset_dir, folder1)
            folder2_path = os.path.join(self.dataset_dir, folder2)
            if not os.path.exists(folder1_path) or not os.path.exists(folder2_path): continue
            img1_list = os.listdir(folder1_path)
            img2_list = os.listdir(folder2_path)
            if not img1_list or not img2_list: continue
            img1 = os.path.join(folder1_path, random.choice(img1_list))
            img2 = os.path.join(folder2_path, random.choice(img2_list))
            pairs.append((img1, img2, 1))

        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1_path, img2_path, label = self.pairs[idx]
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2, torch.tensor([label], dtype=torch.float32)

# === Mạng trích đặc trưng ===
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.backbone = resnet18(pretrained=True)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 128)

    def forward_once(self, x):
        return self.backbone(x)

    def forward(self, x1, x2):
        return self.forward_once(x1), self.forward_once(x2)
# === Loss Contrastive ===
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.7):
        super().__init__()
        self.margin = margin

    def forward(self, out1, out2, label):
        euclidean_distance = nn.functional.pairwise_distance(out1, out2)
        loss = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                          label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss

# === Huấn luyện ===
def train():
    dataset = SiameseDataset(DATASET_DIR, transform=transform)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=BATCH_SIZE)

    model = SiameseNetwork().to(DEVICE)
    criterion = ContrastiveLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(EPOCHS):
        total_loss = 0
        for img1, img2, label in dataloader:
            img1, img2, label = img1.to(DEVICE), img2.to(DEVICE), label.to(DEVICE)
            out1, out2 = model(img1, img2)
            loss = criterion(out1, out2, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}")
        with torch.no_grad():
            sample_img1, sample_img2, label = next(iter(dataloader))
            out1, out2 = model(sample_img1.to(DEVICE), sample_img2.to(DEVICE))
            dist = nn.functional.pairwise_distance(out1, out2)
            print(f"Sample Distance (label {label[0].item()}): {dist.mean().item():.4f}")

    torch.save(model.state_dict(), "../siamese_model.pth")
    print(" Model đã được lưu")

if __name__ == '__main__':
    train()
