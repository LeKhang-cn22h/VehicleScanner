import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset_ocr import OCRDataset
from crnn_model import CRNN
from utils import text_to_indices, ctc_decode, alphabet

transform = transforms.Compose([
    transforms.Resize((32, 128)),
    transforms.ToTensor()
])

train_set = OCRDataset("Dataset/train/image", "Dataset/train/label.txt", transform)
val_set = OCRDataset("Dataset/val/image", "Dataset/val/label.txt", transform)
train_loader = DataLoader(train_set, batch_size=16, shuffle=True, collate_fn=lambda x: x)
val_loader = DataLoader(val_set, batch_size=16, shuffle=False, collate_fn=lambda x: x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CRNN(32, num_classes=len(alphabet) + 1).to(device)
criterion = nn.CTCLoss(blank=0, zero_infinity=True)
optimizer = optim.Adam(model.parameters(), lr=0.001)
best_loss = float('inf')
for epoch in range(200):
    model.train()
    total_loss = 0
    for batch in train_loader:
        images, labels = zip(*batch)

        # Chuẩn bị input
        images = torch.stack(images).to(device)

        # Chuẩn bị label cho CTC Loss
        label_indices = [torch.tensor(text_to_indices(label), dtype=torch.long) for label in labels]
        targets = torch.cat(label_indices).to(device)
        target_lengths = torch.tensor([len(t) for t in label_indices], dtype=torch.long).to(device)

        # Forward

        preds = model(images)  # (B, T, C)
        if preds.shape[0] == images.size(0):  # (B, T, C)
            preds = preds.permute(1, 0, 2)
        preds_log_softmax = preds.log_softmax(2)
        input_lengths = torch.full(size=(images.size(0),), fill_value=preds.size(1), dtype=torch.long).to(device)
        print("input_lengths:", input_lengths)
        print("batch size:", images.size(0))
        print("targets:", targets)
        print("targets.shape:", targets.shape)
        print("target_lengths:", target_lengths)
        print("target_lengths.shape:", target_lengths.shape)
        print("preds.shape:", preds.shape)
        print("preds_log_softmax.shape:", preds_log_softmax.shape)
        print("input_lengths.shape:", input_lengths.shape)

        # Tính loss
        loss = criterion(preds_log_softmax, targets, input_lengths, target_lengths)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    avg_loss = total_loss / len(train_loader)

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), "../../crnn_best.pth")