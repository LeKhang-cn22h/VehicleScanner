import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

MODEL_PATH = 'siamese_model.pth'

# === Định nghĩa lại SiameseNetwork giống lúc train ===
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.backbone = models.resnet18(pretrained=False)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 128)

    def forward_once(self, x):
        return self.backbone(x)

    def forward(self, x1, x2):
        return self.forward_once(x1), self.forward_once(x2)

# === Tiền xử lý ảnh ===
transform = transforms.Compose([
    transforms.Resize((32, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def load_and_prepare_image(path):
    image = Image.open(path).convert("RGB")
    return transform(image).unsqueeze(0)  # [1, 3, H, W]

def main():
    # Load model
    model = SiameseNetwork()
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()

    # ==== Thay đường dẫn 2 ảnh cần so sánh ====
    img_path1 = "1.jpg"
    img_path2 = "3.jpg"

    img1 = load_and_prepare_image(img_path1)
    img2 = load_and_prepare_image(img_path2)

    with torch.no_grad():
        out1, out2 = model(img1, img2)
        euclidean_distance = F.pairwise_distance(out1, out2)

    print(f"Khoảng cách Euclidean: {euclidean_distance.item():.4f}")

    threshold = 0.7  # giống margin lúc train
    if euclidean_distance.item() < threshold:
        print("Hai ảnh giống nhau")
    else:
        print("Hai ảnh khác nhau")

if __name__ == "__main__":
    main()
