import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights
import cv2
from ultralytics import YOLO

# ==== Load model Siamese ====
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 128)

    def forward_once(self, x):
        return self.backbone(x)

    def forward(self, x1, x2):
        return self.forward_once(x1), self.forward_once(x2)

# === Transform ảnh giống lúc train ===
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ==== Load YOLO để crop xe ====
yolo_model = YOLO("yolov8n.pt")

def crop_car_largest(img_path):
    """Crop xe lớn nhất bằng YOLO"""
    results = yolo_model(img_path, verbose=False)
    max_area = 0
    best_crop = None
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if yolo_model.names[cls] == "car":
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    img = cv2.imread(img_path)
                    crop = img[y1:y2, x1:x2]
                    if crop is not None and crop.size > 0:
                        best_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                        max_area = area
    if best_crop is not None:
        return best_crop
    else:
        return Image.open(img_path).convert("RGB")

# ==== Hàm so sánh 2 ảnh ====
def compare_images(img1_path, img2_path, threshold=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = SiameseNetwork().to(device)
    model.load_state_dict(torch.load("siamese_model.pth", map_location=device))
    model.eval()

    # Crop xe
    img1 = crop_car_largest(img1_path)
    img2 = crop_car_largest(img2_path)

    # Transform
    img1 = transform(img1).unsqueeze(0).to(device)
    img2 = transform(img2).unsqueeze(0).to(device)

    # Forward
    with torch.no_grad():
        out1, out2 = model(img1, img2)
        distance = torch.nn.functional.pairwise_distance(out1, out2)

    print(f"Khoảng cách: {distance.item():.4f}")
    if distance.item() < threshold:
        print("✅ Hai ảnh là CÙNG XE")
    else:
        print("❌ Hai ảnh là KHÁC XE")

# ==== Test thử ====
compare_images("xe1 (4).jpg", "xe1B.jpg")# cùng xe 
# compare_images("xe1 (4).jpg", "clear.jpg") khác xe khác màu
# compare_images("xe1 (4).jpg", "xe97B.jpg") #khác xe cùng màu
# compare_images("bearnau.jpg", "bearnauB.jpg") #cùng may

