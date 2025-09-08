import os
import cv2
import torch
import torch.nn as nn
from ultralytics import YOLO
import cloudinary
import cloudinary.uploader
import numpy as np
import easyocr
from dotenv import load_dotenv
from tkinter import Tk, filedialog

# =========================
# Load biến môi trường
# =========================
load_dotenv()

# Cấu hình Cloudinary
cloudinary.config(
    cloud_name=os.getenv('CLOUD_NAME'),
    api_key=os.getenv('CLOUD_API_KEY'),
    api_secret=os.getenv('CLOUD_API_SECRET')
)

# =========================
# PyTorch Model Setup
# =========================
CLASSES = "0123456789ABCDEFGHKLMNPRSTUVXYZ"

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
        x = self.pool(x)
        x = x.view(-1, 128 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Load model từ file .pth
device = "cpu"  # hoặc "cuda" nếu có GPU
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "../cnn_bienso_model.pth")

cnn_model = CNNModel()
cnn_model.load_state_dict(torch.load(model_path, map_location=device))
cnn_model.to(device)
cnn_model.eval()

print("✅ Loaded CNN model (PyTorch)")

# =========================
# OCR Tools
# =========================
reader = easyocr.Reader(['en'])
char_detector = YOLO(r"../trainVungKyTu/runs/detect/train/weights/best.pt")
plate_detector = YOLO("../train_bsx/runs/train/plate_detect4/weights/best.pt")

def predict_char(img):
    img = cv2.resize(img, (64, 64))
    img = img.astype(np.float32) / 255.0
    img = torch.tensor(img).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,64,64)
    with torch.no_grad():
        output = cnn_model(img)
        _, pred_idx = torch.max(output, 1)
    return CLASSES[pred_idx.item()]

def upload_image_to_cloudinary(image_path):
    try:
        response = cloudinary.uploader.upload(image_path, folder="xevao")
        return response['secure_url']
    except Exception as e:
        print("Lỗi upload:", e)
        return None

def fix_common_ocr_mistakes(text):
    corrections = {'I': '1', '|': '1', 'O': '0', 'Q': '0'}
    for wrong, right in corrections.items():
        text = text.replace(wrong, right)
    return text

# =========================
# Nhận diện ký tự bằng CNN
# =========================
def recognize_plate_by_ensemble(plate_img):
    """
    Hàm này detect + nhận diện từng ký tự trong biển số.
    """
    results = char_detector(plate_img, device=0)

    char_boxes = []
    char_images = []
    for r in results:
        for box in (r.boxes or []):
            conf = box.conf[0].item()
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if conf > 0.5:  # ngưỡng tin cậy ký tự
                char_crop = plate_img[y1:y2, x1:x2]
                char_images.append((x1, char_crop))  # giữ x1 để sort trái -> phải
                char_boxes.append((x1, y1, x2, y2))

    if not char_boxes:
        return ""

    # Sort ký tự theo vị trí ngang
    char_images = sorted(char_images, key=lambda x: x[0])
    char_boxes = sorted(char_boxes, key=lambda x: x[0])

    # Hiện bounding box từng ký tự trên biển số
    vis_plate = plate_img.copy()
    for (x1, y1, x2, y2) in char_boxes:
        cv2.rectangle(vis_plate, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imshow("Ký tự detect trong biển số", vis_plate)

    # 👉 Sau đó đưa từng crop vào CNN để classify ký tự
    recognized_text = ""
    for _, char_crop in char_images:
        char_crop = cv2.resize(char_crop, (64, 64))
        char_crop = cv2.cvtColor(char_crop, cv2.COLOR_BGR2GRAY)
        char_crop = char_crop / 255.0
        char_tensor = torch.tensor(char_crop).unsqueeze(0).unsqueeze(0).float().to(device)
        with torch.no_grad():
            output = cnn_model(char_tensor)
            pred = torch.argmax(output, dim=1).item()
        recognized_text += CLASSES[pred]

    return fix_common_ocr_mistakes(recognized_text)

# =========================
# Nhận diện biển số từ File Explorer
# =========================
def detect_license_plate():
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Chọn ảnh biển số",
        filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")]
    )
    if not file_path:
        print("❌ Bạn chưa chọn ảnh nào")
        return None, None

    frame = cv2.imread(file_path)
    results = plate_detector(frame, device=0)

    best_conf = 0
    best_plate = None
    best_box = None

    # Detect biển số xe
    for r in results:
        for box in (r.boxes or []):
            conf = box.conf[0].item()
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if conf > 0.6 and conf > best_conf:
                best_conf = conf
                best_plate = frame[y1:y2, x1:x2]  # 👉 crop biển số
                best_box = (x1, y1, x2, y2)

    if best_plate is None:
        print("❌ Không tìm thấy biển số trong ảnh.")
        return None, None

    # Vẽ bounding box biển số
    x1, y1, x2, y2 = best_box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.putText(frame, f"Plate {best_conf:.2f}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # ===== Detect ký tự trong biển số =====
    char_results = char_detector(best_plate, device=0)

    chars = []
    char_boxes = []

    for r in char_results:
        for box in (r.boxes or []):
            conf = box.conf[0].item()
            cls_id = int(box.cls[0].item())   # class id ký tự
            label = r.names[cls_id]          # tên class (ký tự)
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if conf > 0.5:
                chars.append((x1, label))
                char_boxes.append((x1, y1, x2, y2, label))

    # Sắp xếp ký tự theo vị trí x1 (từ trái qua phải)
    chars_sorted = sorted(chars, key=lambda x: x[0])
    recognized_plate_text = "".join([ch[1] for ch in chars_sorted])

    # Vẽ box ký tự
    for (x1, y1, x2, y2, label) in char_boxes:
        cv2.rectangle(best_plate, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(best_plate, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    print("Biển số nhận được:", recognized_plate_text)

    # Hiện ảnh
    cv2.imshow("Ảnh gốc detect biển số", frame)
    cv2.imshow("Biển số crop + ký tự", best_plate)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    url_image = upload_image_to_cloudinary(file_path)
    return recognized_plate_text, url_image
# =========================
# Main
# =========================
if __name__ == "__main__":
    detect_license_plate()


#
# import os
# import cv2
# import torch
# import torch.nn as nn
# from ultralytics import YOLO
# import cloudinary
# import cloudinary.uploader
# import numpy as np
# import easyocr
# from collections import Counter
# from dotenv import load_dotenv
# from tkinter import Tk, filedialog
#
# # =========================
# # Load biến môi trường
# # =========================
# load_dotenv()
#
# # Cấu hình Cloudinary
# cloudinary.config(
#     cloud_name=os.getenv('CLOUD_NAME'),
#     api_key=os.getenv('CLOUD_API_KEY'),
#     api_secret=os.getenv('CLOUD_API_SECRET')
# )
#
# # =========================
# # PyTorch Model Setup
# # =========================
# CLASSES = "0123456789ABCDEFGHKLMNPRSTUVXYZ"
#
# class CNNModel(nn.Module):
#     def __init__(self, num_classes=len(CLASSES)):
#         super(CNNModel, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
#         self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
#         self.fc1 = nn.Linear(128 * 8 * 8, 128)
#         self.dropout = nn.Dropout(0.5)
#         self.fc2 = nn.Linear(128, num_classes)
#
#     def forward(self, x):
#         x = torch.relu(self.conv1(x))
#         x = self.pool(x)
#         x = torch.relu(self.conv2(x))
#         x = self.pool(x)
#         x = torch.relu(self.conv3(x))
#         x = self.pool(x)
#         x = x.view(-1, 128 * 8 * 8)
#         x = torch.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.fc2(x)
#         return x
#
# # Load model từ file .pth
# device = "cpu"  # hoặc "cuda" nếu có GPU
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# model_path = os.path.join(BASE_DIR, "../cnn_bienso_model.pth")
#
# cnn_model = CNNModel()
# cnn_model.load_state_dict(torch.load(model_path, map_location=device))
# cnn_model.to(device)
# cnn_model.eval()
#
# print("✅ Loaded CNN model (PyTorch)")
#
# # =========================
# # OCR Tools
# # =========================
# reader = easyocr.Reader(['en'])
# char_detector = YOLO(r"../trainVungKyTu/runs/detect/train/weights/best.pt")
# plate_detector = YOLO("../train_bsx/runs/train/plate_detect4/weights/best.pt")
#
# def recognize_by_easyocr(plate_img):
#     results = reader.readtext(plate_img)
#     return "".join([res[1] for res in results]) if results else ""
#
# def predict_char(img):
#     img = cv2.resize(img, (64, 64))
#     img = img.astype(np.float32) / 255.0
#     img = torch.tensor(img).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,64,64)
#     with torch.no_grad():
#         output = cnn_model(img)
#         _, pred_idx = torch.max(output, 1)
#     return CLASSES[pred_idx.item()]
#
# def upload_image_to_cloudinary(image_path):
#     try:
#         response = cloudinary.uploader.upload(image_path, folder="xevao")
#         return response['secure_url']
#     except Exception as e:
#         print("Lỗi upload:", e)
#         return None
#
# def fix_common_ocr_mistakes(text):
#     corrections = {'I': '1', '|': '1', 'O': '0', 'Q': '0'}
#     for wrong, right in corrections.items():
#         text = text.replace(wrong, right)
#     return text
#
# # =========================
# # Nhận diện ký tự bằng CNN
# # =========================
# def recognize_plate_by_cnn(plate_img, show_on_plate=True):
#     results = char_detector.predict(plate_img, conf=0.5, verbose=False)
#
#     if not results or len(results[0].boxes) == 0:
#         return None
#
#     char_boxes = []
#     for box in results[0].boxes:
#         x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
#         w = x2 - x1
#         h = y2 - y1
#         if h / w < 0.25:  # loại gạch ngang
#             continue
#         char_boxes.append((x1, y1, x2, y2))
#
#     if not char_boxes:
#         return None
#
#     # Sắp xếp ký tự
#     total_width = max(b[2] for b in char_boxes) - min(b[0] for b in char_boxes)
#     total_height = max(b[3] for b in char_boxes) - min(b[1] for b in char_boxes)
#
#     if total_width / total_height > 2:
#         char_boxes.sort(key=lambda b: b[0])
#     else:
#         char_boxes.sort(key=lambda b: b[1])
#         median_y = np.median([y1 for _, y1, _, _ in char_boxes])
#         upper = sorted([b for b in char_boxes if b[1] < median_y], key=lambda b: b[0])
#         lower = sorted([b for b in char_boxes if b[1] >= median_y], key=lambda b: b[0])
#         char_boxes = upper + lower
#
#     # Nhận diện ký tự
#     plate_text = ""
#     gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
#     for (x1, y1, x2, y2) in char_boxes:
#         char_img = gray[y1:y2, x1:x2]
#         char_pred = predict_char(char_img)
#         char_pred = fix_common_ocr_mistakes(char_pred)
#         plate_text += char_pred
#
#     return plate_text
#
# def recognize_plate_by_ensemble(plate_img):
#     results = []
#     cnn_text = recognize_plate_by_cnn(plate_img)
#     if cnn_text: results.append(cnn_text)
#     easy_text = recognize_by_easyocr(plate_img)
#     if easy_text: results.append(easy_text)
#
#     print("🔍 CNN:", cnn_text, "| EasyOCR:", easy_text)
#     if not results:
#         return None
#     return Counter(results).most_common(1)[0][0]
#
# # =========================
# # Nhận diện biển số từ File Explorer
# # =========================
# def detect_license_plate():
#     root = Tk()
#     root.withdraw()
#     file_path = filedialog.askopenfilename(
#         title="Chọn ảnh biển số",
#         filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")]
#     )
#     if not file_path:
#         print("❌ Bạn chưa chọn ảnh nào")
#         return None, None
#
#     frame = cv2.imread(file_path)
#     results = plate_detector(frame, device=0)
#
#     best_conf = 0
#     best_plate = None
#     for r in results:
#         for box in (r.boxes or []):
#             conf = box.conf[0].item()
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             if conf > 0.6 and conf > best_conf:
#                 best_conf = conf
#                 best_plate = frame[y1:y2, x1:x2]
#
#     if best_plate is None:
#         print("❌ Không tìm thấy biển số trong ảnh.")
#         return None, None
#
#     recognized_plate_text = recognize_plate_by_ensemble(best_plate)
#     print("Biển số nhận được:", recognized_plate_text)
#
#     url_image = upload_image_to_cloudinary(file_path)
#     return recognized_plate_text, url_image
#
# # =========================
# # Main
# # =========================
# if __name__ == "__main__":
#     detect_license_plate()
