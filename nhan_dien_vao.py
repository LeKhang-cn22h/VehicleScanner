import cv2
import torch
import torch.nn as nn
from ultralytics import YOLO
import winsound
from dotenv import load_dotenv
import os
import cloudinary
import cloudinary.uploader
import numpy as np
import easyocr
from collections import Counter
import time


# Load biến môi trường
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn_model = CNNModel().to(device)
cnn_model.load_state_dict(torch.load("cnn_bienso_model.pth", map_location=device))
cnn_model.eval()
print("✅ Loaded CNN model (PyTorch)")

# =========================
# OCR Tools
# =========================
reader = easyocr.Reader(['en'])

def recognize_by_easyocr(plate_img):
    results = reader.readtext(plate_img)
    return "".join([res[1] for res in results]) if results else ""

# Predict char bằng CNN (PyTorch)
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
    corrections = {'I':'1','|':'1','O':'0','Q':'0','S':'5'}
    for wrong, right in corrections.items():
        text = text.replace(wrong, right)
    return text

def recognize_plate_by_cnn(plate_img, show_on_plate=True):
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    char_boxes = [(x, y, w, h) for cnt in contours for x, y, w, h in [cv2.boundingRect(cnt)] if h > 20 and w > 10]

    if not char_boxes:
        return None

    # Sắp xếp ký tự
    total_width = max(x + w for x, y, w, h in char_boxes) - min(x for x, y, w, h in char_boxes)
    total_height = max(y + h for x, y, w, h in char_boxes) - min(y for x, y, w, h in char_boxes)
    if total_width / total_height > 2:
        char_boxes.sort(key=lambda b: b[0])
    else:
        char_boxes.sort(key=lambda b: b[1])
        median_y = np.median([y for _, y, _, _ in char_boxes])
        upper = sorted([b for b in char_boxes if b[1] < median_y], key=lambda b: b[0])
        lower = sorted([b for b in char_boxes if b[1] >= median_y], key=lambda b: b[0])
        char_boxes = upper + lower

    plate_text = ""
    plate_display = plate_img.copy()

    for idx, (x, y, w, h) in enumerate(char_boxes):
        char_img = gray[y:y+h, x:x+w]
        char_pred = predict_char(char_img)
        char_pred = fix_common_ocr_mistakes(char_pred)
        plate_text += char_pred

        if show_on_plate:
            # Vẽ chữ CNN lên trên ảnh biển số
            cv2.rectangle(plate_display, (x,y), (x+w, y+h), (0,255,0), 1)
            cv2.putText(plate_display, char_pred, (x, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    if show_on_plate:
        cv2.imshow("CNN Characters on Plate", plate_display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return plate_text


def recognize_plate_by_ensemble(plate_img):
    results = []
    cnn_text = recognize_plate_by_cnn(plate_img)
    if cnn_text: results.append(cnn_text)
    easy_text = recognize_by_easyocr(plate_img)
    if easy_text: results.append(easy_text)


    print("🔍 CNN:", cnn_text, "| EasyOCR:", easy_text)
    if not results:
        return None
    return Counter(results).most_common(1)[0][0]

# =========================
# Nhận diện biển số qua webcam
# =========================
def detect_license_plate():
    model = YOLO("runs/detect/train/weights/best.pt")
    cap = cv2.VideoCapture(0)

    best_conf = 0
    best_frame = None
    best_plate = None
    last_ocr_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, device=0)
        for r in results:
            for box in (r.boxes or []):
                conf = box.conf[0].item()
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if conf > 0.8:
                    cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),2)
                    if conf > best_conf:
                        best_conf = conf
                        best_frame = frame.copy()
                        best_plate = frame[y1:y2, x1:x2]

        cv2.imshow("Nhan Dien Bien So", frame)
        current_time = time.time()
        if best_conf > 0.6 and best_plate is not None and (current_time - last_ocr_time >= 15):
            last_ocr_time = current_time
            plate_filename = "bien_so_xe_vao.jpg"
            cv2.imwrite(plate_filename, best_frame)
            url_image_vao = upload_image_to_cloudinary(plate_filename)

            recognized_plate_text = recognize_plate_by_ensemble(best_plate)
            print("Biển số quét được:", recognized_plate_text)

            display_img = best_frame.copy()
            cv2.putText(display_img, recognized_plate_text, (10,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)
            cv2.imshow("Result", display_img)
            winsound.Beep(1000, 500)

            if os.path.exists(plate_filename):
                os.remove(plate_filename)

            cv2.waitKey(0)
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return recognized_plate_text, url_image_vao
