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
    corrections = {'I':'1','|':'1','O':'0','Q':'0'}
    for wrong, right in corrections.items():
        text = text.replace(wrong, right)
    return text

# =========================
# Nhận diện ký tự bằng CNN
# =========================
def recognize_plate_by_cnn(plate_img, show_on_plate=True):
    # Detect ký tự bằng YOLO
    char_detector = YOLO(r"trainVungKyTu/runs/detect/train/weights/best.pt")


    results = char_detector.predict(plate_img, conf=0.5, verbose=False)

    if not results or len(results[0].boxes) == 0:
        return None

    char_boxes = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        w = x2 - x1
        h = y2 - y1

        # Loại bỏ box quá dẹt (nghi là gạch ngang)
        if h / w < 0.25:  # ngưỡng bạn có thể chỉnh (0.2 ~ 0.3)
            continue

        char_boxes.append((x1, y1, x2, y2))

    #
    # char_boxes = []
    # for box in results[0].boxes:
    #     x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
    #     char_boxes.append((x1, y1, x2, y2))

    # Nếu không có box nào
    if not char_boxes:
        return None

    # --- Sắp xếp ký tự ---
    total_width = max(b[2] for b in char_boxes) - min(b[0] for b in char_boxes)
    total_height = max(b[3] for b in char_boxes) - min(b[1] for b in char_boxes)

    if total_width / total_height > 2:
        # 1 hàng ngang → sort theo x
        char_boxes.sort(key=lambda b: b[0])
    else:
        # 2 hàng dọc → sort theo y rồi tách upper/lower
        char_boxes.sort(key=lambda b: b[1])
        median_y = np.median([y1 for _, y1, _, _ in char_boxes])
        upper = sorted([b for b in char_boxes if b[1] < median_y], key=lambda b: b[0])
        lower = sorted([b for b in char_boxes if b[1] >= median_y], key=lambda b: b[0])
        char_boxes = upper + lower

    # --- Nhận dạng ký tự bằng CNN ---
    plate_text = ""
    plate_display = plate_img.copy()
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

    for (x1, y1, x2, y2) in char_boxes:
        char_img = gray[y1:y2, x1:x2]
        char_pred = predict_char(char_img)
        char_pred = fix_common_ocr_mistakes(char_pred)
        plate_text += char_pred

        if show_on_plate:
            cv2.rectangle(plate_display, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(plate_display, char_pred, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    if show_on_plate:
        cv2.imshow("YOLO+CNN Characters on Plate", plate_display)
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
# Nhận diện biển số qua Webcam
# =========================
from tkinter import Tk, filedialog

def detect_license_plate():
    # Mở hộp thoại chọn file ảnh
    root = Tk()
    root.withdraw()
    root.lift()
    root.attributes('-topmost', True)
    file_path = filedialog.askopenfilename(
        title="Chọn ảnh biển số",
        filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")]
    )

    if not file_path:
        print(" Không chọn file.")
        return None, None

    # Đọc ảnh từ file
    frame = cv2.imread(file_path)
    model = YOLO("runs/detect/train/weights/best.pt")

    results = model(frame, device=0)
    best_conf = 0
    best_plate = None
    best_frame = None

    for r in results:
        for box in (r.boxes or []):
            conf = box.conf[0].item()
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if conf > 0.6:  # Ngưỡng tự tin
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if conf > best_conf:
                    best_conf = conf
                    best_frame = frame.copy()
                    best_plate = frame[y1:y2, x1:x2]

    if best_plate is None:
        print("❌ Không phát hiện được biển số.")
        return None, None

    # Nhận diện biển số
    recognized_plate_text = recognize_plate_by_ensemble(best_plate)
    print("Biển số quét được:", recognized_plate_text)

    # Upload ảnh lên Cloudinary
    url_image_vao = upload_image_to_cloudinary(file_path)

    # Hiển thị kết quả
    display_img = best_frame.copy()
    cv2.putText(display_img, recognized_plate_text, (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    cv2.imshow("Kết quả nhận diện", display_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return recognized_plate_text, url_image_vao

# =========================
# Nhận diện biển số từ File Explorer
# =========================
def detect_from_file():
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
    if frame is None:
        print("Không đọc được ảnh:", file_path)
        return None, None

    model = YOLO("runs/detect/train/weights/best.pt")
    results = model(frame, device=0)

    best_conf = 0
    best_plate = None
    for r in results:
        for box in (r.boxes or []):
            conf = box.conf[0].item()
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if conf > 0.6:
                cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),2)
                if conf > best_conf:
                    best_conf = conf
                    best_plate = frame[y1:y2, x1:x2]

    if best_plate is None:
        print("❌ Không tìm thấy biển số trong ảnh.")
        return None, None

    recognized_plate_text = recognize_plate_by_ensemble(best_plate)
    print(" Biển số nhận được:", recognized_plate_text)

    url_image = upload_image_to_cloudinary(file_path)

    display_img = frame.copy()
    cv2.putText(display_img, recognized_plate_text, (10,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)
    # Giữ tỉ lệ ảnh nhưng giới hạn chiều rộng hoặc chiều cao





    # cv2.imshow("Result", display_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return recognized_plate_text, url_image

# =========================
# Main
# =========================
if __name__ == "__main__":
    # mode = input("Chọn chế độ (1=Webcam, 2=Ảnh từ File Explorer): ")
    # if mode == "1":
    #     detect_license_plate()
    # else:
    detect_from_file()
