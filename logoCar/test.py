import cv2
from ultralytics import YOLO
from tkinter import filedialog, Tk

# =========================
# 1. Hàm chọn file ảnh
# =========================
def choose_image():
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Chọn ảnh",
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    return file_path

# =========================
# 2. Load YOLO model
# =========================
yolo_model = YOLO("runs/detect/train5/weights/best.pt")

# =========================
# 3. Hàm detect và crop logo
# =========================
def detect_and_crop_logo(image_path):
    img = cv2.imread(image_path)
    results = yolo_model(img)

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy()
        names = yolo_model.names  # tên class khi train YOLO

        if len(boxes) > 0:
            # chọn box lớn nhất
            areas = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in boxes]
            idx = areas.index(max(areas))

            x1, y1, x2, y2 = map(int, boxes[idx])
            cropped = img[y1:y2, x1:x2]

            class_id = int(cls[idx])
            label = names[class_id]

            # Hiển thị kết quả
            print(f"Logo phát hiện: {label}")
            cv2.imshow("Logo Crop", cropped)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return label, cropped

    print("❌ Không phát hiện logo nào")
    return None, None

# =========================
# 4. Chạy thử
# =========================
img_path = choose_image()
label, cropped = detect_and_crop_logo(img_path)
