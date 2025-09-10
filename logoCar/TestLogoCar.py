import cv2
from ultralytics import YOLO
from tkinter import filedialog, Tk

# =========================
# 1. Chọn ảnh test bằng hộp thoại
# =========================
def choose_image():
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Chọn ảnh test",
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    return file_path

# =========================
# 2. Load model đã train xong (best.pt)
# =========================
model = YOLO("runs/train/logo_experiment/weights/best.pt")

# =========================
# 3. Hàm detect logo
# =========================
def detect_logo(image_path):
    img = cv2.imread(image_path)
    results = model(img)   # chạy detect

    names = model.names    # tên class trong dataset

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy()
        conf = r.boxes.conf.cpu().numpy()

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            class_id = int(cls[i])
            label = names[class_id]
            confidence = conf[i]

            # Vẽ bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{label} {confidence:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)

            print(f"✅ Logo phát hiện: {label} ({confidence:.2f})")

    cv2.imshow("Logo Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# =========================
# 4. Run thử
# =========================
if __name__ == "__main__":
    img_path = choose_image()
    if img_path:
        detect_logo(img_path)
    else:
        print("❌ Chưa chọn ảnh để test")
