import cv2
from ultralytics import YOLO

# Load model đã train
model = YOLO("runs/detect/train/weights/best.pt")
# model = YOLO("runs/car_plate2/weights/best.pt")

# Mở camera (0 = mặc định, nếu không chạy thì thử 1,2,...)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Không mở được camera. Hãy thử đổi source thành 1 hoặc 2.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Không đọc được frame từ camera")
        break

    # Chạy nhận diện với YOLO
    results = model(frame)

    # Vẽ kết quả trực tiếp lên frame
    annotated_frame = results[0].plot()

    # Hiển thị
    cv2.imshow("YOLOv8 Camera Demo", annotated_frame)

    # Bấm 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
