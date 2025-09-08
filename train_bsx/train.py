from ultralytics import YOLO

def train_model():
    model = YOLO("yolov8s.pt")

    # 2. Train model
    results = model.train(
        data="datasets/data.yaml",     # đường dẫn file yaml
        epochs=10,            # số vòng lặp
        batch=16,              # batch size
        imgsz=640,             # kíc    h thước ảnh
        device=0,              # GPU id (hoặc "cpu" nếu không có GPU)
        workers=4,             # số luồng đọc dữ liệu
        patience=30,           # early stopping (nếu val không cải thiện sau 30 epochs thì dừng)
        save=True,             # lưu checkpoint
        project="runs/train",  # nơi lưu kết quả
        name="plate_detect"    # tên folder kết quả
    )
if __name__ == "__main__":
    train_model()