from ultralytics import YOLO

def train_model():
    # Load model
    model = YOLO("yolov8n.pt")

    # Train trên GPU (ép device=0)
    results = model.train(
        data="Datasets/data.yaml",
        epochs=100,
        imgsz=640,
        device=0
    )

if __name__ == "__main__":
    train_model()
 