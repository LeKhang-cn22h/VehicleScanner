import cv2
import os
from ultralytics import YOLO
from tkinter import filedialog, Tk
from mauXe.mauChuDaoXeOto import get_dominant_car_color
from cloudinary_config import upload_image_to_cloudinary


def process_car_image():
    # ========== Chọn ảnh từ thư mục ==========
    root = Tk()
    root.withdraw()  # Ẩn cửa sổ chính
    file_path = filedialog.askopenfilename(
        title="Chọn ảnh xe",
        filetypes=[("Image files", "*.jpg *.png *.jpeg")]
    )
    if not file_path:
        print("Bạn chưa chọn ảnh!")
        return None, None, None

    print("Ảnh đã chọn:", file_path)

    # ========== Upload ảnh gốc lên Cloudinary ==========
    upload_result = upload_image_to_cloudinary(file_path, folder="dauxeoto")
    original_url = upload_result['secure_url']
    print("Link ảnh gốc Cloudinary:", original_url)

    # ========== Load model YOLO ==========
    model = YOLO(r"../logoCar/runs/detect/train5/weights/best.pt")

    # ========== Nhận diện logo ==========
    results = model(file_path)
    img = cv2.imread(file_path)

    crop_urls = []
    for i, r in enumerate(results):
        for box in r.boxes.xyxy:  # xyxy = [x1, y1, x2, y2]
            x1, y1, x2, y2 = map(int, box[:4])
            crop = img[y1:y2, x1:x2]

            crop_name = f"crop_logo_{i}.jpg"
            cv2.imwrite(crop_name, crop)

            # Upload crop logo lên Cloudinary
            crop_upload = upload_image_to_cloudinary(crop_name, folder="dauxeoto")
            crop_url = crop_upload['secure_url']
            crop_urls.append(crop_url)
            print("Link crop logo:", crop_url)

            # Xóa file local crop
            os.remove(crop_name)

    print("Tất cả crop logo đã upload:", crop_urls)

    # ========== Lấy màu chủ đạo của xe ==========
    colors = get_dominant_car_color(file_path)
    print("Màu chủ đạo của xe:", colors)

    # Xóa file gốc local (sau khi upload)
    os.remove(file_path)

    # return cho file khác sử dụng
    return original_url, crop_urls, colors


