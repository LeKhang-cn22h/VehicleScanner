import tkinter as tk
import threading
import time
from datetime import datetime
from nhanDien import detect_license_plate
from firebase_service import FirebaseService
from firebase_admin import firestore
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights
import cv2
from ultralytics import YOLO
from  ketQuaXeOtoRa import process_car_image
import torch.nn as nn
# =====================
# Hàm xử lý quét biển số
# =====================
def run_license_scan(label_status, root):
    firebase_service = FirebaseService()
    db = firestore.client()


    #khởi tạo model siamese

    import torch


    # === Transform y như khi train ===
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # === Load YOLO để crop xe ===
    yolo_model = YOLO("yolov8n.pt")

    def crop_car_largest(img_path):
        """Crop xe lớn nhất từ ảnh (nếu không detect được thì trả ảnh gốc)."""
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

    # === Mạng Siamese giống lúc train ===


    class SiameseNetwork(nn.Module):
        def __init__(self):
            super(SiameseNetwork, self).__init__()
            self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 128)

        def forward_once(self, x):
            return self.backbone(x)

        def forward(self, x1, x2):
            return self.forward_once(x1), self.forward_once(x2)
    while True:

        # 1. Quét biển số
        bien_so, url_image_detected = detect_license_plate()
        if not bien_so:
            label_status.config(text="Không quét được biển số ", bg="red")
            label_status.update()
            time.sleep(2)
            continue

        bien_so_quet = bien_so.replace(".", "").upper()
        print("Biển số quét được:", bien_so_quet)

        # 2. Kiểm tra hợp lệ với Firebase (biển số từ detect_license_plate)
        ds_bien_so = firebase_service.get_all_license_plates()
        print("Biển số quét được:")

        if bien_so_quet not in ds_bien_so:
            label_status.config(text=f"Biển số {bien_so_quet} không hợp lệ ", bg="red")
            label_status.update()
            time.sleep(2)
            continue

        # 3. Lấy dữ liệu biển số từ Firebase
        bien_so_data = firebase_service.get_license_plate_data(bien_so_quet)
        if not bien_so_data:
            label_status.config(text=f"Không lấy được dữ liệu {bien_so_quet} ", bg="red")
            label_status.update()
            time.sleep(2)
            continue

        # Sau khi qua bước trên mới tới process_car_image
        link_goc, link_crops, plate_text = process_car_image()
        if not plate_text:
            label_status.config(text="Không đọc được biển số từ process_car_image", bg="red")
            label_status.update()
            time.sleep(2)
            continue

        plate_text = plate_text.replace(".", "").upper()
        print("Biển số từ process_car_image:", plate_text)

        # 4. Kiểm tra hợp lệ với Firebase (biển số từ process_car_image)
        if plate_text not in ds_bien_so:
            label_status.config(text=f"Biển số {plate_text} không hợp lệ ", bg="red")
            label_status.update()
            time.sleep(2)
            continue

        # 5. Lấy dữ liệu từ Firebase theo plate_text
        bien_so_data2 = firebase_service.get_license_plate_data(plate_text)
        if not bien_so_data2:
            label_status.config(text=f"Không lấy được dữ liệu {plate_text} ", bg="red")
            label_status.update()
            time.sleep(2)
            continue

        # 4. Lấy timeline gần nhất
        from datetime import datetime

        today = datetime.today().strftime("%d%m%Y")
        xe_doc_ref = db.collection("lichsuhoatdong").document(today).collection("xeoto").document(bien_so_quet)

        # Lấy tất cả document trong timeline
        timeline_docs = xe_doc_ref.collection("timeline").list_documents()
        max_index = -1
        for tdoc in timeline_docs:
            name = tdoc.id
            if name.startswith("timeline"):
                try:
                    index = int(name.replace("timeline", ""))
                    if index > max_index:
                        max_index = index
                except ValueError:
                    continue

        timeline_data = None
        if max_index >= 0:
            timeline_doc_id = f"timeline{max_index}"
            timeline_ref = xe_doc_ref.collection("timeline").document(timeline_doc_id)

            # 🔹 Lấy dữ liệu từ timeline gần nhất
            doc_snapshot = timeline_ref.get()
            if doc_snapshot.exists:
                timeline_data = doc_snapshot.to_dict()
                hinhxevao = timeline_data.get("hinhxevao")
                logovao = timeline_data.get("logovao")

                print("Hình xe vào:", hinhxevao)
                print("Logo vào:", logovao)
        else:
            timeline_doc_id = None
            timeline_ref = None
            hinhxevao, logovao = None, None

        #hinhxevao
        #link_goc
        # === Hàm so sánh ảnh ===
        def compare_images(img_path1, img_path2, model_path="siamese_model.pth", threshold=0.5):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Load model
            model = SiameseNetwork().to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()

            # Crop + transform ảnh 1
            img1 = crop_car_largest(img_path1)
            img1 = transform(img1).unsqueeze(0).to(device)

            # Crop + transform ảnh 2
            img2 = crop_car_largest(img_path2)
            img2 = transform(img2).unsqueeze(0).to(device)

            # Forward
            with torch.no_grad():
                out1, out2 = model(img1, img2)
                distance = torch.nn.functional.pairwise_distance(out1, out2).item()

            print(f"Khoảng cách giữa 2 ảnh: {distance:.4f}")
            if distance < threshold:
                print("Cùng xe")
                return True, distance
            else:
                print("Khác xe")
                return False,
                # ================================
                # So sánh link_goc và hinhxevao
                # ================================
            if link_goc and hinhxevao:
                same_car, distance = compare_images(link_goc, hinhxevao, model_path="siamese_model.pth",
                                            threshold=0.5)
                if same_car:
                    print("✅ Ảnh cùng xe")
                else:
                        print("❌ Ảnh khác xe")
            else:
                    print("Không có đủ ảnh để so sánh")

            if logovao and link_crops:
                same_logo, distance_logo = compare_images(logovao, link_crops, model_path="siamese_model.pth",
                                                          threshold=0.5)
                if same_logo:
                    print("✅ Logo cùng xe")
                else:
                    print("❌ Logo khác xe")
            else:
                print("Không có đủ ảnh logo để so sánh")
        #logovao

        # 5. Cập nhật trạng thái xe
        label_status.config(text=f"Biển số {bien_so_quet} hợp lệ ", bg="green")

        trangthai = bien_so_data.get('trangthai')
        if trangthai is False:
            print("Biển số có 'trangthai' = False.")
            firebase_service.update_license_plate_field(bien_so_quet, True)
            firebase_service.delete_license_plate(bien_so_quet)

            # Lấy document xe
            doc = xe_doc_ref.get()
            if doc.exists:
                data = doc.to_dict()
                solanra = data.get("solanra", 0)
            else:
                solanra = 0

            solanra += 1
            xe_doc_ref.set({"solanra": solanra}, merge=True)

            # Thời gian hiện tại
            time_now = datetime.now().strftime("%H:%M:%S")

            # Ghi vào timeline gần nhất
            if timeline_ref:
                timeline_ref.set({
                    "timeout": time_now,
                    "biensoxera": url_image_detected
                }, merge=True)
                print(f"Đã cập nhật timeline {timeline_doc_id}")
            else:
                print("Không tìm thấy timeline để cập nhật.")
        else:
            # Nếu trạng thái True, cảnh báo
            firebase_service.update_canhbao(bien_so_quet, True)
            print("Xe đã ra trước đó, đã gửi cảnh báo.")

        time.sleep(2)  # delay để người dùng thấy thông báo
        root.quit()
        break
# =====================
# GUI Tkinter
# =====================
root = tk.Tk()
root.title("Hệ thống quản lý xe tự động")
root.geometry("700x200")

label_status = tk.Label(root, text="Đang chờ quét xe...", font=("Arial", 18), width=60, height=2, bg="gray")
label_status.pack(pady=40)

# Chạy quét biển số trong thread riêng
threading.Thread(target=run_license_scan, args=(label_status, root), daemon=True).start()

root.mainloop()
