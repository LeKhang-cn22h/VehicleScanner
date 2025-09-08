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
import utils
import requests
import numpy as np
# Hàm kiểm tra và cập nhật số lượt khi ra
def update_soluot_khira(bien_so_xe):
    db = firestore.client()
    doc_col = db.collection("thongtindangky")
    xe_doc_ref = doc_col.get()
    for xe_doc in xe_doc_ref:
        xe_data = xe_doc.to_dict()
        ds_bien_so_xe = [xe_doc['biensoxe'], xe_doc['biensophu']['bienSo']]
        if bien_so_xe in ds_bien_so_xe:
            so_luot_moi = xe_data['luot'] - 1
            if so_luot_moi >= 0:
                update_xe_doc_ref = doc_col.document(xe_doc.id)
                update_xe_doc_ref.update({"luot":so_luot_moi})
                print("Số lượt hợp lệ. Cho xe ra")
                return True
            else:
                print("Số lượt ko đủ. Bạn cần mua thêm lượt")
                return False
    return False

def is_khach_uutien(bien_so_xe):
    db = firestore.client()
    doc_col = db.collection("thongtinkhach")
    xe_doc_ref = doc_col.get()
    for xe_doc in xe_doc_ref:
        xe_data = xe_doc.to_dict()
        if bien_so_xe == xe_data.get('bienso') or bien_so_xe == xe_data.get('biensoxe'):
            if xe_data['uutien']:
                return True
            else:
                return False
    return False

def btn_xet_lai(bien_so_quet, label_status):
    # Kiểm tra số lượt có hợp lệ ko
    is_hople = update_soluot_khira(bien_so_quet)
    if not is_hople:
        label_status.config(text=f"Số lượt ra còn lại của biển số {bien_so_quet} không đủ ❌", bg="yellow")
        label_status.update()
        return False
    return True

def btn_tien_mat():
    title = "Nhắc nhở!!!"
    dialog = tk.Toplevel()   # dùng Toplevel thay vì Tk
    dialog.title(title)
    dialog.grab_set()  # biến thành modal dialog (chặn event ở window chính cho đến khi đóng)

    frame = tk.Frame(dialog)
    frame.pack(pady=5)

    tk.Label(dialog, text=title, fg="yellow", font=("Arial", 20, "bold")).pack(pady=10)
    tk.Label(frame, text="Bạn nhận được tiền chưa?", font=("Arial", 17)).pack(side="left")

    result = [False]  # dùng dict để mutable, lưu kết quả từ callback

    def handle(is_: bool):
        result[0] = is_
        dialog.destroy()

    tk.Button(
        dialog,
        text="Đã nhận được",
        command=lambda: handle(True),
        width=20
    ).pack(pady=5)

    tk.Button(
        dialog,
        text="Không có tiền mặt",
        command=lambda: handle(False),
        width=20
    ).pack(pady=5)

    dialog.wait_window()
    return result[0]

def btn_tu_choi():
    return False

def show_choice_dialog(label_status, biensoxe: str = ""):
    title = "Cảnh báo!!!"
    dialog = tk.Toplevel()   # dùng Toplevel thay vì Tk
    dialog.title(title)
    dialog.grab_set()  # biến thành modal dialog (chặn event ở window chính cho đến khi đóng)

    frame = tk.Frame(dialog)
    frame.pack(pady=5)

    tk.Label(dialog, text=title, fg="red", font=("Arial", 20, "bold")).pack(pady=10)
    tk.Label(frame, text="Biển số ", font=("Arial", 17)).pack(side="left")
    tk.Label(frame, text=biensoxe, fg="blue", font=("Arial", 17, "bold")).pack(side="left")
    tk.Label(frame, text=" không còn đủ lượt ra. Bạn có muốn:", font=("Arial", 17)).pack(side="left")

    result = [False]  # dùng dict để mutable, lưu kết quả từ callback

    def handle_create(biensoxe, label_status, btn):
        actions = {
            1: btn_xet_lai,
            2: lambda: btn_tien_mat(biensoxe, label_status),
            3: btn_tu_choi
        }
        # Lấy hàm phù hợp (nếu không có thì mặc định btn_tu_choi)
        action = actions.get(btn, btn_tu_choi)
        # Gọi hàm
        result[0] = action()
        dialog.destroy()
    
    tk.Button(
        dialog,
        text="Xét lại",
        command=lambda: handle_create(True),
        width=20
    ).pack(pady=5)

    tk.Button(
        dialog,
        text="Tiền mặt",
        command=lambda: handle_create(False),
        width=20
    ).pack(pady=5)

    tk.Button(
        dialog,
        text="Từ chối",
        command=lambda:  dialog.destroy(),
        width=20
    ).pack(pady=5)

    dialog.wait_window()
    return result[0]

# =====================
# Hàm xử lý quét biển số
# =====================
def run_license_scan(label_status, root, label_bsx):
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
    def load_image(img_path):
        """Đọc ảnh từ local path, URL (http/https) hoặc numpy array."""
        # Nếu truyền vào là numpy array (ảnh OpenCV)
        if isinstance(img_path, np.ndarray):
            img_cv = img_path
            img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
            return img_cv, img_pil

        # Nếu là string (đường dẫn hoặc URL)
        if isinstance(img_path, str):
            if img_path.startswith(("http://", "https://")):  # Link URL
                resp = requests.get(img_path, stream=True).content
                img_array = np.asarray(bytearray(resp), dtype=np.uint8)
                img_cv = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                img_pil = Image.open(requests.get(img_path, stream=True).raw).convert("RGB")
                return img_cv, img_pil
            else:  # File local
                img_cv = cv2.imread(img_path)
                img_pil = Image.open(img_path).convert("RGB")
                return img_cv, img_pil

        raise TypeError(f"Không hỗ trợ kiểu dữ liệu {type(img_path)} trong load_image")

    def crop_car_largest(img_path):
        """Crop xe lớn nhất từ ảnh (nếu không detect được thì trả ảnh gốc)."""
        results = yolo_model(img_path, verbose=False)
        max_area = 0
        best_crop = None

        # Đọc ảnh trước để tránh đọc nhiều lần
        img_cv, img_pil = load_image(img_path)

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if yolo_model.names[cls] == "car":
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    area = (x2 - x1) * (y2 - y1)
                    if area > max_area:
                        crop = img_cv[y1:y2, x1:x2]
                        if crop is not None and crop.size > 0:
                            best_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                            max_area = area

        if best_crop is not None:
            return best_crop
        else:
            return img_pil

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

        # 1. Quét biển số đuôi xe
        bien_so, url_image_detected, img_path, best_plate = detect_license_plate()
        if not bien_so:
            label_status.config(text="Không quét được biển số ", bg="red")
            utils.update_label_content(label_bsx, "Không quét được biển số", bg="red")
            utils.update_label_content(label_status, "Không quét được biển số", bg="red")
            time.sleep(2)
            continue

        bien_so_quet = bien_so.replace(".", "").upper()
        utils.update_label_content(label_bsx, bien_so_quet, bg="green")
        utils.update_label_content(label_status, "Biển số quét được: "+ bien_so_quet)
        print("Biển số quét được:", bien_so_quet)

        # 2. Kiểm tra hợp lệ với Firebase (biển số từ detect_license_plate)
        ds_bien_so = firebase_service.get_all_license_plates()
        if bien_so_quet not in ds_bien_so:
            utils.update_label_content(label_status, f"Biển số {bien_so_quet} không có trong bãi xe", bg="red")
            time.sleep(1)
            continue

        # 3. Lấy dữ liệu biển số từ Firebase
        bien_so_data = firebase_service.get_license_plate_data(bien_so_quet)
        if not bien_so_data:
            utils.update_label_content(label_status, f"Không lấy được dữ liệu {bien_so_quet} ", bg="red")
            time.sleep(1)
            continue

        # Sau khi qua bước trên mới tới process_car_image
        link_goc, link_crops, mau, image_dau_xe, bsx_dau, image_logo = process_car_image()
        # if not plate_text:
        #     label_status.config(text="Không đọc được biển số từ process_car_image", bg="red")
        #     label_status.update()
        #     time.sleep(2)
        #     continue

        # plate_text = plate_text.replace(".", "").upper()
        # print("Biển số từ process_car_image:", plate_text)

        # 4. Kiểm tra hợp lệ với Firebase (biển số từ process_car_image)
        # if plate_text not in ds_bien_so:
        #     label_status.config(text=f"Biển số {plate_text} không hợp lệ ", bg="red")
        #     label_status.update()
        #     time.sleep(2)
        #     continue

        # 5. Lấy dữ liệu từ Firebase theo plate_text
        # bien_so_data2 = firebase_service.get_license_plate_data(plate_text)
        # if not bien_so_data2:
        #     label_status.config(text=f"Không lấy được dữ liệu {plate_text} ", bg="red")
        #     label_status.update()
        #     time.sleep(2)
        #     continue

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
                hinhdauxevao = timeline_data.get("hinhxevao")
                logovao = timeline_data.get("logovao")
                logovao = logovao[0]
                hinhduoixevao = timeline_data.get("biensoxevao")

                print("Hình đầu xe vào:", hinhdauxevao)
                print("Hình đuôi xe vào:", hinhduoixevao)
                print("Lô gô vào:", logovao)
        else:
            timeline_doc_id = None
            timeline_ref = None
            hinhdauxevao, logovao, hinhduoixevao = None, None, None
        bsx_dau_vao = None
        bsx_duoi_vao = None
        _,_,_,bsx_dau_vao = detect_license_plate(hinhdauxevao)
        _,_,_,bsx_duoi_vao = detect_license_plate(hinhduoixevao)

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
        if link_goc and hinhdauxevao:
            same_car, distance = compare_images(link_goc, hinhdauxevao, model_path="siamese_model.pth",
                                        threshold=0.5)
            if same_car:
                print("✅ Ảnh cùng xe")
            else:
                    print("❌ Ảnh khác xe")
        else:
                print("Không có đủ ảnh để so sánh")

        if logovao is not None and image_logo is not None:
            same_logo, distance_logo = compare_images(logovao, image_logo, model_path="siamese_model.pth",
                                                        threshold=0.5)
            if same_logo:
                print("✅ Logo cùng xe")
            else:
                print("❌ Logo khác xe")
        else:
            print("Không có đủ ảnh logo để so sánh")
        #logovao

        # Kiểm tra có phải khách ưu tiên ko
        is_khach_uu_tien = is_khach_uutien(bien_so_quet)
        if not is_khach_uu_tien:
            # 5. Cập nhật trạng thái xe
            label_status.config(text=f"Biển số {bien_so_quet} hợp lệ ", bg="green")

            trangthai = bien_so_data.get('trangthai')
            if trangthai is True:
                # Nếu trạng thái True, cảnh báo
                firebase_service.update_canhbao(bien_so_quet, True)
                print("Xe đã ra trước đó, đã gửi cảnh báo.")
                print("Biển số có 'trangthai' = False.")
                break

            # Kiểm tra số lượt có hợp lệ ko
            is_hople = update_soluot_khira(bien_so_quet)
            if not is_hople:
                label_status.config(text=f"Số lượt ra còn lại của biển số {bien_so_quet} không đủ ❌", bg="yellow")
                label_status.update()
                time.sleep(1)

                break
        else:
            trangthai = True
            is_hople = True
        if same_car and same_logo and trangthai and is_hople:
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
            time.sleep(2)  # delay để người dùng thấy thông báo
            break
    data_xe_vao = utils.DataXeVao(
        hinh_dau_xe=hinhdauxevao,
        hinh_duoi_xe=hinhduoixevao,
        bsx_dau=bsx_dau_vao,
        bsx_duoi=bsx_duoi_vao,
        logo=logovao
    )
    sameImage = utils.SameImage(
        same_car=same_car,
        same_logo=same_logo
    )
    return bien_so_quet, img_path, best_plate, image_dau_xe, bsx_dau, image_logo, data_xe_vao, sameImage

# # =====================
# # GUI Tkinter
# # =====================
# root = tk.Tk()
# root.title("Hệ thống quản lý xe tự động")
# root.geometry("700x200")

# label_status = tk.Label(root, text="Đang chờ quét xe...", font=("Arial", 18), width=60, height=2, bg="gray")
# label_status.pack(pady=40)

# # Chạy quét biển số trong thread riêng
# threading.Thread(target=run_license_scan, args=(label_status, root), daemon=True).start()

# root.mainloop()
