import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from firebase_hander import create_time_expired
import requests
import pytz
import threading
import tkinter as tk
import time
import json
import uuid
from google.cloud.firestore import Client
from xeoto.nhanDien import detect_license_plate
from firebase_hander import get_field_from_all_docs
from  xeoto.ketQuaXeOtoVao import process_car_image
import xeoto.utils as utils

FIREBASE_REALTIME_URL = 'https://tramxeuth-default-rtdb.firebaseio.com'
cred = credentials.Certificate("serviceAccountKey.json")
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)
db = firestore.client()


def normalize_plate(plate):
    return plate.replace(".", "").upper() if plate else None


def firebase_put(path, data, include_timestamp=True):
    vn_time = datetime.now(pytz.timezone('Asia/Ho_Chi_Minh'))
    timestamp = vn_time.strftime('%Y-%m-%d %H:%M:%S')
    json_data = {"value": data, "timestamp": timestamp} if not isinstance(data, dict) and include_timestamp else data
    if isinstance(data, dict) and include_timestamp:
        json_data["timestamp"] = timestamp

    url = f"{FIREBASE_REALTIME_URL}/{path}.json"
    response = requests.put(url, json=json_data)
    print(f"[{timestamp}] Ghi {path}: {response.status_code}, {response.text}")


def get_auto():
    try:
        with open("state.json", "r") as f:
            data = json.load(f)
            return data.get("AUTO", False)
    except:
        return False

def on_create(db: Client, biensoxe, url_image_detected, uutien = False):
    time_now = datetime.now().strftime("%H:%M:%S")
    khach_moi = {
        "bienso": biensoxe,
        "image": url_image_detected,
        "timeIn": time_now,
        "timeOut": "",
        "uutien": uutien,
    }
    db.collection("thongtinkhach").add(khach_moi)
    # Ghi Firestore
    today = datetime.today().strftime("%d%m%Y")
    xe_doc_ref = db.collection("lichsuhoatdong").document(today).collection("xeoto").document(biensoxe)
    db.collection("lichsuhoatdong").document(today).set({"ngay": today}, merge=True)

    # Tăng số lần vào
    doc = xe_doc_ref.get()
    solanvao = doc.to_dict().get("solanvao", 0) if doc.exists else 0
    xe_doc_ref.set({"solanvao": solanvao + 1}, merge=True)

    # Lấy dữ liệu từ process_car_image
    link_goc, link_crops, mau, image_dau_xe, bsx_dau, image_logo = process_car_image()

    # Ghi timeline
    time_now = datetime.now().strftime("%H:%M:%S")
    timeline_data = {
        "timein": time_now,
        "biensoxevao": url_image_detected,   # ảnh detect biển số
        "hinhxevao": link_goc,               # ảnh gốc xe upload Cloudinary
        "logovao": link_crops,               # danh sách crop logo
        "logora": None,
        "timeout": None,
        "biensoxera": None,
        "hinhxera": None
    }
    xe_doc_ref.collection("timeline").document("timeline" + str(solanvao)).set(timeline_data)

    # === Cập nhật Realtime Database ===
    firebase_put("trangthaicong", True, include_timestamp=False)
    firebase_put(f"biensotrongbai/{biensoxe}", {
        "trangthai": True,
        "khach": True
    })
    print("Đã ghi dữ liệu thành công.")
    return image_dau_xe, bsx_dau, image_logo

def show_choice_dialog(db, anhxevao, biensoxe: str = ""):
    title = "Cảnh báo!!!"
    dialog = tk.Toplevel()   # dùng Toplevel thay vì Tk
    dialog.title(title)
    dialog.grab_set()  # biến thành modal dialog (chặn event ở window chính cho đến khi đóng)

    frame = tk.Frame(dialog)
    frame.pack(pady=5)

    tk.Label(dialog, text=title, fg="red", font=("Arial", 14, "bold")).pack(pady=10)
    tk.Label(frame, text="Biển số ", font=("Arial", 12)).pack(side="left")
    tk.Label(frame, text=biensoxe, fg="blue", font=("Arial", 12, "bold")).pack(side="left")
    tk.Label(frame, text=" không có trong danh sách. Bạn có muốn:", font=("Arial", 12)).pack(side="left")

    result = [None, None, None]  # dùng dict để mutable, lưu kết quả từ callback

    def handle_create(uutien: bool):
        result[0], result[1], result[2] = on_create(db, biensoxe, anhxevao, uutien)
        dialog.destroy()
    
    tk.Button(
        dialog,
        text="Tạo xe khách",
        command=lambda: handle_create(True),
        width=20
    ).pack(pady=5)

    tk.Button(
        dialog,
        text="Tạo xe phụ",
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
    return tuple(result)

def run_license_scan(label_status, label_bsx):
    while True:
        bien_so, url_image_detected, img_path, best_plate = detect_license_plate()
        bien_so_quet = normalize_plate(bien_so)
        utils.update_label_content(label_bsx, bien_so_quet, bg="green")
        utils.update_label_content(label_status, "Biển số quét được: "+ bien_so_quet)
        print("Biển số quét được:", bien_so_quet)

        # Kiểm tra hợp lệ
        ds_bien_so_raw = get_field_from_all_docs("thongtindangky", "biensoxe")
        ds_map_bien_so_phu_raw = get_field_from_all_docs("thongtindangky", "biensophu")
        ds_bien_so_phu_raw = [item["bienSo"] for item in ds_map_bien_so_phu_raw if item and "bienSo" in item]
        ds_bien_so_khach_raw = get_field_from_all_docs("thongtinkhach", "bienso") + get_field_from_all_docs("thongtinkhach", "biensoxe")

        ds_bien_so = [normalize_plate(val) for val in ds_bien_so_raw if val]
        ds_bien_so_phu = [normalize_plate(val) for val in ds_bien_so_phu_raw if val]
        ds_bien_so_khach = [normalize_plate(val) for val in ds_bien_so_khach_raw if val]

        hop_le = bien_so_quet in ds_bien_so
        hop_le_phu = bien_so_quet in ds_bien_so_phu
        hop_le_khach = bien_so_quet in ds_bien_so_khach

        if hop_le or hop_le_phu or hop_le_khach:
            utils.update_label_content(label_status, f"Biển số {bien_so_quet} hợp lệ", bg="green")

            # Ghi Firestore
            today = datetime.today().strftime("%d%m%Y")
            xe_doc_ref = db.collection("lichsuhoatdong").document(today).collection("xeoto").document(bien_so_quet)
            db.collection("lichsuhoatdong").document(today).set({"ngay": today}, merge=True)

            # Tăng số lần vào
            doc = xe_doc_ref.get()
            solanvao = doc.to_dict().get("solanvao", 0) if doc.exists else 0
            xe_doc_ref.set({"solanvao": solanvao + 1}, merge=True)

            # Lấy dữ liệu từ process_car_image
            link_goc, link_crops, mau, image_dau_xe, bsx_dau, image_logo = process_car_image()

            # Ghi timeline
            time_now = datetime.now().strftime("%H:%M:%S")
            timeline_data = {
                "timein": time_now,
                "biensoxevao": url_image_detected,   # ảnh detect biển số
                "hinhxevao": link_goc,               # ảnh gốc xe upload Cloudinary
                "logovao": link_crops,               # danh sách crop logo
                "logora": None,
                "timeout": None,
                "biensoxera": None,
                "hinhxera": None
            }
            xe_doc_ref.collection("timeline").document("timeline" + str(solanvao)).set(timeline_data)

            # Realtime DB
            firebase_put("trangthaicong", True, include_timestamp=False)
            if hop_le:
                firebase_put(f"biensotrongbai/{bien_so_quet}", {"trangthai": True, "canhbao": False})
            else:
                datetime_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                firebase_put(f"biensotrongbai/{bien_so_quet}", {
                    "trangthai": True,
                    "canhbao": False,
                    "timestamp": datetime_str,
                    "timeExpired": create_time_expired(datetime_str)
                })

            # --- Nếu muốn tự động tắt GUI sau khi hoàn tất ---
            time.sleep(1)  # delay để thấy thông báo
            # root_window.quit()
            break  # thoát vòng lặp

        else:
            utils.update_label_content(label_status, f"Biển số {bien_so_quet} không hợp lệ", bg="red")
            firebase_put("trangthaicong", False, include_timestamp=False)
            image_dau_xe = None
            bsx_dau = None
            image_logo = None
            print("Biển số không hợp lệ.")
            auto = get_auto()
            if auto:
                image_dau_xe, bsx_dau, image_logo = on_create(db, bien_so_quet, url_image_detected, True)
            else:
                image_dau_xe, bsx_dau, image_logo = show_choice_dialog(db, url_image_detected, bien_so_quet)
                utils.update_label_content(label_status, f"Biển số {bien_so_quet} đăng ký thành công", bg="green")
            break
    label_status.update()
    time.sleep(1)
    print("Kết thúc")
    return bien_so_quet, img_path, best_plate, image_dau_xe, bsx_dau, image_logo

# # =======================
# # GUI Tkinter
# # =======================
# root = tk.Tk()
# root.title("Hệ thống quản lý xe tự động")
# root.geometry("700x200")

# label_status = tk.Label(root, text="Đang chờ quét xe...", font=("Arial", 24), width=50, height=5, bg="gray")
# label_status.pack(pady=20)

# # Thread chạy quét biển số
# threading.Thread(target=run_license_scan, args=(label_status, root), daemon=True).start()

# root.mainloop()
