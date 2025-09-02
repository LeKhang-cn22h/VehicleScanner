import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
from firebase_hander import create_time_expired
import requests
import uuid
import pytz
import json

import tkinter as tk
from nhan_dien_vao import detect_license_plate
from firebase_hander import get_field_from_all_docs
from FrontCarPhoto import capture_and_upload_front_image

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

def on_create(db, biensoxe, anhxevao, uutien = False):
    time_now = datetime.now().strftime("%H:%M:%S")
    khach_moi = {
        "bienso": biensoxe,
        "image": anhxevao,
        "timeIn": time_now,
        "timeOut": "",
        "uutien": uutien,
    }
    db.collection("thongtinkhach").add(khach_moi)
    # === Bước 3: Ghi Firestore nếu hợp lệ ===
    today = datetime.today().strftime("%d%m%Y")
    xe_doc_ref = db.collection("lichsuhoatdong").document(today).collection("xe").document(biensoxe)
    ghithat=db.collection("lichsuhoatdong").document(today).set({"ngay": today})
    # Tăng số lần vào
    doc = xe_doc_ref.get()
    solanvao = doc.to_dict().get("solanvao", 0) if doc.exists else 0
    xe_doc_ref.set({"solanvao": solanvao + 1}, merge=True)

    # Chụp ảnh đầu xe và upload lên Cloudinary
    image_url_vao = capture_and_upload_front_image()

    # === Ghi vào timeline ===
    time_now = datetime.now().strftime("%H:%M:%S")
    timeline_data = {
        "timeIn": time_now,
        "imageIn": anhxevao,
        "hinhdauxevao": image_url_vao,
        "timeOut": None,
        "imageOut": None,
        "hinhdauxera":None
    }
    # if image_url_vao:
    #     timeline_data["hinhdauxevao"] = image_url_vao

    # Document 1: ID random
    timeline_id = str(uuid.uuid4())[:16]
    # xe_doc_ref.collection("timeline").document(timeline_id).set(timeline_data)

    # Document 2: Tên cố định 'xevao'
    xe_doc_ref.collection("timeline").document("timeline"+str(solanvao)).set(timeline_data)

    # === Cập nhật Realtime Database ===
    firebase_put("trangthaicong", True, include_timestamp=False)
    firebase_put(f"biensotrongbai/{biensoxe}", {
        "trangthai": True,
        "khach": True
    })
    print("Đã ghi dữ liệu thành công.")

def show_choice_dialog(db, anhxevao, biensoxe: str = ""):
    title = "Cảnh báo!!!"
    # message = f"Biển số {biensoxe} không có trong danh sách. Bạn có muốn: "
    dialog = tk.Tk()
    dialog.title(title)
    frame = tk.Frame(dialog)

    tk.Label(dialog, text=title, fg="red", font=("Arial", 14, "bold")).pack(pady=10)
    tk.Label(frame, text="Biển số ", font=("Arial", 12)).pack(side="left")
    tk.Label(frame, text=biensoxe, fg="blue", font=("Arial", 12, "bold")).pack(side="left")
    tk.Label(frame, text=" không có trong danh sách. Bạn có muốn:", font=("Arial", 12)).pack(side="left")


    tk.Button(
        dialog,
        text="Tạo xe khách",
        command=lambda: (on_create(db, biensoxe, anhxevao, True) if on_create else None, dialog.destroy()),
        width=20
    ).pack(pady=5)

    tk.Button(
        dialog,
        text="Tạo xe phụ",
        command=lambda: (on_create(db, biensoxe, anhxevao, False) if on_create else None, dialog.destroy()),
        width=20
    ).pack(pady=5)

    tk.Button(
        dialog,
        text="Từ chối",
        command=lambda:  dialog.destroy(),
        width=20
    ).pack(pady=5)

    dialog.mainloop()

def get_auto():
    try:
        with open("state.json", "r") as f:
            data = json.load(f)
            return data.get("AUTO", False)
    except:
        return False
# === Bước 1: Quét biển số xe ===
bien_so, url_image_detected = detect_license_plate()
bien_so_quet = normalize_plate(bien_so)

print("Biển số quét được:", bien_so_quet)


# === Bước 2: Kiểm tra hợp lệ ===
ds_bien_so_raw = get_field_from_all_docs("thongtindangky", "biensoxe")
ds_map_bien_so_phu_raw = get_field_from_all_docs("thongtindangky", "biensophu")
ds_bien_so_khach_raw = get_field_from_all_docs("thongtinkhach", "bienso")
if ds_map_bien_so_phu_raw:
    ds_bien_so_phu_raw = [item["bienSo"] for item in ds_map_bien_so_phu_raw if "bienSo" in item]
else:
    ds_bien_so_phu_raw = []

ds_bien_so = [normalize_plate(val) for val in ds_bien_so_raw if val]
ds_bien_so_phu = [normalize_plate(val) for val in ds_bien_so_phu_raw if val]
ds_bien_so_khach = [normalize_plate(val) for val in ds_bien_so_khach_raw if val]

hop_le = bien_so_quet in ds_bien_so
hop_le_phu = bien_so_quet in ds_bien_so_phu
hop_le_khach = bien_so_quet in ds_bien_so_khach

print(" Danh sách hợp lệ:", ds_bien_so, ds_bien_so_khach, ds_bien_so_phu)

if not (hop_le or hop_le_phu or hop_le_khach):
    firebase_put("trangthaicong", False, include_timestamp=False)
    #Xuất hiện dialog cho lựa chọn
    print("Biển số không hợp lệ.")
    auto = get_auto()
    if auto:
        on_create(db, bien_so_quet, url_image_detected, True)
    else:
        show_choice_dialog(db, url_image_detected, bien_so_quet)
    exit()

# === Bước 3: Ghi Firestore nếu hợp lệ ===
today = datetime.today().strftime("%d%m%Y")
xe_doc_ref = db.collection("lichsuhoatdong").document(today).collection("xe").document(bien_so_quet)
ghithat=db.collection("lichsuhoatdong").document(today).set({"ngay": today})
# Tăng số lần vào
doc = xe_doc_ref.get()
solanvao = doc.to_dict().get("solanvao", 0) if doc.exists else 0
xe_doc_ref.set({"solanvao": solanvao + 1}, merge=True)

# Chụp ảnh đầu xe và upload lên Cloudinary
image_url_vao = capture_and_upload_front_image()

# === Ghi vào timeline ===
time_now = datetime.now().strftime("%H:%M:%S")
timeline_data = {
    "timeIn": time_now,
    "imageIn": url_image_detected,
    "hinhdauxevao": image_url_vao,
    "timeOut": None,
    "imageOut": None,
    "hinhdauxera":None
}
# if image_url_vao:
#     timeline_data["hinhdauxevao"] = image_url_vao

# Document 1: ID random
timeline_id = str(uuid.uuid4())[:16]
# xe_doc_ref.collection("timeline").document(timeline_id).set(timeline_data)

# Document 2: Tên cố định 'xevao'
xe_doc_ref.collection("timeline").document("timeline"+str(solanvao)).set(timeline_data)

# === Cập nhật Realtime Database ===
firebase_put("trangthaicong", True, include_timestamp=False)
if hop_le:
    firebase_put(f"biensotrongbai/{bien_so_quet}", {
        "trangthai": True,
        "canhbao": False
    })
else:
    datetime_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    firebase_put(f"biensotrongbai/{bien_so_quet}", {
        "trangthai": True,
        "canhbao": False,
        "timestamp": datetime_str,
        "timeExpired": create_time_expired(datetime_str),
    })
print("Đã ghi dữ liệu thành công.")