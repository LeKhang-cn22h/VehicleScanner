import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
from firebase_hander import create_time_expired
import requests
import uuid
import pytz
import os
from dotenv import load_dotenv

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

# === Bước 1: Quét biển số xe ===
bien_so, url_image_detected = detect_license_plate()
bien_so_quet = normalize_plate(bien_so)

print("Biển số quét được:", bien_so_quet)

# === Bước 2: Kiểm tra hợp lệ ===
ds_bien_so_raw = get_field_from_all_docs("thongtindangky", "biensoxe")
ds_map_bien_so_phu_raw = get_field_from_all_docs("thongtindangky", "biensophu")
ds_bien_so_phu_raw = [item["bienSo"] for item in ds_map_bien_so_phu_raw if "bienSo" in item]
ds_bien_so = [normalize_plate(val) for val in ds_bien_so_raw if val]
ds_bien_so_phu = [normalize_plate(val) for val in ds_bien_so_phu_raw if val]

hop_le = bien_so_quet in ds_bien_so
hop_le_phu = bien_so_quet in ds_bien_so_phu
print(" Danh sách hợp lệ:", ds_bien_so)

if not (hop_le or hop_le_phu):
    firebase_put("trangthaicong", False, include_timestamp=False)
    print("Biển số không hợp lệ.")
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