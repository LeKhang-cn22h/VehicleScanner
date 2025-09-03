import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
from firebase_hander import create_time_expired
import requests
import pytz
import threading
import tkinter as tk
import time

from nhanDien import detect_license_plate
from firebase_hander import get_field_from_all_docs
from  ketQuaXeOto import process_car_image

FIREBASE_REALTIME_URL = 'https://tramxeuth-default-rtdb.firebaseio.com'
cred = credentials.Certificate("../serviceAccountKey.json")
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


def run_license_scan(label_status, root_window):
    while True:
        bien_so, url_image_detected = detect_license_plate()
        bien_so_quet = normalize_plate(bien_so)
        print("Biển số quét được:", bien_so_quet)

        # Kiểm tra hợp lệ
        ds_bien_so_raw = get_field_from_all_docs("thongtindangky", "biensoxe")
        ds_map_bien_so_phu_raw = get_field_from_all_docs("thongtindangky", "biensophu")
        ds_bien_so_phu_raw = [item["bienSo"] for item in ds_map_bien_so_phu_raw if item and "bienSo" in item]

        ds_bien_so = [normalize_plate(val) for val in ds_bien_so_raw if val]
        ds_bien_so_phu = [normalize_plate(val) for val in ds_bien_so_phu_raw if val]

        hop_le = bien_so_quet in ds_bien_so
        hop_le_phu = bien_so_quet in ds_bien_so_phu

        if hop_le or hop_le_phu:
            label_status.config(text=f"Biển số {bien_so_quet} hợp lệ ✅", bg="green")

            # Ghi Firestore
            today = datetime.today().strftime("%d%m%Y")
            xe_doc_ref = db.collection("lichsuhoatdong").document(today).collection("xeoto").document(bien_so_quet)
            db.collection("lichsuhoatdong").document(today).set({"ngay": today}, merge=True)

            # Tăng số lần vào
            doc = xe_doc_ref.get()
            solanvao = doc.to_dict().get("solanvao", 0) if doc.exists else 0
            xe_doc_ref.set({"solanvao": solanvao + 1}, merge=True)

            # Lấy dữ liệu từ process_car_image
            link_goc, link_crops, mau = process_car_image()

            # Ghi timeline
            time_now = datetime.now().strftime("%H:%M:%S")
            timeline_data = {
                "timein": time_now,
                "biensoxevao": url_image_detected,   # ảnh detect biển số
                "hinhxevao": link_goc,               # ảnh gốc xe upload Cloudinary
                "logovao": link_crops,               # danh sách crop logo
                "mauxechudaovao": mau,               # màu chủ đạo xe
                "mauxechudaora": None,
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
            root_window.quit()
            break  # thoát vòng lặp

        else:
            label_status.config(text=f"Biển số {bien_so_quet} không hợp lệ ❌", bg="red")
            firebase_put("trangthaicong", False, include_timestamp=False)

        label_status.update()
        time.sleep(1)


# =======================
# GUI Tkinter
# =======================
root = tk.Tk()
root.title("Hệ thống quản lý xe tự động")
root.geometry("700x200")

label_status = tk.Label(root, text="Đang chờ quét xe...", font=("Arial", 24), width=50, height=5, bg="gray")
label_status.pack(pady=20)

# Thread chạy quét biển số
threading.Thread(target=run_license_scan, args=(label_status, root), daemon=True).start()

root.mainloop()
