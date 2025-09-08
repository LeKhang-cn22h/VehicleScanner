import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
from firebase_hander import create_time_expired
import pytz
import tkinter as tk
import time

from nhanDien import detect_license_plate
from firebase_hander import get_field_from_all_docs
from face_detection.train_face import capture_face_and_upload
from tkinter import messagebox
from PIL import Image, ImageTk
from io import BytesIO
import requests

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



def open_result_window(bien_so_quet, plate_url, face_url,uutien ,parent):
    result = {}
    win = tk.Toplevel(parent)
    win.title("Kết quả quét")

    # Label hiển thị biển số
    label_bienso = tk.Label(win, text=f"Biển số quét được: {bien_so_quet}", font=("Arial", 14))
    label_bienso.pack(pady=10)

    # Khung chứa ảnh
    frame_images = tk.Frame(win)
    frame_images.pack(pady=10)

    # Hiện ảnh biển số
    if plate_url:
        try:
            if plate_url.startswith("http"):
                response = requests.get(plate_url)
                img_data = response.content
            else:
                with open(plate_url, "rb") as f:
                    img_data = f.read()

            img = Image.open(BytesIO(img_data)).resize((300, 200))
            tk_img_plate = ImageTk.PhotoImage(img)

            label_plate = tk.Label(frame_images, image=tk_img_plate)
            label_plate.image = tk_img_plate
            label_plate.pack(side="left", padx=10)
        except Exception as e:
            print("Lỗi load ảnh biển số:", e)

    # Hiện ảnh khuôn mặt
    if face_url:
        try:
            if face_url.startswith("http"):
                response = requests.get(face_url)
                img_data = response.content
            else:
                with open(face_url, "rb") as f:
                    img_data = f.read()

            img = Image.open(BytesIO(img_data)).resize((200, 200))
            tk_img_face = ImageTk.PhotoImage(img)

            label_face = tk.Label(frame_images, image=tk_img_face)
            label_face.image = tk_img_face
            label_face.pack(side="right", padx=10)
        except Exception as e:
            print("Lỗi load ảnh khuôn mặt:", e)

    # ==========================
    # Chọn ưu tiên
    # ==========================
    frame_choice = tk.Frame(win)
    frame_choice.pack(pady=10)

    tk.Label(frame_choice, text="Bạn có muốn ưu tiên không?", font=("Arial", 12)).pack(anchor="w")

    uutien_var = tk.BooleanVar(value=False)
    tk.Radiobutton(frame_choice, text="Ưu tiên", variable=uutien_var, value=True).pack(anchor="w")
    tk.Radiobutton(frame_choice, text="Không ưu tiên", variable=uutien_var, value=False).pack(anchor="w")

    # ==========================
    # Hàm xác nhận -> trả về kết quả
    # ==========================
    def on_confirm():
        result["bien_so"] = bien_so_quet
        result["plate_img"] = plate_url
        result["face_img"]= face_url
        result["uutien"] = uutien_var.get()
        win.destroy()

    btn_save = tk.Button(win, text="Xác nhận", command=on_confirm, bg="green", fg="white", font=("Arial", 12))
    btn_save.pack(pady=15)

    # Chờ người dùng đóng cửa sổ
    parent.wait_window(win)

    return (
        result.get("bien_so"),
        result.get("plate_img"),
        result.get("face_img"),
        result.get("uutien"),
    )

    # ==========================
    # Hàm lưu vào Firestore
    # ==========================
def save_to_firestore_uutien(bien_so, plate_url, face_url, uutien):
    data = {
        "biensoxe": bien_so,
        "uutien": uutien,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    print("Dữ liệu để lưu:", data)

    # Lưu vào collection "thongtinkhach"
    db.collection("thongtinkhach").add(data)
    print("Đã lưu vào Firestore!")

    # Lưu ngày
    today = datetime.today().strftime("%d%m%Y")
    db.collection("lichsuhoatdong").document(today).set({"ngay": today}, merge=True)

    # Document xe trong ngày
    xe_doc_ref = db.collection("lichsuhoatdong").document(today).collection("xemay").document(bien_so)

    # Tăng số lần vào
    doc = xe_doc_ref.get()
    solanvao = doc.to_dict().get("solanvao", 0) if doc.exists else 0
    xe_doc_ref.set({"solanvao": solanvao + 1}, merge=True)

    # =============================
    # Tạo timeline mới
    # =============================
    timeline_col_ref = xe_doc_ref.collection("timeline")

    # Tạo id timeline tự tăng
    existing_docs = list(timeline_col_ref.list_documents())
    max_index = -1
    for d in existing_docs:
        try:
            idx = int(d.id.replace("timeline", ""))
            if idx > max_index:
                max_index = idx
        except:
            continue
    new_index = max_index + 1
    timeline_doc_id = f"timeline{new_index}"

    time_now = datetime.now().strftime("%H:%M:%S")
    timeline_data = {
        "timein": time_now,
        "biensoxevao": plate_url,
        "khuonmatvao": face_url,
        "timeout": None,
        "biensoxera": None,
        "khuonmatra": None
    }
    timeline_col_ref.document(timeline_doc_id).set(timeline_data, merge=True)
    print(f"Đã lưu timeline {timeline_doc_id} cho xe {bien_so}")

    # Realtime DB
    firebase_put("trangthaicong", True, include_timestamp=False)
    datetime_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    firebase_put(f"biensotrongbai/{bien_so}", {
        "timestamp": datetime_str,
        "khach": uutien
    })




def run_license_scan( root):
    while True:
        # lấy từ nhận diện biển số xe
        bien_so, url_image_detected = detect_license_plate()
        bien_so_quet = normalize_plate(bien_so)

        # Kiểm tra hợp lệ
        ds_bien_so_raw = get_field_from_all_docs("thongtindangky", "biensoxe")
        ds_map_bien_so_phu_raw = get_field_from_all_docs("thongtindangky", "biensophu")
        ds_bien_so_phu_raw = [item["bienSo"] for item in ds_map_bien_so_phu_raw if item and "bienSo" in item]

        ds_bien_so = [normalize_plate(val) for val in ds_bien_so_raw if val]
        ds_bien_so_phu = [normalize_plate(val) for val in ds_bien_so_phu_raw if val]

        hop_le = bien_so_quet in ds_bien_so
        hop_le_phu = bien_so_quet in ds_bien_so_phu



        if hop_le or hop_le_phu:



            # Ghi Firestore
            today = datetime.today().strftime("%d%m%Y")
            xe_doc_ref = db.collection("lichsuhoatdong").document(today).collection("xemay").document(bien_so_quet)
            db.collection("lichsuhoatdong").document(today).set({"ngay": today}, merge=True)

            # Tăng số lần vào
            doc = xe_doc_ref.get()
            solanvao = doc.to_dict().get("solanvao", 0) if doc.exists else 0
            xe_doc_ref.set({"solanvao": solanvao + 1}, merge=True)

            # Chụp ảnh khuôn mặt
            image_url_vao = capture_face_and_upload()


            # Ghi timeline
            time_now = datetime.now().strftime("%H:%M:%S")
            timeline_data = {
                "timein": time_now,
                "biensoxevao": url_image_detected,
                "khuonmatvao": image_url_vao,
                "timeout": None,
                "biensoxera": None,
                "khuonmatra": None
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

            return url_image_detected, image_url_vao, time_now,bien_so_quet




        else:


            # Hỏi có muốn đăng ký khách không
            answer = messagebox.askyesno("Đăng ký khách", "Bạn có muốn đăng ký biển số xe với tư cách là khách không?")

            if answer:
                root.withdraw()

                bien_so, plate_url = detect_license_plate()
                biensoquet = normalize_plate(bien_so)
                image_url_vao = capture_face_and_upload()
                uutien= None

                # Lấy kết quả từ cửa sổ
                bien_so, plate_url, face_url, uutien = open_result_window(biensoquet, plate_url,image_url_vao,uutien, root)

                # Bây giờ mới gọi lưu vào Firestore
                save_to_firestore_uutien(bien_so, plate_url, face_url, uutien)

                # Hiện lại cửa sổ chính
                root.deiconify()

                # ✅ Trả về đủ 4 giá trị cho giao diện
                return plate_url, face_url, datetime.now().strftime("%H:%M:%S"), bien_so

            else:


                print("Người dùng không đăng ký khách")
                return None  # không có gì để ghi


        time.sleep(1)




