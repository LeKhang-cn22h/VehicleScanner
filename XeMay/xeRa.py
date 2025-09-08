import tkinter as tk
from PIL import Image, ImageTk
import requests
from io import BytesIO
import threading
from datetime import datetime
from nhanDien import detect_license_plate
from firebase_service import FirebaseService
from face_detection.train_face import capture_face_and_upload
from deepface import DeepFace
from firebase_admin import firestore
import time

# =====================
# Hàm xử lý quét và so khớp
# =====================
def run_license_scan_ra(label_status, canvas_old, canvas_new,root):
    firebase_service = FirebaseService()
    db = firestore.client()

    while True:
        # 1. Quét biển số
        bien_so, url_image_detected = detect_license_plate()
        check_plate = False
        if not bien_so:
            check_plate=False
            return check_plate
            continue
        bien_so_quet = bien_so.replace(".", "").upper()
        print("Biển số quét được:", bien_so_quet)

        # 2. Kiểm tra hợp lệ
        ds_bien_so = firebase_service.get_all_license_plates()
        if bien_so_quet not in ds_bien_so:
            label_status.config(text=f"Biển số {bien_so_quet} không hợp lệ ", bg="red")
            label_status.update()
            time.sleep(2)
            continue

        # 3. Lấy dữ liệu biển số
        bien_so_data = firebase_service.get_license_plate_data(bien_so_quet)
        if not bien_so_data:

            time.sleep(2)
            continue

        # 4. Lấy timeline gần nhất
        today = datetime.today().strftime("%d%m%Y")
        xe_doc_ref = db.collection("lichsuhoatdong").document(today).collection("xemay").document(bien_so_quet)
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

        if max_index >= 0:
            timeline_doc_id = f"timeline{max_index}"
            timeline_ref = xe_doc_ref.collection("timeline").document(timeline_doc_id)
            timeline_data = timeline_ref.get().to_dict()
            url_khuonmatvao = timeline_data.get("khuonmatvao")
            print("URL khuôn mặt vào gần nhất:", url_khuonmatvao)

        else:

            timeline_doc_id = None
            timeline_ref = None
            url_khuonmatvao = None

        # 5. Chụp khuôn mặt mới
        image_url_new_face = capture_face_and_upload()

        # # 6. Hiển thị 2 ảnh lên GUI
        # def show_image_from_url(canvas, url):
        #     if url:
        #         try:
        #             response = requests.get(url)
        #             img = Image.open(BytesIO(response.content)).resize((200, 200))
        #             img_tk = ImageTk.PhotoImage(img)
        #             canvas.img_tk = img_tk  # lưu reference
        #             canvas.create_image(0, 0, anchor="nw", image=img_tk)
        #         except:
        #             pass
        #
        # canvas_old.delete("all")
        # canvas_new.delete("all")
        # show_image_from_url(canvas_old, url_khuonmatvao)
        # show_image_from_url(canvas_new, image_url_new_face)



        same_person = False
        CUSTOM_THRESHOLD = 0.35  # ngưỡng tùy chỉnh (càng thấp càng khắt khe)

        if url_khuonmatvao and image_url_new_face:
            try:
                result = DeepFace.verify(
                    img1_path=image_url_new_face,
                    img2_path=url_khuonmatvao,
                    model_name="ArcFace",  # model chính xác
                    detector_backend="retinaface",  # backend dò khuôn mặt tốt
                    distance_metric="cosine",  # metric phù hợp ArcFace
                    align=True,
                    enforce_detection=True
                )

                dist = float(result.get("distance", 1.0))
                thr = float(result.get("threshold", 0.0))
                verified_default = result.get("verified", False)

                # So sánh bằng ngưỡng custom
                verified_custom = dist <= CUSTOM_THRESHOLD
                same_person = verified_default and verified_custom

                print(f"📊 Khoảng cách = {dist:.4f}, Ngưỡng mặc định = {thr:.4f}, Ngưỡng custom = {CUSTOM_THRESHOLD}")
                if same_person:

                    print("Kết quả: CÙNG 1 NGƯỜI")
                else:
                    print(" Kết quả: KHÁC NGƯỜI")

            except Exception as e:
                print("Lỗi khi so khớp khuôn mặt:", e)

        if same_person:
            check_plate = True

            # Xử lý tiếp dữ liệu phía sau như update Firestore
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
                        "biensoxera": url_image_detected,
                        "khuonmatra": image_url_new_face
                    }, merge=True)
                    print(f"Đã cập nhật timeline {timeline_doc_id}")
                else:
                    print("Không tìm thấy timeline để cập nhật.")
                return
            else:


                # Nếu trạng thái True, cảnh báo
                firebase_service.update_canhbao(bien_so_quet, True)
                check_warn = True

                print("Xe đã ra trước đó, đã gửi cảnh báo.")
                return check_warn
            time.sleep(2)  # delay để người dùng thấy thông báo


            break
        else:

            label_status.config(text=f" Không phải cùng người", bg="red")

        label_status.update()
        time.sleep(2)  # delay giữa các lần quét

# =====================
# GUI Tkinter
# # =====================
# root = tk.Tk()
# root.title("Hệ thống quản lý xe tự động")
# root.geometry("700x400")
#
# label_status = tk.Label(root, text="Đang chờ quét xe...", font=("Arial", 18), width=60, height=2, bg="gray")
# label_status.pack(pady=10)
#
# frame_images = tk.Frame(root)
# frame_images.pack()
#
# canvas_old = tk.Canvas(frame_images, width=200, height=200, bg="white")
# canvas_old.pack(side="left", padx=20)
#
# canvas_new = tk.Canvas(frame_images, width=200, height=200, bg="white")
# canvas_new.pack(side="right", padx=20)
#
#
#
# root.mainloop()



