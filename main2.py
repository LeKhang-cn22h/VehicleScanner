from nhan_dien_vao import detect_license_plate
from firebase_service import FirebaseService
import time
from FrontCarPhoto import capture_and_upload_front_image
from datetime import datetime
from firebase_admin import firestore

def normalize_plate(plate):
    if not plate:
        return None
    return plate.replace(".", "").upper()

# Trừ số lượt khi ra
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
                return True
            else:
                print("Số lượt ko đủ. Bạn cần mua thêm lượt")
                return False
    return False

# Quét biển
bien_so, url_image_detected = detect_license_plate()
bien_so_quet = normalize_plate(bien_so)

print("Biển số quét được:", bien_so_quet)

# Lấy danh sách biển số từ Firebase
firebase_service = FirebaseService()
ds_bien_so = firebase_service.get_all_license_plates()

print("Danh sách biển số trong DB:", ds_bien_so)

# So sánh
if bien_so_quet:
    if bien_so_quet in ds_bien_so:
        # === TIẾP TỤC XỬ LÝ TRẠNG THÁI ===
        bien_so_data = firebase_service.get_license_plate_data(bien_so_quet)
        if bien_so_data:
            trangthai = bien_so_data.get('trangthai')
            if trangthai is False:
                print("Biển số có 'trangthai' = False.")

                firebase_service.update_license_plate_field(True)
                firebase_service.delete_license_plate(bien_so_quet)

                # === THAO TÁC FIRESTORE ===
                db = firestore.client()
                today = datetime.today().strftime("%d%m%Y")
                xe_doc_ref = db.collection("lichsuhoatdong").document(today).collection("xe").document(bien_so_quet)
                db.collection("lichsuhoatdong").document(today).set({"ngay": today}, merge=True)

                # Lấy document xe
                doc = xe_doc_ref.get()
                if doc.exists:
                    data = doc.to_dict()
                    solanra = data.get("solanra", 0)
                else:
                    solanra = 0

                # Cập nhật số lần ra
                solanra += 1
                xe_doc_ref.set({"solanra": solanra}, merge=True)

                # Lấy ảnh đầu xe
                image_url_vao = capture_and_upload_front_image()
                time_now = datetime.now().strftime("%H:%M:%S")

                # Ghi vào timeline gần nhất (tức là timeline có index lớn nhất)
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
                    timeline_ref.set({
                        "timeOut": time_now,
                        "imageOut": url_image_detected,
                        "hinhdauxera": image_url_vao
                    }, merge=True)
                    print(f"Đã cập nhật timeline {timeline_doc_id}")
                else:
                    print("Không tìm thấy timeline để cập nhật.")
            else:
                # Nếu trạng thái ban đầu là True, thực hiện cảnh báo và đợi cập nhật
                firebase_service.update_canhbao(bien_so_quet, True)
                time.sleep(10)

                # Kiểm tra lại trạng thái
                bien_so_data = firebase_service.get_license_plate_data(bien_so_quet)
                if bien_so_data:
                    trangthai = bien_so_data.get('trangthai')
                    print("Cập nhật trạng thái sau 10s:", trangthai)
                    if trangthai is False:
                        print("Biển số có 'trangthai' = False.")

                        firebase_service.update_license_plate_field(True)
                        firebase_service.delete_license_plate(bien_so_quet)

                        # === Tương tự ghi log ra như trên ===
                        db = firestore.client()
                        today = datetime.today().strftime("%d%m%Y")
                        xe_doc_ref = db.collection("lichsuhoatdong").document(today).collection("xe").document(bien_so_quet)
                        db.collection("lichsuhoatdong").document(today).set({"ngay": today}, merge=True)

                        doc = xe_doc_ref.get()
                        if doc.exists:
                            data = doc.to_dict()
                            solanra = data.get("solanra", 0)
                        else:
                            solanra = 0

                        solanra += 1
                        xe_doc_ref.set({"solanra": solanra}, merge=True)

                        image_url_vao = capture_and_upload_front_image()
                        time_now = datetime.now().strftime("%H:%M:%S")

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
                            xe_doc_ref.collection("timeline").document(timeline_doc_id).set({
                                "timeOut": time_now,
                                "imageOut": url_image_detected,
                                "hinhdauxera": image_url_vao
                            }, merge=True)
                            print(f"Đã cập nhật timeline {timeline_doc_id}")
                        else:
                            print("Không tìm thấy timeline để cập nhật.")
        else:
            print("Không lấy được dữ liệu của biển số.")
    else:
        print("Biển số không có trong DB.")
else:
    print("Không quét được biển số.")


from nhan_dien_vao import detect_license_plate
from firebase_service import FirebaseService
import time
from FrontCarPhoto import capture_and_upload_front_image
def normalize_plate(plate):
    if not plate:
        return None
    return (plate.replace(".", "").upper())


# plate = detect_license_plate()
# bien_so_quet = normalize_plate(plate)
bien_so, url_image_detected = detect_license_plate()
bien_so_quet = normalize_plate(bien_so)

print("Biển số quét được:", bien_so_quet)

#  Lấy danh sách biển số từ Firebase
firebase_service = FirebaseService()
ds_bien_so = firebase_service.get_all_license_plates()

print("Danh sách biển số trong DB:", ds_bien_so)

#  So sánh
if bien_so_quet:
    if bien_so_quet in ds_bien_so:
        bien_so_data = firebase_service.get_license_plate_data(bien_so_quet)
        if bien_so_data:
            trangthai = bien_so_data.get('trangthai')  # Lấy giá trị 'trangthai' từ dữ liệu
            if trangthai is False:
                print("Biển số có 'trangthai' = False.")

                #  Cập nhật trangthaicong = True
                firebase_service.update_license_plate_field(True)
                firebase_service.delete_license_plate(bien_so_quet)
            else:
                firebase_service.update_canhbao(bien_so_quet,True)
                time.sleep(10)
                bien_so_data = firebase_service.get_license_plate_data(bien_so_quet)
                if bien_so_data:
                    trangthai = bien_so_data.get('trangthai')  # Lấy giá trị 'trangthai' từ dữ liệu
                    print("Cap nhat trang thai sau 10s: ",trangthai)
                    if trangthai is False:
                        print("Biển số có 'trangthai' = False.")

                        # Cập nhật trangthaicong = True
                        firebase_service.update_license_plate_field(True)
                        firebase_service.delete_license_plate(bien_so_quet)


        else:
            print("Không lấy được dữ liệu của biển số.")
    else:
        print("Biển số không có trong DB.")
else:
    print(" Không quét được biển số.")

