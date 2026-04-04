from datetime import datetime
from XeMay.nhanDien import detect_license_plate
from firebase_service import FirebaseService
from face_detection.train_face import capture_face_and_upload
from deepface import DeepFace
from firebase_admin import firestore
import time


def run_license_scan_ra(root):
    firebase_service = FirebaseService()
    db = firestore.client()

    # ==============================
    # 1. Quét biển số
    # ==============================
    bien_so, url_image_detected = detect_license_plate()
    if not bien_so:
        return {"success": False, "message": "Không quét được biển số", "data": None}

    bien_so_quet = bien_so.replace(".", "").upper()
    print("Biển số quét được:", bien_so_quet)

    # ==============================
    # 2. Kiểm tra hợp lệ
    # ==============================
    ds_bien_so = firebase_service.get_all_license_plates()
    if bien_so_quet not in ds_bien_so:
        return {"success": False, "message": f"Biển số {bien_so_quet} không hợp lệ", "data": None}

    # ==============================
    # 3. Lấy dữ liệu biển số
    # ==============================
    bien_so_data = firebase_service.get_license_plate_data(bien_so_quet)
    if not bien_so_data:
        return {"success": False, "message": f"Không tìm thấy dữ liệu cho {bien_so_quet}", "data": None}

    # ==============================
    # 4. Lấy timeline gần nhất
    # ==============================
    today = datetime.today().strftime("%d%m%Y")
    xe_doc_ref = (
        db.collection("lichsuhoatdong")
        .document(today)
        .collection("xemay")
        .document(bien_so_quet)
    )
    timeline_docs = xe_doc_ref.collection("timeline").list_documents()

    max_index = 0
    for tdoc in timeline_docs:
        name = tdoc.id
        if name.startswith("timeline"):
            try:
                index = int(name.replace("timeline", ""))
                if index > max_index:
                    max_index = index
            except ValueError:
                continue

    timeline_ref      = None
    url_khuonmatvao   = None
    url_xevao         = None
    timeIn            = None

    if max_index >= 0:
        timeline_doc_id = f"timeline{max_index}"
        timeline_ref    = xe_doc_ref.collection("timeline").document(timeline_doc_id)
        timeline_data   = timeline_ref.get().to_dict()
        if timeline_data:
            url_khuonmatvao = timeline_data.get("khuonmatvao")
            url_xevao       = timeline_data.get("biensoxevao")
            timeIn          = timeline_data.get("timein")

    # ==============================
    # Helper: so khớp khuôn mặt
    # ==============================
    def verify_face(img_new, img_old, label=""):
        CUSTOM_THRESHOLD = 0.35
        try:
            result = DeepFace.verify(
                img1_path=img_new,
                img2_path=img_old,
                model_name="ArcFace",
                detector_backend="retinaface",
                distance_metric="cosine",
                align=True,
                enforce_detection=False
            )
            dist             = float(result.get("distance", 1.0))
            verified_default = result.get("verified", False)
            verified_custom  = dist <= CUSTOM_THRESHOLD
            same_person      = verified_default and verified_custom
            print(f"[{label}] Khoảng cách = {dist:.4f}, Ngưỡng custom = {CUSTOM_THRESHOLD}")
            return same_person
        except Exception as e:
            print(f"Lỗi so khớp khuôn mặt [{label}]: {e}")
            return False

    # ==============================
    # Helper: ghi timeline ra
    # ==============================
    def ghi_timeline_ra(time_now, url_xe_ra, url_mat_ra):
        if timeline_ref:
            timeline_ref.set({
                "timeout":    time_now,
                "biensoxera": url_xe_ra,
                "khuonmatra": url_mat_ra
            }, merge=True)
            print(f"✅ Đã ghi timeline ra: {timeline_doc_id}")
        else:
            print("❌ Không tìm thấy timeline để cập nhật")

    # ==============================
    # Helper: tăng số lần ra
    # ==============================
    def tang_solanra():
        doc_xe = xe_doc_ref.get()
        solanra = doc_xe.to_dict().get("solanra", 0) if doc_xe.exists else 0
        xe_doc_ref.set({"solanra": solanra + 1}, merge=True)

    # ==============================
    # 5. Trường hợp xe KHÁCH
    # ==============================
    if firebase_service.has_khach(bien_so_quet):
        image_url_new_face = capture_face_and_upload()
        if not image_url_new_face:
            return {"success": False, "message": "Không chụp được khuôn mặt khách", "data": None}

        print("debug link mat khách:")
        print("  mat_ra :", image_url_new_face)
        print("  mat_vao:", url_khuonmatvao)

        same_person = verify_face(image_url_new_face, url_khuonmatvao, label="Khách") if url_khuonmatvao else False

        if not same_person:
            return {"success": False, "message": "Khuôn mặt khách không khớp", "data": None}

        # ✅ Xử lý Firebase
        firebase_service.delete_license_plate(bien_so_quet)
        firebase_service.update_license_plate_field(bien_so_quet, True)
        tang_solanra()

        time_now = datetime.now().strftime("%H:%M:%S")
        ghi_timeline_ra(time_now, url_image_detected, image_url_new_face)

        root.deiconify()
        return {
            "success": True,
            "message": "Xác thực khách thành công, xe được ra",
            "data": {
                "bien_so":   bien_so_quet,
                "anh_xe_ra": url_image_detected,
                "timeIn":    timeIn,
                "anh_xe_vao": url_xevao,
                "mat_vao":   url_khuonmatvao,
                "time_now":  time_now,
                "mat_ra":    image_url_new_face
            }
        }

    # ==============================
    # 6. Trường hợp xe BÌNH THƯỜNG
    # ==============================
    trangthai = bien_so_data.get('trangthai', False)

    if trangthai is True:
        # Xe đã ra trước đó → cảnh báo
        firebase_service.update_canhbao(bien_so_quet, True)
        time.sleep(10)
        bien_so_data = firebase_service.get_license_plate_data(bien_so_quet)
        if bien_so_data and bien_so_data.get('trangthai') is False:
            firebase_service.update_license_plate_field(bien_so_quet, True)
            firebase_service.delete_license_plate(bien_so_quet)
        return {
            "success": False,
            "message": "Xe đã ra trước đó, đã gửi cảnh báo",
            "data": {"bien_so": bien_so_quet, "anh_xe_ra": None, "mat_ra": None}
        }

    # trangthai is False → xe đang trong bãi, cho ra bình thường
    image_url_new_face = capture_face_and_upload()
    if not image_url_new_face:
        return {"success": False, "message": "Không chụp được khuôn mặt", "data": None}

    same_person = verify_face(image_url_new_face, url_khuonmatvao, label="Bình thường") if url_khuonmatvao else False

    if not same_person:
        return {"success": False, "message": "Khuôn mặt không khớp với người vào", "data": None}

    # ==============================
    # Kiểm tra số lượt
    # ==============================
    collection_ref = db.collection("thongtindangky")

    matched_doc = None
    for doc in collection_ref.where("biensoxe", "==", bien_so_quet).stream():
        matched_doc = doc
        break
    if not matched_doc:
        for doc in collection_ref.where("biensophu", "==", bien_so_quet).stream():
            matched_doc = doc
            break

    if matched_doc:
        data = matched_doc.to_dict()
        for key, value in data.items():
            if "luot" in key.lower() and isinstance(value, (int, float)) and value <= 0:
                # Hết lượt → warning, chưa xử lý Firebase
                return {
                    "warning": True,
                    "message": "Bạn đã hết lượt mua vé, bạn có muốn trả tiền mặt 1 lượt không",
                    "data": {
                        "bien_so":    bien_so_quet,
                        "anh_xe_ra":  url_image_detected,
                        "mat_ra":     image_url_new_face,
                        "timeIn":     timeIn,
                        "anh_xe_vao": url_xevao,
                        "mat_vao":    url_khuonmatvao,
                        "time_now":   datetime.now().strftime("%H:%M:%S")
                    }
                }

    # ✅ Xử lý Firebase — phải trước return
    firebase_service.update_license_plate_field(bien_so_quet, True)
    firebase_service.delete_license_plate(bien_so_quet)
    tang_solanra()

    time_now = datetime.now().strftime("%H:%M:%S")
    ghi_timeline_ra(time_now, url_image_detected, image_url_new_face)

    return {
        "success": True,
        "message": "Xác thực xe bình thường thành công, xe được ra",
        "data": {
            "bien_so":    bien_so_quet,
            "anh_xe_ra":  url_image_detected,
            "mat_ra":     image_url_new_face,
            "timeIn":     timeIn,
            "anh_xe_vao": url_xevao,
            "mat_vao":    url_khuonmatvao,
            "time_now":   time_now
        }
    }