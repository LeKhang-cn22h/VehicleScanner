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

    # 1. Quét biển số
    bien_so, url_image_detected = detect_license_plate()
    if not bien_so:
        return {"success": False, "message": "Không quét được biển số", "data": None}

    bien_so_quet = bien_so.replace(".", "").upper()
    print("Biển số quét được:", bien_so_quet)

    # 2. Kiểm tra hợp lệ
    ds_bien_so = firebase_service.get_all_license_plates()
    if bien_so_quet not in ds_bien_so:
        return {"success": False, "message": f"Biển số {bien_so_quet} không hợp lệ", "data": None}

    # 3. Lấy dữ liệu biển số
    bien_so_data = firebase_service.get_license_plate_data(bien_so_quet)
    if not bien_so_data:
        return {"success": False, "message": f"Không tìm thấy dữ liệu cho {bien_so_quet}", "data": None}

    # 4. Lấy timeline gần nhất
    today = datetime.today().strftime("%d%m%Y")
    xe_doc_ref = db.collection("lichsuhoatdong").document(today).collection("xemay").document(bien_so_quet)
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

    timeline_ref, url_khuonmatvao, url_xevao, timeIn = None, None, None, None
    if max_index >= 0:
        timeline_doc_id = f"timeline{max_index}"
        timeline_ref = xe_doc_ref.collection("timeline").document(timeline_doc_id)
        timeline_data = timeline_ref.get().to_dict()
        if timeline_data:
            url_khuonmatvao = timeline_data.get("khuonmatvao")
            url_xevao = timeline_data.get("biensoxevao")
            timeIn = timeline_data.get("timein")

    # ==============================
    # 5. Kiểm tra nếu xe là loại "khách"
    # ==============================
    if firebase_service.has_khach(bien_so_quet):
        image_url_new_face = capture_face_and_upload()
        if not image_url_new_face:
            return {"success": False, "message": "Không chụp được khuôn mặt khách", "data": None}

        same_person = False
        CUSTOM_THRESHOLD = 0.35
        print("debug link mat")
        print(image_url_new_face)
        print(url_khuonmatvao)
        print(url_xevao)
        print(timeIn)
        if url_khuonmatvao:
            try:
                result = DeepFace.verify(
                    img1_path=image_url_new_face,
                    img2_path=url_khuonmatvao,
                    model_name="ArcFace",
                    detector_backend="retinaface",
                    distance_metric="cosine",
                    align=True,
                    enforce_detection=False
                )
                dist = float(result.get("distance", 1.0))
                verified_default = result.get("verified", False)
                verified_custom = dist <= CUSTOM_THRESHOLD
                same_person = verified_default and verified_custom
                print(f"[Khách] Khoảng cách = {dist:.4f}, Ngưỡng custom = {CUSTOM_THRESHOLD}")
            except Exception as e:
                return {"success": False, "message": f"Lỗi khi so khớp khuôn mặt khách: {e}", "data": None}

        if same_person:
            firebase_service.delete_license_plate(bien_so_quet)
            firebase_service.update_license_plate_field(bien_so_quet, True)

            doc_xe = xe_doc_ref.get()
            solanra = doc_xe.to_dict().get("solanra", 0) if doc_xe.exists else 0
            xe_doc_ref.set({"solanra": solanra + 1}, merge=True)

            time_now = datetime.now().strftime("%H:%M:%S")
            if timeline_ref:
                timeline_ref.set({
                    "timeout": time_now,
                    "biensoxera": url_image_detected,
                    "khuonmatra": image_url_new_face
                }, merge=True)

            root.deiconify()
            return {
                "success": True,
                "message": "Xác thực khách thành công, xe được ra",
                "data": {
                    "bien_so": bien_so_quet,
                    "anh_xe_ra": url_image_detected,
                    "timeIn": timeIn,
                    "anh_xe_vao": url_xevao,
                    "mat_vao": url_khuonmatvao,
                    "time_now": time_now,
                    "mat_ra": image_url_new_face
                }
            }
        else:
            return {"success": False, "message": "Khuôn mặt khách không khớp", "data": None}

    # ==============================
    # 6. Trường hợp xe bình thường
    # ==============================
    trangthai = bien_so_data.get('trangthai', False)
    if trangthai is False:
        image_url_new_face = capture_face_and_upload()
        if not image_url_new_face:
            return {"success": False, "message": "Không chụp được khuôn mặt", "data": None}

        same_person = False
        CUSTOM_THRESHOLD = 0.35
        if url_khuonmatvao:
            try:
                result = DeepFace.verify(
                    img1_path=image_url_new_face,
                    img2_path=url_khuonmatvao,
                    model_name="ArcFace",
                    detector_backend="retinaface",
                    distance_metric="cosine",
                    align=True,
                    enforce_detection=False
                )
                dist = float(result.get("distance", 1.0))
                verified_default = result.get("verified", False)
                verified_custom = dist <= CUSTOM_THRESHOLD
                same_person = verified_default and verified_custom
                print(f"[Bình thường] Khoảng cách = {dist:.4f}, Ngưỡng custom = {CUSTOM_THRESHOLD}")
            except Exception as e:
                return {"success": False, "message": f"Lỗi khi so khớp khuôn mặt: {e}", "data": None}

        if same_person:
            collection_ref = db.collection("thongtindangky")

            # Truy vấn: lấy doc mà biensoxe hoặc biensophu khớp
            matched_doc = None
            docs = collection_ref.where("biensoxe", "==", bien_so_quet).stream()
            for doc in docs:
                matched_doc = doc
                break  # chỉ cần 1 doc khớp

            if not matched_doc:
                docs_phu = collection_ref.where("biensophu", "==", bien_so_quet).stream()
                for doc in docs_phu:
                    matched_doc = doc
                    break  # chỉ cần 1 doc khớp

            if matched_doc:
                data = matched_doc.to_dict()
                # Kiểm tra lượt
                for key, value in data.items():
                    if "luot" in key.lower() and isinstance(value, (int, float)) and value <=0:
                        return {
                            "warning": True,
                            "message": "Bạn đã hết lượt mua vé, bạn có muốn trả tiền mặt 1 lượt không",
                                "data": {
                        "bien_so": bien_so_quet,
                        "anh_xe_ra": url_image_detected,
                        "mat_ra": image_url_new_face,
                        "timeIn": timeIn,
                        "anh_xe_vao": url_xevao,
                        "mat_vao": url_khuonmatvao,
                        "time_now": datetime.now().strftime("%H:%M:%S")
                    }}

            return {
                "success": True,
                "message": "Xác thực xe bình thường thành công",
                "data": {
                    "bien_so": bien_so_quet,
                    "anh_xe_ra": url_image_detected,
                    "mat_ra": image_url_new_face,
                    "timeIn": timeIn,
                    "anh_xe_vao": url_xevao,
                    "mat_vao": url_khuonmatvao,
                    "time_now": datetime.now().strftime("%H:%M:%S")
                }}
            firebase_service.update_license_plate_field(bien_so_quet, True)
            firebase_service.delete_license_plate(bien_so_quet)

            doc_xe = xe_doc_ref.get()
            solanra = doc_xe.to_dict().get("solanra", 0) if doc_xe.exists else 0
            xe_doc_ref.set({"solanra": solanra + 1}, merge=True)

            time_now = datetime.now().strftime("%H:%M:%S")
            if timeline_ref:
                timeline_ref.set({
                    "timeout": time_now,
                    "biensoxera": url_image_detected,
                    "khuonmatra": image_url_new_face
                }, merge=True)

            return {
                "success": True,
                "message": "Xác thực bình thường thành công, xe được ra",
                "data": {
                    "bien_so": bien_so_quet,
                    "anh_xe_ra": url_image_detected,
                    "timeIn": timeIn,
                    "anh_xe_vao": url_xevao,
                    "mat_vao": url_khuonmatvao,
                    "time_now": time_now,
                    "mat_ra": image_url_new_face
                }
            }
        else:
            return {"success": False, "message": "Khuôn mặt không khớp", "data": None}

    else:
        # Xe đã ra trước đó
        firebase_service.update_canhbao(bien_so_quet, True)
        time.sleep(10)
        bien_so_data = firebase_service.get_license_plate_data(bien_so_quet)
        if bien_so_data and bien_so_data.get('trangthai') is False:
            # Xử lý như xe ra
            firebase_service.update_license_plate_field(bien_so_quet, True)
            firebase_service.delete_license_plate(bien_so_quet)

        return {
            "success": False,
            "message": "Xe đã ra trước đó, đã gửi cảnh báo",
            "data": {"bien_so": bien_so_quet, "anh_xe_ra": None, "mat_ra": None}
        }

# # Lấy danh sách biển số từ Firebase
# firebase_service = FirebaseService()
# ds_bien_so = firebase_service.get_all_license_plates()
# def run_license_scan_ra():
#     firebase_service = FirebaseService()
#     db = firestore.client()
#
#     # 1. Quét biển số
#     bien_so, url_image_detected = detect_license_plate()
#     if not bien_so:
#         return {
#             "success": False,
#             "message": "Không quét được biển số",
#             "data": None
#         }
#
#     bien_so_quet = bien_so.replace(".", "").upper()
#     print("Biển số quét được:", bien_so_quet)
#
#     # 2. Kiểm tra hợp lệ
#     ds_bien_so = firebase_service.get_all_license_plates()
#     if bien_so_quet not in ds_bien_so:
#         return {
#             "success": False,
#             "message": f"Biển số {bien_so_quet} không hợp lệ",
#             "data": None
#         }
#
#     # 3. Lấy dữ liệu biển số
#     bien_so_data = firebase_service.get_license_plate_data(bien_so_quet)
#     if not bien_so_data:
#         return {
#             "success": False,
#             "message": f"Không tìm thấy dữ liệu cho {bien_so_quet}",
#             "data": None
#         }
#
#     # 4. Lấy timeline gần nhất
#     today = datetime.today().strftime("%d%m%Y")
#     xe_doc_ref = db.collection("lichsuhoatdong").document(today).collection("xemay").document(bien_so_quet)
#     timeline_docs = xe_doc_ref.collection("timeline").list_documents()
#     max_index = -1
#     for tdoc in timeline_docs:
#         name = tdoc.id
#         if name.startswith("timeline"):
#             try:
#                 index = int(name.replace("timeline", ""))
#                 if index > max_index:
#                     max_index = index
#             except ValueError:
#                 continue
#
#     # Khởi tạo mặc định
#     timeline_doc_id = None
#     timeline_ref = None
#     url_khuonmatvao = None
#     url_xevao = None
#     timeIn = None
#
#     if max_index >= 0:
#         timeline_doc_id = f"timeline{max_index}"
#         timeline_ref = xe_doc_ref.collection("timeline").document(timeline_doc_id)
#         timeline_data = timeline_ref.get().to_dict()
#         url_khuonmatvao = timeline_data.get("khuonmatvao")
#         url_xevao = timeline_data.get("biensoxevao")
#         timeIn = timeline_data.get("timein")
#
#     # 5. Chụp khuôn mặt mới
#     image_url_new_face = capture_face_and_upload()
#     same_person = False
#     CUSTOM_THRESHOLD = 0.35
#
#     if url_khuonmatvao and image_url_new_face:
#         try:
#             result = DeepFace.verify(
#                 img1_path=image_url_new_face,
#                 img2_path=url_khuonmatvao,
#                 model_name="ArcFace",
#                 detector_backend="retinaface",
#                 distance_metric="cosine",
#                 align=True,
#                 enforce_detection=True
#             )
#             dist = float(result.get("distance", 1.0))
#             thr = float(result.get("threshold", 0.0))
#             verified_default = result.get("verified", False)
#             verified_custom = dist <= CUSTOM_THRESHOLD
#             same_person = verified_default and verified_custom
#
#             print(f"Khoảng cách = {dist:.4f}, Ngưỡng mặc định = {thr:.4f}, Ngưỡng custom = {CUSTOM_THRESHOLD}")
#         except Exception as e:
#             return {
#                 "success": False,
#                 "message": f"Lỗi khi so khớp khuôn mặt: {e}",
#                 "data": None
#             }
#
#     if same_person:
#         trangthai = bien_so_data.get('trangthai')
#         if trangthai is False:
#             # Cập nhật trạng thái
#             firebase_service.update_license_plate_field(bien_so_quet, True)
#             firebase_service.delete_license_plate(bien_so_quet)
#
#             # Đếm số lần ra
#             doc = xe_doc_ref.get()
#             if doc.exists:
#                 data = doc.to_dict()
#                 solanra = data.get("solanra", 0)
#             else:
#                 solanra = 0
#             solanra += 1
#             xe_doc_ref.set({"solanra": solanra}, merge=True)
#
#             # Ghi vào timeline
#             time_now = datetime.now().strftime("%H:%M:%S")
#             if timeline_ref:
#                 timeline_ref.set({
#                     "timeout": time_now,
#                     "biensoxera": url_image_detected,
#                     "khuonmatra": image_url_new_face
#                 }, merge=True)
#
#             return {
#                 "success": True,
#                 "message": "Xác thực thành công, xe được ra",
#                 "data": {
#                     "bien_so": bien_so_quet,
#                     "anh_xe_ra": url_image_detected,
#                     "timeIn": timeIn,
#                     "anh_xe_vao": url_xevao,
#                     "mat_vao": url_khuonmatvao,
#                     "time_now": time_now,
#                     "mat_ra": image_url_new_face
#                 }
#             }
#         else:
#             # Xe đã ra trước đó
#             firebase_service.update_canhbao(bien_so_quet, True)
#             return {
#                 "success": True,
#                 "warning": True,
#                 "message": "Xe đã ra trước đó, đã gửi cảnh báo",
#                 "data": {"bien_so": bien_so_quet}
#             }
#     else:
#         return {
#             "success": False,
#             "message": "Khuôn mặt không khớp với người vào",
#             "data": None
#         }
#
#
#
#
