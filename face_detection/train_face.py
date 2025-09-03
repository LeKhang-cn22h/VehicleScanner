import cv2
from deepface import DeepFace
from PIL import Image
import tempfile
import os
import time
from cloudinary_config import upload_image_to_cloudinary


def capture_face_and_upload():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Không thể mở camera")
        return None

    result_url = None
    last_capture_time = 0
    capture_interval = 10  # giây

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không thể chụp ảnh từ camera")
            break

        cv2.imshow("Camera", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        # Kiểm tra khuôn mặt mỗi 5 giây
        if time.time() - last_capture_time >= capture_interval:
            last_capture_time = time.time()
            try:
                faces = DeepFace.extract_faces(frame, detector_backend="opencv")
                if len(faces) > 0:
                    print("✅ Phát hiện khuôn mặt!")

                    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
                        img_pil.save(temp_file.name)
                        result = upload_image_to_cloudinary(temp_file.name, folder="KhuonMatVao")

                    if os.path.exists(temp_file.name):
                        os.remove(temp_file.name)

                    result_url = result.get("secure_url") if result else None
                    break

            except Exception:
                pass

    cap.release()
    cv2.destroyAllWindows()
    return result_url
