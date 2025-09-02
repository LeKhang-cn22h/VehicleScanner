import cv2
import time
from deepface import DeepFace

# ==== Tham số ====
DELAY = 5  # chụp cách nhau 5 giây
MIN_FACE_SIZE = 100
BLUR_VAR_MIN = 80
IOU_MAX_SAME = 0.7
CUSTOM_THRESHOLD = 0.35

# ==== Hàm phụ ====
def variance_of_laplacian(img_gray):
    return cv2.Laplacian(img_gray, cv2.CV_64F).var()

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
    yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    inter = interW * interH
    areaA = boxA[2]*boxA[3]
    areaB = boxB[2]*boxB[3]
    union = areaA + areaB - inter + 1e-6
    return inter / union

# Haar chỉ để detect & crop ROI
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)

captured_paths = []
count = 0
last_capture_time = 0.0
last_box = None
last_result_text = ""

while True:
    ok, frame = cap.read()
    if not ok:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Vẽ bbox
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Tự chụp 2 ảnh cách nhau DELAY giây
    if len(faces) > 0 and len(captured_paths) < 2 and (time.time() - last_capture_time) >= DELAY:
        (x, y, w, h) = max(faces, key=lambda b: b[2]*b[3])
        if w >= MIN_FACE_SIZE and h >= MIN_FACE_SIZE:
            if last_box is None or iou((x,y,w,h), last_box) <= IOU_MAX_SAME:
                face_roi = frame[y:y+h, x:x+w]
                roi_gray = gray[y:y+h, x:x+w]
                if variance_of_laplacian(roi_gray) >= BLUR_VAR_MIN:
                    filename = f"face_{count}.jpg"
                    cv2.imwrite(filename, face_roi)
                    captured_paths.append(filename)
                    last_capture_time = time.time()
                    last_box = (x, y, w, h)
                    print(f"📸 Đã chụp: {filename}")
                    count += 1

    # Khi đủ 2 ảnh thì so sánh
    if len(captured_paths) == 2:
        try:
            result = DeepFace.verify(
                img1_path=captured_paths[0],
                img2_path=captured_paths[1],
                model_name="ArcFace",
                detector_backend="retinaface",
                distance_metric="cosine",
                align=True,
                enforce_detection=True
            )
            print("📊 Kết quả DeepFace:", result)

            dist = float(result.get("distance", 1.0))
            thr = float(result.get("threshold", 0.0))

            # check với ngưỡng custom
            verified_custom = dist <= CUSTOM_THRESHOLD
            final_verified = result.get("verified", False) and verified_custom

            if final_verified:
                last_result_text = f"✅ Cùng 1 người | dist={dist:.3f}"
                print("✅ Kết luận: CÙNG 1 NGƯỜI")
            else:
                last_result_text = f"❌ Khác người | dist={dist:.3f}"
                print("❌ Kết luận: KHÁC NGƯỜI")

        except Exception as e:
            last_result_text = f"⚠️ Lỗi: {str(e)}"
            print(last_result_text)

        # Hiện kết quả 2 giây rồi thoát
        cv2.putText(frame, last_result_text, (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        cv2.imshow("Camera", frame)
        cv2.waitKey(2000)
        break

    # Overlay thông tin
    cv2.putText(frame, f"Auto capture every {DELAY}s when face is clear", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    if last_result_text:
        cv2.putText(frame, last_result_text, (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    cv2.imshow("Camera", frame)
    # Nhấn q để thoát sớm
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
