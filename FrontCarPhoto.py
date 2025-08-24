import cv2
import time
import os
import winsound
from cloudinary_config import upload_image_to_cloudinary

def capture_and_upload_front_image(filename="front_car.jpg"):
    time.sleep(2)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Không thể mở camera")
        return None

    print("Mở camera xem trước trong 5 giây...")
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không thể đọc dữ liệu từ camera")
            break

        cv2.imshow("Camera Preview - Nhấn 'q' để thoát sớm", frame)

        # Đợi 5 giây hoặc nhấn 'q' để thoát
        if time.time() - start_time > 5 or cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Chụp ảnh cuối cùng
    if ret:
        winsound.Beep(1000, 150)
        cv2.imwrite(filename, frame)
        print("Đã chụp ảnh:", filename)

        try:
            res = upload_image_to_cloudinary(filename)  # ✅ đúng tên hàm bạn import
            if os.path.exists(filename):
                os.remove(filename)
            url = res
            print("Đã upload ảnh lên Cloudinary:", url)
            if os.path.exists(filename):
                os.remove(filename)
                print("Đã xóa ảnh local:", filename)
            return url
        except Exception as e:
            print(f"Lỗi upload ảnh: {e}")
            return None
    else:
        print("Không thể chụp ảnh")

    cap.release()
    cv2.destroyAllWindows()

# # Gọi hàm
# if __name__ == "__main__":
#     capture_and_upload_front_image()
