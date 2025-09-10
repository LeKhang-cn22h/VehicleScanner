# qr_scanner.py
import cv2
from pyzbar.pyzbar import decode

def scan_qr_code():
    """Quét mã QR từ camera và trả về dữ liệu chuỗi."""
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print(" Không thể mở camera.")
        return None

    scanned_data = None
    print("Đang mở camera để quét QR... Bấm 'q' để thoát.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print(" Không thể đọc từ camera.")
            break

        # Giải mã QR code trong khung hình
        for barcode in decode(frame):
            qr_data = barcode.data.decode("utf-8")
            (x, y, w, h) = barcode.rect

            # Vẽ khung và hiển thị dữ liệu
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, qr_data, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            scanned_data = qr_data
            cap.release()
            cv2.destroyAllWindows()
            return scanned_data  # giá trị để dùng

        cv2.imshow("Quét mã QR", frame)

        # Thoát thủ công
        # Kiểm tra nếu người dùng bấm 'q' hoặc đóng cửa sổ (X)
        if (cv2.waitKey(1) & 0xFF == ord('q')) or cv2.getWindowProperty("Quét mã QR", cv2.WND_PROP_VISIBLE) < 1:
            print("Đã thoát quét QR.")
            break

    cap.release()
    cv2.destroyAllWindows()
    return None
