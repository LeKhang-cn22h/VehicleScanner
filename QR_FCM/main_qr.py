# main.py
import subprocess
import time
from hander_QR import scan_qr_code
import firebase_admin
from firebase_admin import credentials, firestore, messaging

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

def send_notification_to_all_tokens(message_body):

    docs = db.collection("thongtinadmin").stream()
    for doc in docs:
        fcm_tokens = doc.to_dict().get("fcmTokens", [])
        for token in fcm_tokens:
            try:
                message = messaging.Message(
                    notification=messaging.Notification(
                        title="Thông điệp từ QR",
                        body=message_body
                    ),
                    token=token
                )
                response = messaging.send(message)
                print(f"Đã gửi thông báo tới token {token[:10]}...: {response}")
            except Exception as e:
                print(f"Lỗi gửi tới token {token}: {e}")
   

print("Đang quét mã QR...")
qr_value = scan_qr_code()

if qr_value:
    print(f"Quét được: {qr_value}")
    send_notification_to_all_tokens(qr_value)
else:
    print(" Không quét được mã QR.")

