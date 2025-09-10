from _datetime import timedelta
from datetime import datetime

import os
import firebase_admin
from firebase_admin import credentials, firestore

# Lấy thư mục gốc của file firebase_hander.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Ghép đường dẫn tuyệt đối tới serviceAccountKey.json
cred_path = os.path.join(BASE_DIR, "serviceAccountKey.json")

# Khởi tạo Firebase App (nếu chưa có)
if not firebase_admin._apps:
    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred, {
                'databaseURL': 'https://tramxeuth-default-rtdb.firebaseio.com/'
            })

db = firestore.client()

def create_time_expired(timestamp):
    # Lấy tất cả documents trong collection
    docs = db.collection("quydinh").stream()
    field_gioihangio = "gioihangio"
    field_gioihanngay = "gioihanngay"
    dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
    for doc in docs:
        doc_data = doc.to_dict()
        if doc_data[field_gioihanngay] is not None:
            dt = dt.replace(hour = 23, minute = 59, second = 59)
        else:
            dt += timedelta(hours = doc_data[field_gioihangio])

    new_str = dt.strftime("%Y-%m-%d %H:%M:%S")
    return new_str

def get_field_from_all_docs(collection_name, field_name):

    field_values = []

    # Lấy tất cả documents trong collection
    docs = db.collection(collection_name).stream()

    for doc in docs:
        doc_data = doc.to_dict()
        if field_name in doc_data:
            field_values.append(doc_data[field_name])
        else:
            print(f"Document {doc.id} không có field {field_name}")

    return field_values
# Sử dụng hàm
collection_name = "thongtindangky"  # Thay bằng tên collection của bạn
field_name = "biensoxe"
field_name_phu = "biensophu"
values = get_field_from_all_docs(collection_name, field_name)
values_phu = get_field_from_all_docs(collection_name, field_name_phu)
print(f"Các giá trị của field {field_name}:")
print(f"Các giá trị của field phụ {field_name_phu}:")
for value in values:
    print(value)