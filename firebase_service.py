from firebase_admin import credentials, db
import firebase_admin
import os

class FirebaseService:
    def __init__(self):
        if not firebase_admin._apps:
            # Lấy đường dẫn tuyệt đối tới file json cùng cấp
            dir_path = os.path.dirname(os.path.abspath(__file__))
            cred_path = os.path.join(dir_path, "serviceAccountKey.json")

            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred, {
                'databaseURL': 'https://tramxeuth-default-rtdb.firebaseio.com/'
            })

    def get_all_license_plates(self):
        """
        Trả về danh sách tất cả các biển số trong 'biensotrongbai'
        """
        ref = db.reference("biensotrongbai")
        data = ref.get()
        return list(data.keys()) if data else []

    def get_license_plate_data(self, plate):
        """
        Lấy toàn bộ dữ liệu của một biển số từ nhánh 'license_plates'
        """
        ref = db.reference(f'biensotrongbai/{plate}')
        return ref.get()

    def update_license_plate_field(self, plate,value):

        ref = db.reference(f'trangthaicong')  # ✅ Cùng cấp với biensotrongbai
        ref.set(value)
    def update_canhbao(self,plate,value):
        ref = db.reference(f'/biensotrongbai/{plate}/canhbao')
        ref.set(value)

    def delete_license_plate(self, plate):
        ref = db.reference(f'/biensotrongbai/{plate}')
        ref.delete()
