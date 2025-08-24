# cloudinary_config.py
import cloudinary
import cloudinary.uploader
import os
from dotenv import load_dotenv

# Load biến môi trường từ file .env
load_dotenv()

# Cấu hình Cloudinary
cloudinary.config(
    cloud_name=os.getenv('CLOUD_NAME'),
    api_key=os.getenv('CLOUD_API_KEY'),
    api_secret=os.getenv('CLOUD_API_SECRET')
)

def upload_image_to_cloudinary(image_path):
    """
    Tải ảnh lên Cloudinary và trả về URL ảnh.
    :param image_path: đường dẫn ảnh trên máy local.
    :return: URL ảnh trên Cloudinary nếu thành công, None nếu lỗi.
    """
    try:
        result = cloudinary.uploader.upload(image_path)
        return result.get("secure_url")
    except Exception as e:
        print(f"❌ Lỗi khi upload ảnh: {e}")
        return None
