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

def upload_image_to_cloudinary(image_path, folder=None):
    options = {}
    if folder:
        options["folder"] = folder
    return cloudinary.uploader.upload(image_path, **options)

