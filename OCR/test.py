import torch
from PIL import Image
from torchvision import transforms  # BẠN QUÊN DÒNG NÀY
from crnn_model import CRNN  # Model bạn đã định nghĩa
from utils import ctc_decode, alphabet  # Hàm decode và bảng chữ cái

# Cấu hình thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tải model đã huấn luyện
model = CRNN(32, num_classes=len(alphabet) + 1).to(device)
model.load_state_dict(torch.load("../../crnn_best.pth", map_location=device))
model.eval()

# Tiền xử lý ảnh đầu vào
transform = transforms.Compose([
    transforms.Resize((32, 128)),
    transforms.ToTensor()
])


# Hàm dự đoán văn bản từ ảnh
def predict_image(image_path):
    image = Image.open(image_path).convert("L")  # chuyển ảnh sang grayscale
    image = transform(image).unsqueeze(0).to(device)  # [1, 1, 32, 128]

    with torch.no_grad():
        preds = model(image)  # [T, B, C] ← T: seq_len, B: batch_size
        preds = preds.permute(1, 0, 2)  # [B, T, C] ← chuẩn cho CTC decode

        output = torch.nn.functional.log_softmax(preds, dim=2)
        _, pred_indices = output.max(2)  # [B, T]
        decoded_text = ctc_decode(output)

    return decoded_text

# Thử nghiệm với một ảnh
img_path = "thresh.jpg"
text = predict_image(img_path)
print("Kết quả nhận dạng:", text)
