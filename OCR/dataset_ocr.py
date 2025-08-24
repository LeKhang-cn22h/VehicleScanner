import os
from PIL import Image
from torch.utils.data import Dataset

class OCRDataset(Dataset):
    def __init__(self, img_folder, label_file, transform=None):
        self.img_folder = img_folder
        self.transform = transform
        with open(label_file, 'r', encoding='utf-8') as f:
            self.samples = [line.strip().split(maxsplit=1) for line in f]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, label = self.samples[idx]
        img_name = os.path.basename(img_name)  # lấy đúng tên file
        img_path = os.path.join(self.img_folder, img_name)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Không tìm thấy ảnh: {img_path}")

        image = Image.open(img_path).convert("L")

        if self.transform:
            image = self.transform(image)

        return image, label

