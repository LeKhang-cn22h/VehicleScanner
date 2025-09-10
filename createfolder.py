import os
import random
import shutil

# Đường dẫn gốc
dataset_path = "datacharlabel"
images_path = os.path.join(dataset_path, "images")
labels_path = os.path.join(dataset_path, "labels")

# Đường dẫn lưu output
output_path = "dataset_split"
os.makedirs(output_path, exist_ok=True)

# Tạo các thư mục con train/val/test
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(output_path, "images", split), exist_ok=True)
    os.makedirs(os.path.join(output_path, "labels", split), exist_ok=True)

# Lấy danh sách file ảnh
all_images = [f for f in os.listdir(images_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
random.shuffle(all_images)

# Tỉ lệ chia
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

train_count = int(len(all_images) * train_ratio)
val_count = int(len(all_images) * val_ratio)

train_files = all_images[:train_count]
val_files = all_images[train_count:train_count+val_count]
test_files = all_images[train_count+val_count:]

def copy_files(file_list, split):
    for img_file in file_list:
        label_file = os.path.splitext(img_file)[0] + ".txt"

        # Copy image
        shutil.copy(os.path.join(images_path, img_file),
                    os.path.join(output_path, "images", split, img_file))

        # Copy label nếu có
        if os.path.exists(os.path.join(labels_path, label_file)):
            shutil.copy(os.path.join(labels_path, label_file),
                        os.path.join(output_path, "labels", split, label_file))

# Copy theo từng split
copy_files(train_files, "train")
copy_files(val_files, "val")
copy_files(test_files, "test")

print(f"Hoàn thành chia dữ liệu: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")
