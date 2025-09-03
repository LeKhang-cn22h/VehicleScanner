from ultralytics import YOLO
import cv2
import numpy as np
from sklearn.cluster import KMeans
import webcolors


# ==============================
# Hàm tìm tên màu gần nhất
# ==============================
def closest_color(requested_color):
    min_colors = {}
    for hex_value, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(hex_value)
        rd = (r_c - requested_color[0]) ** 2
        gd = (g_c - requested_color[1]) ** 2
        bd = (b_c - requested_color[2]) ** 2
        min_colors[(rd + gd + bd)] = name
    return min_colors[min(min_colors.keys())]


def get_color_name(rgb_tuple):
    try:
        return webcolors.rgb_to_name(rgb_tuple)
    except ValueError:
        return closest_color(rgb_tuple)


# ==============================
# Hàm tìm màu chủ đạo
# ==============================
def dominant_color(image, k=3):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_rgb = img_rgb.reshape((-1, 3))

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(img_rgb)

    colors = kmeans.cluster_centers_
    counts = np.bincount(kmeans.labels_)
    dominant = colors[np.argmax(counts)]
    return tuple(dominant.astype(int))  # (R,G,B)


# ==============================
# Hàm chính: Trả về list tên màu chủ đạo
# ==============================
def get_dominant_car_color(img_path, model_path="yolov8n.pt"):
    model = YOLO(model_path)
    results = model(img_path)
    img = cv2.imread(img_path)

    for r in results:
        for box, cls in zip(r.boxes.xyxy, r.boxes.cls):
            if int(cls) == 2:  # class 2 = car
                x1, y1, x2, y2 = map(int, box)
                car_roi = img[y1:y2, x1:x2]

                if car_roi.size == 0:
                    continue

                color_rgb = dominant_color(car_roi)
                color_name = get_color_name(color_rgb)

                return color_name   # trả về ngay màu đầu tiên

    return None  # nếu không có xe

