import tkinter as tk
import subprocess
import sys
import json
from functools import partial
import cv2
from PIL import Image, ImageTk
from xeoto.xeVao import run_license_scan
import xeoto.xeRa as xeRa
import xeoto.utils as utils
import requests
from io import BytesIO
import numpy as np
from datetime import datetime
import threading
import os

# # --- Hàm chạy file Python song song ---
# def run_file(file_path):
#     try:
#         subprocess.Popen([sys.executable, file_path])
#     except Exception as e:
#         tk.messagebox.showerror("Lỗi", f"Không thể chạy file {file_path}:\n{e}")

# --- Đọc file state.json nếu cần ---
# Đường dẫn tới thư mục chứa exe hoặc script
# Nếu file không tồn tại → tạo
with open("state.json", "w") as f:
    json.dump({"AUTO": False}, f)

try:
    with open("state.json", "r") as f:
        data = json.load(f)
        autoData = data.get("AUTO", False)
except FileNotFoundError:
    autoData = False

def handle_image(image):
    """Trả về ImageTk.PhotoImage từ local path hoặc URL"""
    if isinstance(image, str) and image.startswith(("http://", "https://")):
        # Nếu là URL → tải ảnh về trước
        resp = requests.get(image, stream=True)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
    else:
        # Nếu là file local
        img = Image.open(image).convert("RGB")

    # Resize về (250, 150)
    img = img.resize((250, 150), Image.LANCZOS)
    return ImageTk.PhotoImage(img)

def canvas_image(window, row, column=4, rowspan=1, columnspan=1):
    canvas = tk.Canvas(window, width=250, height=150, bg="gray", highlightthickness=1, highlightbackground="black")
    canvas.grid(row=row, column=column, rowspan=rowspan, columnspan=columnspan,padx=10, pady=10)

    # Thêm chữ "Chưa có dữ liệu" vào giữa canvas
    canvas.create_text(
        125, 75,  # toạ độ (x= , y= ) là giữa canvas
        text="Chưa có dữ liệu",
        font=("Arial", 16),
        fill="black"
    )

    return canvas

def canvas_image_default(canvas):
    # Clear nội dung cũ (nếu cần)
    canvas.delete("all")

    # Cấu hình lại canvas (nếu muốn)
    canvas.config(width=250, height=150, bg="gray", highlightthickness=1, highlightbackground="black")

    # Thêm chữ "Chưa có dữ liệu" vào giữa canvas
    canvas.create_text(
        125, 75,  # giữa canvas
        text="Chưa có dữ liệu",
        font=("Arial", 16),
        fill="black"
    )
    canvas.update()

def load_image(canvas, img_path=""):
    # Xoá hết nội dung cũ trên canvas
    canvas.delete("all")

    if img_path:
        tk_img = handle_image(img_path)  # Trả về ImageTk.PhotoImage
        # Vẽ ảnh vào giữa canvas
        canvas.create_image(125, 75, image=tk_img)
        canvas.image = tk_img  # giữ tham chiếu
    else:
        # Hiển thị chữ "Chưa có dữ liệu"
        canvas.create_text(
            125, 75,
            text="Chưa có dữ liệu",
            font=("Arial", 16),
            fill="black"
        )

    canvas.update()

def show_plate_on_canvas(canvas, best_plate):
    # Xóa nội dung cũ trên canvas
    canvas.delete("all")

    if best_plate is None:
        canvas.create_text(
            canvas.winfo_width() // 2,
            canvas.winfo_height() // 2,
            text="❌ Không có dữ liệu",
            font=("Arial", 20),
            fill="black"
        )
        return

    img_pil = None

    # Nếu best_plate là numpy array (ảnh OpenCV)
    if isinstance(best_plate, np.ndarray):
        try:
            img_rgb = cv2.cvtColor(best_plate, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
        except Exception as e:
            print("❌ Lỗi convert numpy array:", e)
            return

    # Nếu best_plate là URL
    elif isinstance(best_plate, str) and best_plate.startswith(("http://", "https://")):
        try:
            resp = requests.get(best_plate, stream=True)
            resp.raise_for_status()
            img_pil = Image.open(BytesIO(resp.content)).convert("RGB")
        except Exception as e:
            print("❌ Lỗi load ảnh từ URL:", e)
            return

    # Nếu best_plate là đường dẫn local
    elif isinstance(best_plate, str):
        try:
            img_pil = Image.open(best_plate).convert("RGB")
        except Exception as e:
            print("❌ Lỗi load ảnh từ file:", e)
            return

    else:
        print("❌ best_plate không phải định dạng hỗ trợ:", type(best_plate))
        return

    # Resize ảnh để vừa canvas
    w, h = canvas.winfo_width(), canvas.winfo_height()
    img_pil = img_pil.resize((w, h), Image.LANCZOS)

    # Convert sang Tkinter Image
    tk_img = ImageTk.PhotoImage(img_pil)

    # Vẽ ảnh vào giữa canvas
    canvas.create_image(w // 2, h // 2, image=tk_img)
    canvas.image = tk_img  # giữ tham chiếu
    canvas.update()

def label_custom_text(window, title, content, row, column, rowspan=1, columnspan=1, width=25, height=3):
    # Tạo Text widget
    text_widget = tk.Text(window, width=width, height=height, font=("Arial", 16), cursor="arrow")
    text_widget.grid(row=row, column=column, rowspan=rowspan, columnspan=columnspan, padx=10, pady=5)

    # Chèn title và content
    text_widget.insert("end", title + ":\n")
    text_widget.insert("end", content)

    # Đánh dấu title để in đậm
    text_widget.tag_add("bold", "1.0", f"1.{len(title)+1}")  # từ ký tự 0 đến len(title)+1
    text_widget.tag_config("bold", font=("Arial", 16, "bold"), justify="left")

    text_widget.tag_add("center", "2.0", "end")  # dòng 2 trở đi
    text_widget.tag_config("center", justify="center")
    
    # Không cho chỉnh sửa text
    text_widget.config(state="disabled", bg="gray")

    return text_widget

def reset_canvas(listCanvasVao: utils.ListCanvas, listCanvasRa: utils.ListCanvas, labels_info: utils.LabelGroup):
        
    # Lấy thời gian hiện tại
    now = datetime.now()

    # Định dạng theo dd-mm-yy hh:mm:ss
    formatted_time = now.strftime("%d-%m-%y %H:%M:%S")

    # reset labels infor
    utils.update_label_content(labels_info.bien_so_xe,"Chưa có dữ liệu")
    utils.update_label_content(labels_info.thong_bao,"Chưa có dữ liệu")
    utils.update_label_content(labels_info.thanh_tien,"5000")
    utils.update_label_content(labels_info.thong_bao, formatted_time, bg="gray")

    # reset image vào
    canvas_image_default(listCanvasVao.bsx_dau)
    canvas_image_default(listCanvasVao.bsx_duoi)
    canvas_image_default(listCanvasVao.dau_xe)
    canvas_image_default(listCanvasVao.duoi_xe)
    canvas_image_default(listCanvasVao.logo)

    # reset image ra
    canvas_image_default(listCanvasRa.bsx_dau)
    canvas_image_default(listCanvasRa.bsx_duoi)
    canvas_image_default(listCanvasRa.dau_xe)
    canvas_image_default(listCanvasRa.duoi_xe)
    canvas_image_default(listCanvasRa.logo)

def btn_quet_xe_vao(listCanvasVao: utils.ListCanvas, listCanvasRa: utils.ListCanvas, labels_info: utils.LabelGroup):
    # reset all image
    reset_canvas(listCanvasVao=listCanvasVao, listCanvasRa=listCanvasRa, labels_info=labels_info)

    # thực hiện quá trình xử lí xe vào
    bien_so, image_duoi_xe, bsx_duoi, image_dau_xe, bsx_dau, image_logo = run_license_scan(label_status=labels_info.thong_bao, label_bsx=labels_info.bien_so_xe)
    
    # update ảnh xử lý xe vào
    show_plate_on_canvas(listCanvasVao.bsx_dau, best_plate=bsx_dau)
    show_plate_on_canvas(canvas=listCanvasVao.bsx_duoi, best_plate=bsx_duoi)
    load_image(canvas=listCanvasVao.dau_xe, img_path=image_dau_xe)
    load_image(canvas=listCanvasVao.duoi_xe, img_path=image_duoi_xe)
    show_plate_on_canvas(canvas=listCanvasVao.logo, best_plate=image_logo)

def btn_quet_xe_ra(window, listCanvasVao: utils.ListCanvas, listCanvasRa: utils.ListCanvas, labels_info: utils.LabelGroup):
    # reset all image
    reset_canvas(listCanvasVao=listCanvasVao, listCanvasRa=listCanvasRa, labels_info=labels_info)

    # thực hiện quá trình xử lí xe ra
    bien_so_quet, image_duoi_xe, bsx_duoi, image_dau_xe, bsx_dau, image_logo, data_xe_vao, sameImage = xeRa.run_license_scan(labels_info.thong_bao, window, labels_info.bien_so_xe)
    data_xe_vao: utils.DataXeVao = data_xe_vao
    sameImage: utils.SameImage = sameImage

    # update ảnh xe vào
    show_plate_on_canvas(listCanvasVao.bsx_dau, best_plate=data_xe_vao.bsx_dau)
    show_plate_on_canvas(canvas=listCanvasVao.bsx_duoi, best_plate=data_xe_vao.bsx_duoi)
    load_image(canvas=listCanvasVao.dau_xe, img_path=data_xe_vao.hinh_dau_xe)
    load_image(canvas=listCanvasVao.duoi_xe, img_path=data_xe_vao.hinh_duoi_xe)
    show_plate_on_canvas(canvas=listCanvasVao.logo, best_plate=data_xe_vao.logo)

    # update ảnh xử lý xe ra
    show_plate_on_canvas(listCanvasRa.bsx_dau, best_plate=bsx_dau)
    show_plate_on_canvas(canvas=listCanvasRa.bsx_duoi, best_plate=bsx_duoi)
    load_image(canvas=listCanvasRa.dau_xe, img_path=image_dau_xe)
    load_image(canvas=listCanvasRa.duoi_xe, img_path=image_duoi_xe)
    show_plate_on_canvas(canvas=listCanvasRa.logo, best_plate=image_logo)

def click_auto(btnAuto):
    global autoData
    autoData = not autoData
    btnAuto.config(text=f"Auto: {'Bật' if autoData else 'Tắt'}")
    with open("state.json", "w") as f:
        json.dump({"AUTO": autoData}, f)

def on_close(window):
    print("Đang thoát chương trình...")
    base_path = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else os.path.dirname(os.path.abspath(__file__))
    state_path = os.path.join(base_path, "state.json")
    data = {"AUTO": False}  # ví dụ cập nhật
    with open("state.json", "w") as f:
        json.dump({"AUTO": autoData}, f)
    window.destroy()   # đóng cửa sổ
    subprocess.run([sys.executable, "giaodienchinh.py"])

def btn_Qr():
    subprocess.run([sys.executable, "QR_FCM/main_qr.py"])

def run_main_o_to():

    # --- Tạo cửa sổ chính ---
    window = tk.Tk()
    window.title("Bãi giữ xe thông minh")
    window.geometry("300x250")

    # Tiêu đề
    label_vao = tk.Label(window, text="Vào", font=("Arial", 24))
    label_vao.grid(row=1, column=0, columnspan=2, padx=10, pady=10)

    label_vao = tk.Label(window, text="Ra", font=("Arial", 24))
    label_vao.grid(row=1, column=3, columnspan=2, padx=10, pady=10)

    # Thông báo
    label_bien_so_xe = label_custom_text(window=window, title="BSX", content="Chưa có dữ liệu", row=0, column=0)
    label_thong_bao = label_custom_text(window=window, title="Thông báo", content="Chưa có dữ liệu", row=0, column=1, columnspan=2)
    label_thanh_tien = label_custom_text(window=window, title="Thành tiền", content="5000", row=0, column=3)
    label_thoi_gioi_vao = label_custom_text(window=window, title="Thời gian vào", content="Chưa có dữ liệu", row=0, column=4)

    labels = utils.LabelGroup(
        bien_so_xe=label_bien_so_xe,
        thong_bao=label_thong_bao,
        thanh_tien=label_thanh_tien,
        thoi_gioi_vao=label_thoi_gioi_vao
    )

    ##############################
    #          Lúc vào           #
    ##############################

    # bsx đầu
    canvas_bsx_dau = canvas_image(window=window, row=2, column=0)
    # bsx đuôi
    canvas_bsx_duoi = canvas_image(window=window, row=2, column=1)
    # ảnh đầu xe
    canvas_dau_xe = canvas_image(window=window, row=3, column=0)
    # ảnh đuôi xe
    canvas_duoi_xe = canvas_image(window=window, row=3, column=1)
    # logo
    canvas_logo = canvas_image(window=window, row=4, column=0, columnspan=2)

    listCanvasVao = utils.ListCanvas(
        bsx_dau=canvas_bsx_dau,
        bsx_duoi=canvas_bsx_duoi,
        dau_xe=canvas_dau_xe,
        duoi_xe=canvas_duoi_xe,
        logo=canvas_logo
    )

    ##############################
    #           Lúc ra           #
    ##############################

    # bsx đầu
    canvas_bsx_dau_ra = canvas_image(window=window, row=2, column=3)
    # bsx đuôi
    canvas_bsx_duoi_ra = canvas_image(window=window, row=2, column=4)
    # ảnh đầu xe
    canvas_dau_xe_ra = canvas_image(window=window, row=3, column=3)
    # ảnh đuôi xe
    canvas_duoi_xe_ra = canvas_image(window=window, row=3, column=4)
    # logo
    canvas_logo_ra = canvas_image(window=window, row=4, column=3, columnspan=2)

    listCanvasRa = utils.ListCanvas(
        bsx_dau=canvas_bsx_dau_ra,
        bsx_duoi=canvas_bsx_duoi_ra,
        dau_xe=canvas_dau_xe_ra,
        duoi_xe=canvas_duoi_xe_ra,
        logo=canvas_logo_ra
    )

    # --- Nút bấm ---
    btn1 = tk.Button(window, text="Quét xe ô tô vào", font=("Arial", 16), command=partial(btn_quet_xe_vao, listCanvasVao, listCanvasRa, labels), width=20, height=5, bg='yellow')
    btn1.grid(row=2, column=2, padx=10, pady=10)

    btn2 = tk.Button(window, text="Quét xe ô tô ra", font=("Arial", 16), command=partial(btn_quet_xe_ra, window, listCanvasVao, listCanvasRa, labels), width=20, height=5, bg='yellow')
    btn2.grid(row=3, column=2, padx=10, pady=10)

    btnAuto = tk.Button(window, text=f"Auto: {'Bật' if autoData else 'Tắt'}", command=lambda: click_auto(btnAuto), width=20, height=3, bg='orange')
    btnAuto.grid(row=4, column=2, padx=10, pady=10)

    btn3 = tk.Button(window, text="Quét QR", command=lambda: threading.Thread(target=btn_Qr).start(), width=20, height=5, bg="lightblue")
    btn3.grid(row=1, column=2, padx=10, pady=(10, 10))  # pady trên 30, dưới 10

    window.state('zoomed')

    # Đóng bằng nút (X)
    window.protocol("WM_DELETE_WINDOW", lambda: on_close(window=window))

    window.mainloop()

# if __name__ == "__main__":
#     run_main_o_to()