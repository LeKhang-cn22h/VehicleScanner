import tkinter as tk
import subprocess
import sys
import cv2
from io import BytesIO
from firebase_service import FirebaseService
from PIL import Image, ImageTk
from xeVao import run_license_scan
import requests
from tkinter import messagebox
from firebase_admin import firestore
from test import run_license_scan_ra
# --- Hàm chạy file Python song song ---
def run_file(file_path):
    try:
        subprocess.Popen([sys.executable, file_path])
    except Exception as e:
        tk.messagebox.showerror("Lỗi", f"Không thể chạy file {file_path}:\n{e}")


def handle_image(image_path_or_url):
    try:
        if image_path_or_url.startswith("http"):  # Nếu là URL
            response = requests.get(image_path_or_url, stream=True)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
        else:  # Nếu là file local
            img = Image.open(image_path_or_url)

        img = img.resize((250, 150), Image.LANCZOS)
        tk_img = ImageTk.PhotoImage(img)
        return tk_img
    except Exception as e:
        print(f"[ERROR] Không load được ảnh: {e}")
        return None

def canvas_image(window, row, column=4, rowspan=1, columnspan=1, w=400, h=350):
    canvas = tk.Canvas(window, width=w, height=h, bg="gray", highlightthickness=1, highlightbackground="black")
    canvas.grid(row=row, column=column, rowspan=rowspan, columnspan=columnspan, padx=10, pady=10)

    # Thêm chữ "Chưa có dữ liệu" vào đúng giữa canvas
    # canvas.create_text(
    #     w // 2, h // 2,
    #     text="Chưa có dữ liệu",
    #     font=("Arial", 16),
    #     fill="black",
    #     anchor="center"
    # )

    return canvas

def load_image(canvas, img_path=""):
    canvas.delete("all")
    canvas.update_idletasks()
    w, h =  400,350

    if img_path:
        try:
            if img_path.startswith("http"):  # Nếu là URL thì tải về
                response = requests.get(img_path)
                response.raise_for_status()
                img_pil = Image.open(BytesIO(response.content))
            else:  # Nếu là file local
                img_pil = Image.open(img_path)

            # Resize cho vừa canvas
            img_pil = img_pil.resize((w, h), Image.LANCZOS)
            tk_img = ImageTk.PhotoImage(img_pil)

            canvas.create_image(w // 2, h // 2, image=tk_img, anchor="center")
            canvas.image = tk_img
        except Exception as e:
            canvas.create_text(w // 2, h // 2, text=f"Lỗi ảnh: {e}", font=("Arial", 12), fill="red")
    else:
        canvas.create_text(w // 2, h // 2, text="Chưa có dữ liệu", font=("Arial", 16), fill="black")

    canvas.update()

def show_plate_on_canvas(canvas, best_plate):
    # Xoá nội dung cũ (ảnh hoặc chữ) trên canvas
    canvas.delete("all")
    canvas.update_idletasks()  # bắt buộc cập nhật trước khi lấy size
    # w = canvas.winfo_width()
    # h = canvas.winfo_height()
    w = 400
    h = 350
    # if best_plate is None:
    #     canvas.create_text(
    #         w // 2, h // 2,
    #         text="Không có dữ liệu",
    #         font=("Arial", 20),
    #         fill="black",
    #         anchor="center"
    #     )
    #     return

    img_rgb = cv2.cvtColor(best_plate, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb).resize((w, h), Image.LANCZOS)
    tk_img = ImageTk.PhotoImage(img_pil)
    canvas.create_image(w // 2, h // 2, image=tk_img, anchor="center")
    canvas.image = tk_img
    canvas.update()


def label_custom_text(window, title, content, row, column,
                      rowspan=1, columnspan=1, width=25, height=3,
                      content_color="black"):
    text_widget = tk.Text(window, width=width, height=height,
                          font=("Arial", 16), cursor="arrow")
    text_widget.grid(row=row, column=column,
                     rowspan=rowspan, columnspan=columnspan,
                     padx=5, pady=3)

    # Title (đen, in đậm)
    text_widget.insert("end", title + ":\n", ("bold",))
    text_widget.tag_config("bold", font=("Arial", 16, "bold"), foreground="black")

    # Content (màu có thể thay đổi)
    text_widget.insert("end", content, ("content",))
    text_widget.tag_config("content", foreground=content_color)

    # Căn giữa dòng 2 trở đi
    text_widget.tag_add("center", "2.0", "end")
    text_widget.tag_config("center", justify="center")

    text_widget.config(state="disabled", bg="gray")
    return text_widget


def btn_quet_xe_vao():
    # Gọi hàm lấy dữ liệu
    plate_img, face_img,timeIn,plate_text = run_license_scan( window)

    # Hiển thị ảnh biển số lên canvas
    load_image(canvas_plate_url_vao, img_path=plate_img)

    # Hiển thị ảnh khuôn mặt lên canvas
    load_image(canvas_face_url_vao, img_path=face_img)
    label_custom_text(window, "Biển số xe", plate_text,
                      row=1, column=2, rowspan=1, columnspan=1,
                      width=25, height=3)
    label_custom_text(window, "Thời gian vào", timeIn,
                      row=0, column=1, rowspan=1, columnspan=1,
                      width=25, height=3)
    label_custom_text(window, "Thông báo", "Quét xe vào thành công",
                      row=0, column=2, width=25, height=3,
                      content_color="cyan")
    window.after(5000, clear_data)


def btn_quet_xe_ra():
    result = run_license_scan_ra(window)

    if  result.get("success"):
        label_custom_text(window, "Thông báo", result["message"],
                          row=0, column=2, width=25, height=3,
                          content_color="red")
        return
    if result.get("warning"):
        data = result.get("data", {})
        bien_so = data.get("bien_so")
        anh_xe_ra = data.get("anh_xe_ra")  # có thể None nếu chưa ra
        timeIn = data.get("timeIn")
        anh_xe_vao = data.get("anh_xe_vao")
        mat_vao = data.get("mat_vao")
        timeOut = data.get("time_now")
        mat_ra = data.get("mat_ra")

        # Load hình luôn, kể cả hết lượt
        if anh_xe_vao:
            load_image(canvas_plate_url_vao, img_path=anh_xe_vao)
        if mat_vao:
            load_image(canvas_face_url_vao, img_path=mat_vao)
        if anh_xe_ra:
            load_image(canvas_plate_url_ra, img_path=anh_xe_ra)
        if mat_ra:
            load_image(canvas_face_url_ra, img_path=mat_ra)

        # Hiển thị cảnh báo
        label_custom_text(window, "Thông báo", result["message"],
                          row=0, column=2, width=25, height=3,
                          content_color="red")

        # Hộp thoại Yes/No
        confirm = messagebox.askyesno("Cảnh báo", result["message"] + "\nBạn có muốn tiếp tục xử lý không?")
        if confirm:
            firebase_service = FirebaseService()
            firebase_service.update_license_plate_field(bien_so, True)
            firebase_service.delete_license_plate(bien_so)

        window.after(5000, clear_data)
        return

    data = result["data"]
    bien_so = data.get("bien_so") or data.get("biensoxe")
    anh_xe_ra = data["anh_xe_ra"]
    timeIn = data["timeIn"]
    anh_xe_vao = data["anh_xe_vao"]
    mat_vao = data["mat_vao"]
    timeOut = data["time_now"]
    mat_ra = data["mat_ra"]

    load_image(canvas_plate_url_vao, img_path=anh_xe_vao)
    load_image(canvas_face_url_vao, img_path=mat_vao)

    load_image(canvas_plate_url_ra, img_path=anh_xe_ra)
    load_image(canvas_face_url_ra, img_path=mat_ra)

    label_custom_text(window, "Biển số xe", bien_so,
                      row=1, column=2, rowspan=1, columnspan=1,
                      width=25, height=3)
    label_custom_text(window, "Thời gian vào", timeIn,
                      row=0, column=1, rowspan=1, columnspan=1,
                      width=25, height=3)
    label_custom_text(window, "Thời gian ra", timeOut,
                      row=0, column=3, rowspan=1, columnspan=1,
                      width=25, height=3)
    label_custom_text(window, "Thông báo", result["message"],
                      row=0, column=2, width=25, height=3,
                      content_color="cyan")

    window.after(5000, clear_data)


def clear_data():
    # Xóa canvas (ảnh)
    canvas_plate_url_vao.delete("all")
    canvas_face_url_vao.delete("all")
    canvas_plate_url_ra.delete("all")
    canvas_face_url_ra.delete("all")
    # Xóa các label hiển thị
    label_custom_text(window, "Biển số xe", "",
                      row=1, column=2, rowspan=1, columnspan=1,
                      width=25, height=3)
    label_custom_text(window, "Thời gian vào", "",
                      row=0, column=1, rowspan=1, columnspan=1,
                      width=25, height=3)
    label_custom_text(window, "Thông báo", "",
                      row=0, column=2, width=25, height=3)
def on_close():
    print("Đang thoát chương trình...")
    window.destroy()   # đóng cửa sổ
    sys.exit(0)        # thoát hẳn chương trình (nếu cần)

# --- Tạo cửa sổ chính ---
window = tk.Tk()
window.title("Bãi giữ xe thông minh")

# Thông báo
label_bien_so_xe = label_custom_text(
    window=window,
    title="BSX",
    content="---",
    row=1,
    column=2
)

label_thanh_tien = label_custom_text(window=window, title="Thành tiền", content="5000", row=0, column=4)

label_thoi_gioi_vao = label_custom_text(
    window=window,
    title="Thời gian vào",
    content="---",
    row=0,
    column=1
)
label_thoi_gioi_ra = label_custom_text(
    window=window,
    title="Thời gian ra",
    content="---",
    row=0,
    column=3
)
label_thong_bao = label_custom_text(window=window, title="Thông báo", content="---", row=0, column=2)
# Row 2: plate_url_vao
canvas_plate_url_vao = canvas_image(window=window, row=2, column=0, columnspan=2)

# Row 3: face_url_vao
canvas_face_url_vao = canvas_image(window=window, row=3, column=0, columnspan=2)



# hinh plate_url ra
canvas_plate_url_ra = canvas_image(window=window, row=2, column=3)
# hinh face_url ra
canvas_face_url_ra = canvas_image(window=window, row=3, column=3)

# --- Nút bấm ---
btn1 = tk.Button(
    window,
    text="Quét xe máy vào",
    font=("Arial", 16),
    command=btn_quet_xe_vao,
    width=20,
    height=5,
    bg='yellow'
)
btn1.grid(row=2, column=2, padx=10, pady=(10, 30))  # pady trên 10, dưới 30

btn2 = tk.Button(
    window,
    text="Quét xe máy ra",
    font=("Arial", 16),
    command=btn_quet_xe_ra,
    width=20,
    height=5,
    bg='yellow'
)
btn2.grid(row=3, column=2, padx=10, pady=(30, 10))  # pady trên 30, dưới 10


window.state('zoomed')
window.protocol("WM_DELETE_WINDOW", on_close)
window.mainloop()