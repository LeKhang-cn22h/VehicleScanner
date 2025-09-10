import json
import sys
import tkinter as tk
import subprocess
from XeMay.giaoDienXeMay import run_main_xe_may
from xeoto.giaoDienXeOto import run_main_o_to

def on_close(window):
    print("Đang thoát chương trình...")
    window.destroy()   # đóng cửa sổ
    sys.exit(0)        # thoát hẳn chương trình (nếu cần)

def btn_Xe_May(window):
    window.destroy()
    run_main_xe_may()

def btn_O_to(window):
    window.destroy()
    run_main_o_to()

# --- Tạo cửa sổ chính ---
window = tk.Tk()
window.title("Bãi giữ xe thông minh")
window.geometry("500x300")
# Thông báo
label_title = tk.Label(window, text="Chọn loại xe để quét", font=("Arial", 20, "bold"))
label_title.pack(padx=10, pady=10)

# --- Nút bấm ---
btn1 = tk.Button(
    window,
    text="Xe máy",
    font=("Arial", 17, "bold"),
    command=lambda: btn_Xe_May(window=window),
    width=13,
    height=2,
    bg='lightblue'
)
btn1.pack(padx=10, pady=10)

btn2 = tk.Button(
    window,
    text="Ô tô",
    font=("Arial", 17, "bold"),
    command=lambda: btn_O_to(window=window),
    width=13,
    height=2,
    bg='lightblue'
)
btn2.pack(padx=10, pady=10)

window.protocol("WM_DELETE_WINDOW", lambda: on_close(window=window))
window.mainloop()