import tkinter as tk
import subprocess
import sys
import json
from functools import partial

# --- Đọc file state.json nếu cần ---
try:
    with open("state.json", "r") as f:
        data = json.load(f)
        autoData = data.get("AUTO", False)
except:
    autoData = False

# --- Hàm chạy file Python song song ---
def run_file(file_path):
    try:
        subprocess.Popen([sys.executable, file_path])
    except Exception as e:
        tk.messagebox.showerror("Lỗi", f"Không thể chạy file {file_path}:\n{e}")

# --- Tạo cửa sổ chính ---
window = tk.Tk()
window.title("Bãi giữ xe thông minh")
window.geometry("300x250")

# --- Nút bấm ---
btn1 = tk.Button(window, text="quét xe máy vào", command=partial(run_file, "xeVao.py"), width=25, height=2)
btn2 = tk.Button(window, text="Quét xe máy ra", command=partial(run_file, "xeRa.py"), width=25, height=2)


btn1.pack(pady=10)
btn2.pack(pady=10)
window.mainloop()