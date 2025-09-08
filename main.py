import tkinter as tk
import subprocess
import sys
import json

with open("state.json", "r") as f:
    data = json.load(f)
    autoData = data.get("AUTO")

def run_file1():
    subprocess.run([sys.executable, "QR_FCM/main_qr.py"])

def run_file2():
    subprocess.run([sys.executable, "main1.py"])

def run_file3():
    subprocess.run([sys.executable, "main2.py"])

def click_auto():
    global autoData
    autoData = not autoData
    btnAuto.config(text=f"Auto: {'Bật' if autoData else 'Tắt'}")
    with open("state.json", "w") as f:
        json.dump({"AUTO": autoData}, f)

def on_closing():
    with open("state.json", "w") as f:
        json.dump({"AUTO": False}, f)
    window.destroy()
    
# Tạo cửa sổ chính
window = tk.Tk()
window.title("Bãi giữ xe thông minh")
window.geometry("300x300")

window.protocol("WM_DELETE_WINDOW", on_closing)

btn1 = tk.Button(window, text="Quét QR", command=run_file1, width=20, height=2)
btn2 = tk.Button(window, text="Quét biển số lúc xe vào", command=run_file2, width=20, height=2)
btnAuto = tk.Button(window, text=f"Auto: {'Bật' if autoData else 'Tắt'}", command=click_auto, width=20, height=2)
btn3 = tk.Button(window, text="Quét biển số lúc xe ra", command=run_file3, width=20, height=2)

btn1.pack(pady=10)
btn2.pack(pady=10)
btnAuto.pack(pady=10)
btn3.pack(pady=10)

window.mainloop()
