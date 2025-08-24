import tkinter as tk
import subprocess
import sys

def run_file1():
    subprocess.run([sys.executable, "hander_QR.py"])

def run_file2():
    subprocess.run([sys.executable, "main1.py"])

def run_file3():
    subprocess.run([sys.executable, "main2.py"])

# Tạo cửa sổ chính
window = tk.Tk()
window.title("Bãi giữ xe thông minh")
window.geometry("300x200")

btn1 = tk.Button(window, text="Quét QR", command=run_file1, width=20, height=2)
btn2 = tk.Button(window, text="Quét biển số lúc xe vào", command=run_file2, width=20, height=2)
btn3 = tk.Button(window, text="Quét biển số lúc xe ra", command=run_file3, width=20, height=2)

btn1.pack(pady=10)
btn2.pack(pady=10)
btn3.pack(pady=10)

window.mainloop()
