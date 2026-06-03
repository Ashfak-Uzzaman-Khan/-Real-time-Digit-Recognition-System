import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk
import cv2

from webcam_digit_recognition import WebcamDigitRecognizer


recognizer = WebcamDigitRecognizer("digit_cnn_model.keras")


window = tk.Tk()
window.title("Real-time Digit Identifier")
window.geometry("900x650")
window.configure(bg="#121212")

title = Label(
    window,
    text="Real-Time Digit Identifier",
    font=("Segoe UI", 22, "bold"),
    bg="#121212",
    fg="white"
)
title.pack(pady=10)

video_label = Label(window)
video_label.pack()

info_label = Label(
    window,
    text="Digit: - | Confidence: -",
    font=("Segoe UI", 16),
    bg="#121212",
    fg="#00ffcc"
)
info_label.pack(pady=10)

status = Label(
    window,
    text="Status: Camera Off",
    font=("Segoe UI", 12),
    bg="#121212",
    fg="gray"
)
status.pack()

def start_camera():
    recognizer.cap = cv2.VideoCapture(0)
    recognizer.running = True
    status.config(text="Status: Camera Running", fg="lightgreen")
    update_frame()

def stop_camera():
    recognizer.running = False
    if recognizer.cap:
        recognizer.cap.release()
    video_label.config(image="")
    status.config(text="Status: Camera Stopped", fg="red")


def update_frame():
    if not recognizer.running:
        return

    ret, frame = recognizer.cap.read()
    if not ret:
        window.after(10, update_frame)
        return

    digit_img, bbox = recognizer.preprocess(frame)

    if digit_img is not None:
        digit, conf = recognizer.predict(digit_img)
        if digit is not None:
            info_label.config(
                text=f"Digit: {digit} | Confidence: {conf:.2f}%"
            )
            x, y, w, h = bbox
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb).resize((640, 480))
    imgtk = ImageTk.PhotoImage(img)

    video_label.imgtk = imgtk
    video_label.config(image=imgtk)

    window.after(10, update_frame)


btn_frame = tk.Frame(window, bg="#121212")
btn_frame.pack(pady=15)

Button(btn_frame, text="Start Camera", width=15,
       bg="#00aa00", fg="white", command=start_camera).grid(row=0, column=0, padx=10)

Button(btn_frame, text="Stop Camera", width=15,
       bg="#aa0000", fg="white", command=stop_camera).grid(row=0, column=1, padx=10)


window.mainloop()
