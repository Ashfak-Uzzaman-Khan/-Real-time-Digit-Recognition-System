import tkinter as tk
from tkinter import Label, Button, Frame
from PIL import Image, ImageTk
import cv2

from webcam_digit_recognition import WebcamDigitRecognizer

recognizer = WebcamDigitRecognizer("digit_cnn_model.keras")

window = tk.Tk()
window.title("Real-time Digit Identifier")
window.geometry("900x700")
window.configure(bg="#0d0d0d")
window.resizable(True, True)

BG_DARK     = "#0d0d0d"
BG_CARD     = "#161616"
ACCENT      = "#00e5b0"
ACCENT_DIM  = "#008c6c"
TEXT_WHITE  = "#f0f0f0"
TEXT_GRAY   = "#555555"
GREEN_BTN   = "#1a6b40"
GREEN_HOV   = "#25a060"
RED_BTN     = "#7a1f1f"
RED_HOV     = "#b83232"

header = Frame(window, bg=BG_CARD)
header.pack(fill="x")

Frame(header, bg=ACCENT, height=3).pack(fill="x")

Label(
    header,
    text="REAL-TIME DIGIT IDENTIFIER",
    font=("Consolas", 17, "bold"),
    bg=BG_CARD,
    fg=TEXT_WHITE,
    pady=12
).pack()

Label(
    header,
    text="CNN  ·  MNIST  ·  OpenCV  ·  TensorFlow",
    font=("Consolas", 8),
    bg=BG_CARD,
    fg=ACCENT_DIM,
    pady=2
).pack()

Frame(header, bg="#242424", height=1).pack(fill="x", pady=(6, 0))

video_border = Frame(window, bg=ACCENT, padx=2, pady=2)
video_border.pack(pady=14)

video_inner = Frame(video_border, bg=BG_DARK)
video_inner.pack()

video_label = Label(video_inner, bg=BG_DARK)
video_label.pack()

info_panel = Frame(window, bg=BG_CARD, padx=24, pady=10)
info_panel.pack(fill="x", padx=36, pady=(0, 6))

info_label = Label(
    info_panel,
    text="Digit:      |     Confidence: ",
    font=("Consolas", 14, "bold"),
    bg=BG_CARD,
    fg=ACCENT,
)
info_label.pack(side="left")

status = Label(
    info_panel,
    text="●  Camera Off",
    font=("Consolas", 10),
    bg=BG_CARD,
    fg=TEXT_GRAY,
)
status.pack(side="right")

def start_camera():
    recognizer.cap = cv2.VideoCapture(0)
    recognizer.running = True
    status.config(text="●  Camera Running", fg=ACCENT)
    btn_start.config(state="disabled", bg="#0f3d25")
    btn_stop.config(state="normal", bg=RED_BTN)
    update_frame()

def stop_camera():
    recognizer.running = False
    if recognizer.cap:
        recognizer.cap.release()
    video_label.config(image="")
    info_label.config(text="Digit:      |     Confidence: ")
    status.config(text="●  Camera Off", fg=TEXT_GRAY)
    btn_stop.config(state="disabled", bg="#3d0f0f")
    btn_start.config(state="normal", bg=GREEN_BTN)

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
                text=f"Digit: {digit}     |     Confidence: {conf:.2f}%"
            )
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 229, 176), 2)

    fps = recognizer.calculate_fps()
    status.config(text=f"●  Camera Running     FPS: {fps}", fg=ACCENT)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb).resize((640, 480))
    imgtk = ImageTk.PhotoImage(img)

    video_label.imgtk = imgtk
    video_label.config(image=imgtk)

    window.after(10, update_frame)

btn_frame = Frame(window, bg=BG_DARK)
btn_frame.pack(pady=14)

btn_start = Button(
    btn_frame,
    text="▶   START CAMERA",
    font=("Consolas", 11, "bold"),
    bg=GREEN_BTN, fg=TEXT_WHITE,
    activebackground=GREEN_HOV,
    activeforeground=TEXT_WHITE,
    relief="flat",
    padx=22, pady=10,
    cursor="hand2",
    command=start_camera
)
btn_start.grid(row=0, column=0, padx=14)
btn_start.bind("<Enter>", lambda e: e.widget.config(bg=GREEN_HOV) if e.widget["state"] == "normal" else None)
btn_start.bind("<Leave>", lambda e: e.widget.config(bg=GREEN_BTN) if e.widget["state"] == "normal" else None)

btn_stop = Button(
    btn_frame,
    text="■   STOP CAMERA",
    font=("Consolas", 11, "bold"),
    bg=RED_BTN, fg=TEXT_WHITE,
    activebackground=RED_HOV,
    activeforeground=TEXT_WHITE,
    relief="flat",
    padx=22, pady=10,
    cursor="hand2",
    state="disabled",
    command=stop_camera
)
btn_stop.grid(row=0, column=1, padx=14)
btn_stop.bind("<Enter>", lambda e: e.widget.config(bg=RED_HOV) if e.widget["state"] == "normal" else None)
btn_stop.bind("<Leave>", lambda e: e.widget.config(bg=RED_BTN) if e.widget["state"] == "normal" else None)

Frame(window, bg="#1a1a1a", height=1).pack(fill="x", pady=(8, 0))

Label(
    window,
    text="CNN trained on MNIST  ·  Confidence threshold: 60%",
    font=("Consolas", 8),
    bg=BG_DARK,
    fg=TEXT_GRAY,
    pady=6
).pack()

window.mainloop()