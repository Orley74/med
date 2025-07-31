import tkinter as tk
from tkinter import Label
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image, ImageTk
import time
from utils import *

# Wczytaj obraz serca z kanałem alpha (RGBA)
heart_img = cv2.imread("./heart.png", cv2.IMREAD_UNCHANGED)
def draw_heart(img, heart_img, x, y, size=(50, 50)):
    """
    Rysuje obrazek `heart_img` w pozycji (x, y) na tle `img` jako zwykły BGR.
    """
    heart_resized = cv2.resize(heart_img, size)

    h, w, _ = heart_resized.shape

    # Sprawdź, czy mieści się na tle
    if x < 0 or y < 0 or x + w > img.shape[1] or y + h > img.shape[0]:
        return

    img[y:y+h, x:x+w] = heart_resized

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

target = BodyParts.randomPart()
latest_result = None

# Callback do landmarker-a
def print_result(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result

# Ustawienia modelu
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result
)

# Inicjalizacja kamery i GUI
cap = cv2.VideoCapture(0)
root = tk.Tk()
root.title("Med")
label = Label(root)
label.pack()

# Utwórz landmarker
landmarker = PoseLandmarker.create_from_options(options)


# Główna pętla aktualizująca obraz
def update_frame():
    global latest_result

    ret, frame = cap.read()
    if not ret:
        root.after(10, update_frame)
        return

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    timestamp_ms = int(time.time() * 1000)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    landmarker.detect_async(mp_image, timestamp_ms)

    if latest_result and latest_result.pose_landmarks:
        h, w, _ = frame.shape
        for idx, lm in enumerate(latest_result.pose_landmarks[0]):
            cx, cy = int(lm.x * w), int(lm.y * h)
            if idx in target:
                cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)

        # Wyznacz pozycję serca
        coords = BodyParts.getHeartCoords(latest_result.pose_landmarks[0], frame.shape)
        if coords:
            cx, cy = coords
            draw_heart(frame, heart_img, cy - 25, cx - 25)

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=img)
    label.imgtk = imgtk
    label.configure(image=imgtk)
    root.after(10, update_frame)

update_frame()
root.mainloop()
cap.release()
landmarker.close()
