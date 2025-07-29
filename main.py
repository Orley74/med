import tkinter as tk
from tkinter import Label
import cv2
import mediapipe as mp
from PIL import Image, ImageTk

# Konfiguracja MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

root = tk.Tk()
root.title("MediaPipe Pose - uproszczona głowa i czerwona ręka")

label = Label(root)
label.pack()

# ID punktów głowy w MediaPipe (nos, oczy, uszy itd.)
head_landmark_ids = list(range(0, 11))

# ID punktów prawej ręki (MediaPipe: prawe ramię, łokieć, nadgarstek, palce)
right_arm_ids = [12, 14, 16]

def update_frame():
    ret, frame = cap.read()
    if not ret:
        root.after(10, update_frame)
        return

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        # Rysowanie połączeń całego szkieletu
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=None,  # wyłącz rysowanie punktów
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(150, 150, 150), thickness=2)
        )

        h, w, _ = frame.shape
        for idx, lm in enumerate(results.pose_landmarks.landmark):
            cx, cy = int(lm.x * w), int(lm.y * h)

            # Pomijamy szczegóły głowy
            if idx in head_landmark_ids:
                continue

            # Czerwona prawa ręka
            if idx in right_arm_ids:
                cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)  # RED
            else:
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)  # GREEN (inne punkty)

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=img)

    label.imgtk = imgtk
    label.configure(image=imgtk)
    root.after(10, update_frame)

update_frame()
root.mainloop()
cap.release()
