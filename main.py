import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import numpy as np
from utils import *
from load_images import *
import random

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

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

# Utwórz landmarker
landmarker = PoseLandmarker.create_from_options(options)

# Główna pętla aktualizująca obraz
def run():
    global latest_result

    #wylosowanie pierwszej czesci ciala
    target = BodyParts.randomPart()
    place = random.randint(0,3)
    random_part = random.randint(0,4)
    cx_head,cy_head = 0,0
    while True:
        ret, frame = cap.read()
        key = cv2.waitKey(1)

        if not ret:
            continue
        if key == ord('q'):
            break
        elif key == ord(' '):
            target = BodyParts.randomPart()
            random_part = random.randint(0,4)
            place = random.randint(0,3)
        # detekcja czesci ciala i utworzenie maski (kopia klatki z dodanym 4 kanalem obecnie wypelnionym zerami)
        timestamp_ms = int(time.time() * 1000)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        landmarker.detect_async(mp_image, timestamp_ms)
        mask_rgba = np.zeros((frame.shape[0], frame.shape[1], 4), dtype=np.uint8)

        if latest_result and latest_result.pose_landmarks:
            h, w, _ = frame.shape
            # rysowanie kolek w miejscach detekcji
            for idx, lm in enumerate(latest_result.pose_landmarks[0]):
                x, y = int(lm.x * w), int(lm.y * h)
                if idx in target:
                    """
                     cv2.circle(mask_rgba, (x, y), 8, (0, 255, 0, 255))
                     mask_rgba - maska wejsciowa
                     (x,y) - polozenie kropki
                     8 - wielkosc kropki
                     (0,255,0,255) - pierwsze 3 to kolor w formacie BGR, ostatnia widoczosc 0 brak, 255 max
                    """
                    cv2.circle(mask_rgba, (x, y), 8, (0, 255, 0, 255), -1)
            try:
                cx,cy = ImageUtils.getHeartCoords(latest_result.pose_landmarks[0], frame.shape)
                cx_head,cy_head = ImageUtils.getHeadCoords(latest_result.pose_landmarks[0], frame.shape)

            except Exception as e:
                print(e)

            ImageUtils.draw_rgba(mask_rgba, heart_img, cx, cy, size=(50, 50))
            ImageUtils.draw_rgba(mask_rgba, helmet_img, cx_head,cy_head, size=(120, 70))


            Injures.noPart_simple(frame, mask_rgba, random_part, place, frame.shape, latest_result.pose_landmarks[0])
            
        # dodaje obrazki z maski do zdjecia z klatki

        blended = ImageUtils.blend_rgba_over_bgr(frame.copy(), mask_rgba)
        mask_rgba_bgr = cv2.cvtColor(mask_rgba, cv2.COLOR_RGBA2BGR)

        # wyswietlenie 3 widokow, oryginalu, maski i polaczonego maski z oryginalem
        cv2.imshow("Blended", blended)       
        cv2.imshow("frame", frame)
        cv2.imshow("RGBA Mask", mask_rgba_bgr)
        
    
run()
cap.release()
landmarker.close()
