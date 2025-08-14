import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
import time
import numpy as np
from utils import *
from load_images import *
import random
# --- ZAMIANA VideoCapture(0) NA PONIŻSZE ---

import time
import cv2
from picamera2 import Picamera2

class Picamera2Cap:
    def __init__(self, size=(1640, 1232)):
        self.picam2 = Picamera2()
        # Szybka konfiguracja do CV: RGB888 + mniejsza rozdzielczość
        cfg = self.picam2.create_preview_configuration(
            main={"size": size, "format": "RGB888"}
        )
        self.picam2.configure(cfg)
        self.picam2.start()
        time.sleep(0.2)  # krótka stabilizacja
        self._last_bgr = None

    def read(self):
        # Pobieramy klatkę jako RGB (idealne pod MediaPipe)
        frame_rgb = self.picam2.capture_array()
        # Do podglądu w OpenCV konwertujemy na BGR
        self._last_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        # Jeśli chcesz podać do MediaPipe, użyj frame_rgb
        return True, self._last_bgr, frame_rgb

    def release(self):
        self.picam2.stop()
        self.picam2.close()

# Reszta Twojego kodu może zostać prawie bez zmian:
# ret, frame = cap.read()
# tu 'frame' to BGR do cv2.imshow, a masz też 'frame_rgb' do MediaPipe

# --- Konfiguracje globalne ---
selected_index = -1
IMAGE_SIZE = (75, 75)
MARGIN = 20
bleeding_stopped = False 
selected_main = None
selected_variant = None
TIME = 60  
start_time = None  
awaiting_click_to_stop_bleeding = False
click_x, click_y = None, None  # zapis kliknięcia użytkownika
frame_detect = 5 # co ktora klatke nakladac nowa detekcje / odciazyc plytke

main_images = [ipmed_img]
variant_images = [helmet_img, staza_img, gaza_img]

# --- Obsługa kliknięcia ---
def gallery_click(event, x, y, flags, param):
    global selected_index, selected_main, selected_variant
    h, w = param
    img_w, img_h = IMAGE_SIZE
    y_main = h - 100 - MARGIN

    if event == cv2.EVENT_LBUTTONDOWN:
        for i in range(len(main_images)):
            img_x = MARGIN + i * (100 + MARGIN)
            if img_x <= x <= img_x + 100 and y_main <= y <= y_main + 100:
                selected_index = -1 if selected_index == i else i
                selected_main = i
                selected_variant = None
                print("Wybrales IPMED")
                return

        if selected_index is not None and selected_index >= 0:
            for j in range(len(variant_images)):
                vx = MARGIN + selected_index * (100 + MARGIN) + j * (75 + MARGIN)
                vy = y_main - 75 - MARGIN
                if vx <= x <= vx + 75 and vy <= y <= vy + 75:
                    selected_variant = j
                    return

        selected_index = -1
        selected_main = None
        selected_variant = None

def body_click(event, x, y, flags, param):
    global click_x, click_y, awaiting_click_to_stop_bleeding
    if not awaiting_click_to_stop_bleeding:
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        click_x, click_y = x, y

def combined_click_callback(event, x, y, flags, param):
    gallery_click(event, x, y, flags, param)
    body_click(event, x, y, flags, param)

# --- MediaPipe Setup ---
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

latest_result = None

def print_result(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result
)

cap = Picamera2Cap(size=(1640, 1232))
landmarker = PoseLandmarker.create_from_options(options)

# --- Główna pętla ---
def run():
    global latest_result, start_time
    global bleeding_stopped, selected_main, selected_variant
    global target, place, click_x, click_y, awaiting_click_to_stop_bleeding

    if start_time is None:
        start_time = time.time()

    target = BodyParts.randomPart()
    place = random.randint(0, 3)
    random_part = random.randint(0, 4)

    cv2.namedWindow("Blended")
    ret, frame, frame_rgb = cap.read()
    if not ret:
        print("Nie udało się uruchomić kamery.")
        return
    landmarker.detect_async(mp_image, timestamp_ms)
    counter = 0
    cv2.setMouseCallback("Blended", combined_click_callback, param=(frame.shape[0], frame.shape[1]))
    
    while True:
        counter += 1
        if not bleeding_stopped and selected_main == 0 and selected_variant == 1:
            print("Wybrales STAZE")
            awaiting_click_to_stop_bleeding = True

        ret, frame_rgb, frame = cap.read()
        if not ret:
            continue

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord(' '):
            target = BodyParts.randomPart()
            place = random.randint(0, 3)
            random_part = random.randint(0, 4)
            click_x, click_y = None, None

        timestamp_ms = int(time.time() * 1000)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        if counter % frame_detect == 0:
            landmarker.detect_async(mp_image, timestamp_ms)

        h, w = frame.shape[:2]
        mask_rgba = np.zeros((h, w, 4), dtype=np.uint8)

        ImageUtils.draw_gallery_background(mask_rgba, selected_index, main_images, variant_images, image_size=IMAGE_SIZE, margin=MARGIN)

        if latest_result and latest_result.pose_landmarks:
            landmarks = latest_result.pose_landmarks[0]
            for idx, lm in enumerate(landmarks):
                x, y = int(lm.x * w), int(lm.y * h)
                if idx in target:
                    cv2.circle(mask_rgba, (x, y), 8, (0, 255, 0, 255), -1)

            try:
                cx, cy = ImageUtils.getHeartCoords(landmarks, frame.shape)
                ImageUtils.draw_rgba(mask_rgba, heart_img, cx, cy, size=(50, 50))
            except Exception as e:
                print("Błąd przy obliczaniu serca:", e)

            if not bleeding_stopped:
                elapsed = int(time.time() - start_time)
                remaining = max(0, TIME - elapsed)
                ImageUtils.draw_timer(mask_rgba, remaining)
                Injures.noPart_simple(frame, mask_rgba, random_part, place, landmarks)

                if place < len(target):
                    idx = target[place]
                else:
                    idx = target[-1]
                lm = landmarks[idx]
                wound_x, wound_y = int(lm.x * w), int(lm.y * h)

                if click_x is not None and click_y is not None:
                    if abs(click_x - wound_x) <= 50 and abs(click_y - wound_y) <= 50:
                        print("Rana została opatrzona!")
                        ImageUtils.draw_Success(mask_rgba)
                        bleeding_stopped = True
                        awaiting_click_to_stop_bleeding = False
                        click_x, click_y = None, None
                    else:
                        print("Kliknięto poza raną.")
                        click_x = None
                        click_y = None

        blended = ImageUtils.blend_rgba_over_bgr(frame, mask_rgba)
        cv2.imshow("Blended", blended)

    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()

run()