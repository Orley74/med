import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
import time
import numpy as np
from utils import *
from load_images import *
import random

# --- KAMERA: Picamera2 ---
from picamera2 import Picamera2

class Picamera2Cap:
    def __init__(self, size=(1640, 1232)):
        self.picam2 = Picamera2()
        cfg = self.picam2.create_preview_configuration(main={"size": size, "format": "RGB888"})
        self.picam2.configure(cfg)
        self.picam2.start()
        time.sleep(0.2)
        self._last_bgr = None

    def read(self):
        frame_rgb = self.picam2.capture_array()                 # RGB (dla MediaPipe idealne)
        self._last_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)  # BGR (do OpenCV)
        return True, self._last_bgr, frame_rgb                  # ret, frame_bgr, frame_rgb

    def release(self):
        self.picam2.stop()
        self.picam2.close()

# ================== OPTYKA / VR CONFIG ==================
# Marginesy bezpieczeństwa jako UŁAMKI rozmiaru ramki
SAFE_MARGIN_X_FRAC = 0.05  # 5% szerokości po lewej i prawej (czarne pasy)
SAFE_MARGIN_Y_FRAC = 0.04  # 4% wysokości u góry i dołu (czarne pasy)

# Presety dla zniekształcenia beczkowatego (k1, k2) – dopasuj do headsetu
HEADSET_PRESETS = {
    "cardboard_v1":   (0.441, 0.156),
    "cardboard_v2":   (0.34,  0.55),
    "bobovr_z4":      (0.30,  0.02),
    "fisheye_strong": (0.55,  0.10),
}
CURRENT_HEADSET = "cardboard_v1"
BARREL_K1, BARREL_K2 = HEADSET_PRESETS[CURRENT_HEADSET]

# Przekształcenia
DISTORT = True         # włącz/wyłącz beczkę
TILT_ENABLE = True     # włącz/wyłącz perspektywiczny tilt (trapez)
TILT_FRAC = 0.01       # siła tilt’u (ułamek szerokości)

# Stereoskopia
zoom = 0.72            # oddalenie (1.0 = brak)
EYE_WIDTH_FRAC = 0.60  # szerokość jednego oka względem całej ramki
IPD_FRAC = 0.065       # 6–7% szerokości (~62–67 mm dla Cardboard/BoboVR)

# Cache map dla beczki (przyspieszenie)
_barrel_cache = {}

def barrel_distort(img: np.ndarray, k1=BARREL_K1, k2=BARREL_K2) -> np.ndarray:
    """Beczkowe zniekształcenie dla pojedynczego oka."""
    h, w = img.shape[:2]
    key = (h, w, k1, k2)
    if key not in _barrel_cache:
        K = np.array([[w, 0, w/2],
                      [0, w, h/2],
                      [0, 0,   1 ]], np.float32)
        D = np.array([k1, k2, 0, 0, 0], np.float32)
        _barrel_cache[key] = cv2.initUndistortRectifyMap(K, D, None, K, (w, h), cv2.CV_32FC1)
    m1, m2 = _barrel_cache[key]
    return cv2.remap(img, m1, m2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

def perspective_tilt(img: np.ndarray, which: str) -> np.ndarray:
    if not TILT_ENABLE:
        return img
    h, w = img.shape[:2]
    off = TILT_FRAC * w
    src = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    if which == 'left':
        dst = np.float32([[0, 0], [w - off, off], [0, h], [w - off, h - off]])
    else:
        dst = np.float32([[off, off], [w, 0], [off, h - off], [w, h]])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

def create_sbs(base_bgr: np.ndarray) -> np.ndarray:
    """Tworzy obraz SBS z marginesami, zoomem, beczką, tilt’em i IPD."""
    global zoom
    h, w = base_bgr.shape[:2]

    # 1) Zoom out treści (przed marginesami)
    base_small = cv2.resize(base_bgr, (int(w * zoom), int(h * zoom)))

    # 2) Marginesy bezpieczeństwa (procentowe)
    mx = max(0, int(w * SAFE_MARGIN_X_FRAC))
    my = max(0, int(h * SAFE_MARGIN_Y_FRAC))
    bg = np.zeros((h, w, 3), np.uint8)

    avail_w = max(1, w - 2 * mx)
    avail_h = max(1, h - 2 * my)
    scale = min(avail_w / base_small.shape[1], avail_h / base_small.shape[0], 1.0)
    new_w = max(1, int(base_small.shape[1] * scale))
    new_h = max(1, int(base_small.shape[0] * scale))
    content = cv2.resize(base_small, (new_w, new_h)) if (new_w, new_h) != base_small.shape[1::-1] else base_small

    x0 = mx + (avail_w - new_w) // 2
    y0 = my + (avail_h - new_h) // 2
    bg[y0:y0 + new_h, x0:x0 + new_w] = content
    base = bg  # po marginesach

    # 3) Ustawienia oczu i IPD
    eye_width = int(w * EYE_WIDTH_FRAC)
    ipd_px = int(w * IPD_FRAC)
    cx = w // 2

    # Źródła dla lewego i prawego oka (przed zniekształceniami)
    left_src  = base[:, cx - eye_width // 2 - ipd_px // 2 : cx + eye_width // 2 - ipd_px // 2]
    right_src = base[:, cx - eye_width // 2 + ipd_px // 2 : cx + eye_width // 2 + ipd_px // 2]

    # 4) Beczka i tilt na każde oko
    left  = barrel_distort(left_src)  if DISTORT else left_src
    right = barrel_distort(right_src) if DISTORT else right_src
    left  = perspective_tilt(left, 'left')
    right = perspective_tilt(right, 'right')

    # 5) Złożenie SBS
    sbs = np.hstack((left, right))
    return sbs

# ================== LOGIKA APLIKACJI ==================
# --- Konfiguracje globalne (Twoje) ---
selected_index = -1
IMAGE_SIZE = (75, 75)
MARGIN = 20
bleeding_stopped = False
selected_main = None
selected_variant = None
TIME = 60
start_time = None
awaiting_click_to_stop_bleeding = False
click_x, click_y = None, None
frame_detect = 50

main_images = [ipmed_img]
variant_images = [helmet_img, staza_img, gaza_img]

# --- Obsługa kliknięcia (jak było) ---
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
    # Uwaga: w trybie SBS współrzędne mogą być nieidealnie zgodne przez zniekształcenia.
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

    # Okno VR w pełnym ekranie
    window_name = "VR"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    ret, frame_bgr, frame_rgb = cap.read()
    if not ret:
        print("Nie udało się uruchomić kamery.")
        return

    timestamp_ms = int(time.time() * 1000)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    landmarker.detect_async(mp_image, timestamp_ms)
    counter = 0

    # Callback myszy – param to oryginalny rozmiar ramki (przed SBS)
    cv2.setMouseCallback(window_name, combined_click_callback, param=(frame_bgr.shape[0], frame_bgr.shape[1]))

    while True:
        counter += 1

        if not bleeding_stopped and selected_main == 0 and selected_variant == 1:
            # Wybrano stazę
            awaiting_click_to_stop_bleeding = True

        ret, frame_bgr, frame_rgb = cap.read()
        if not ret:
            continue

        key = cv2.waitKey(1) & 0xFF
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

        h, w = frame_bgr.shape[:2]
        mask_rgba = np.zeros((h, w, 4), dtype=np.uint8)

        ImageUtils.draw_gallery_background(mask_rgba, selected_index, main_images, variant_images,
                                           image_size=IMAGE_SIZE, margin=MARGIN)

        if latest_result and latest_result.pose_landmarks:
            landmarks = latest_result.pose_landmarks[0]
            for idx, lm in enumerate(landmarks):
                x, y = int(lm.x * w), int(lm.y * h)
                if idx in target:
                    cv2.circle(mask_rgba, (x, y), 8, (0, 255, 0, 255), -1)

            try:
                cx, cy = ImageUtils.getHeartCoords(landmarks, frame_bgr.shape)
                ImageUtils.draw_rgba(mask_rgba, heart_img, cx, cy, size=(50, 50))
            except Exception as e:
                print("Błąd przy obliczaniu serca:", e)

            if not bleeding_stopped:
                elapsed = int(time.time() - start_time)
                remaining = max(0, TIME - elapsed)
                ImageUtils.draw_timer(mask_rgba, remaining)
                Injures.noPart_simple(frame_bgr, mask_rgba, random_part, place, landmarks)

                idx = target[place] if place < len(target) else target[-1]
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

        # Połączenie nakładki z obrazem
        blended = ImageUtils.blend_rgba_over_bgr(frame_bgr, mask_rgba)

        # Render SBS VR + fullscreen
        sbs = create_sbs(blended)
        cv2.imshow(window_name, sbs)

    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()

run()
