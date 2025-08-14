#!/usr/bin/env python3
import os, sys, time, threading
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from utils import ImageUtils, BodyParts, Injures
from load_images import ipmed_img, helmet_img, staza_img, gaza_img, heart_img
from picamera2 import Picamera2

# ====== GStreamer / RTSP ======
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
gi.require_version('GObject', '2.0')
from gi.repository import Gst, GstRtspServer, GObject

# ================== KONFIG ==================
# — Kamera —
CAM_SIZE = (640, 480)       # (width, height) libcamera preview
CAM_FPS  = 20

# — RTSP —
RTSP_PORT = 8554
RTSP_PATH = "/vr"             # rtsp://<host>:8554/vr

# — MediaPipe —
MODEL_PATH = "pose_landmarker_lite.task"
ANALYSIS_INTERVAL_MS = 1000    # co ile ms liczona jest nowa maska

# — Render / VR —
WINDOW_NAME = "VR"
SCREEN_FPS  = 60              # odświeżanie renderu
IMAGE_SIZE  = (75, 75)
MARGIN      = 20
TIME_LIMIT  = 60

# Marginesy bezpieczeństwa jako UŁAMKI rozmiaru ramki (czarne pasy)
SAFE_MARGIN_X_FRAC = 0.05     # po bokach
SAFE_MARGIN_Y_FRAC = 0.04     # góra/dół

# Presety soczewek
HEADSET_PRESETS = {
    "cardboard_v1":   (0.441, 0.156),
    "cardboard_v2":   (0.34,  0.55),
    "bobovr_z4":      (0.30,  0.02),
    "fisheye_strong": (0.55,  0.10),
}
CURRENT_HEADSET = "cardboard_v1"
BARREL_K1, BARREL_K2 = HEADSET_PRESETS[CURRENT_HEADSET]

DISTORT     = True
TILT_ENABLE = True
TILT_FRAC   = 0.01
zoom        = 0.72
EYE_WIDTH_FRAC = 0.60
IPD_FRAC       = 0.065

# ================== BUFFERY WSPÓŁDZIELONE ==================
buffers = {
    "lock": threading.Lock(),
    "raw_bgr": None,     # ostatnia surowa klatka BGR z kamery
    "raw_rgb": None,     # ostatnia surowa klatka RGB (dla MediaPipe)
    "mask_rgba": None,   # ostatnia wyliczona maska RGBA
    "frame_size": None,  # (h, w)
}

# ================== Picamera2 WRAPPER ==================
class Picamera2Cap:
    def __init__(self, size=(1640, 1232), fps=30):
        self.picam2 = Picamera2()
        self.size = size
        self.fps  = fps
        cfg = self.picam2.create_preview_configuration(
            main={"size": (size[0], size[1]), "format": "RGB888"}
        )
        self.picam2.configure(cfg)
        self.picam2.start()
        time.sleep(0.2)

    def read_both(self):
        """Zwraca BGR (dla OpenCV) + RGB (dla MediaPipe)."""
        frame_rgb = self.picam2.capture_array()                   # RGB
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)    # BGR
        return True, frame_bgr, frame_rgb

    def close(self):
        self.picam2.stop()
        self.picam2.close()

# ================== OPTYKA / VR ==================
_barrel_cache = {}
def barrel_distort(img: np.ndarray, k1=BARREL_K1, k2=BARREL_K2) -> np.ndarray:
    h, w = img.shape[:2]
    key = (h, w, k1, k2)
    if key not in _barrel_cache:
        K = np.array([[w, 0, w/2],
                      [0, w, h/2],
                      [0, 0,   1 ]], np.float32)
        D = np.array([k1, k2, 0, 0, 0], np.float32)
        _barrel_cache[key] = cv2.initUndistortRectifyMap(
            K, D, None, K, (w, h), cv2.CV_32FC1
        )
    m1, m2 = _barrel_cache[key]
    return cv2.remap(img, m1, m2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

def perspective_tilt(img: np.ndarray, which: str) -> np.ndarray:
    if not TILT_ENABLE: return img
    h, w = img.shape[:2]
    off = TILT_FRAC * w
    src = np.float32([[0,0],[w,0],[0,h],[w,h]])
    if which=='left':
        dst = np.float32([[0,0],[w-off,off],[0,h],[w-off,h-off]])
    else:
        dst = np.float32([[off,off],[w,0],[off,h-off],[w,h]])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

def create_sbs(base_bgr: np.ndarray) -> np.ndarray:
    global zoom
    h, w = base_bgr.shape[:2]

    # Zoom out
    base_small = cv2.resize(base_bgr, (int(w*zoom), int(h*zoom)))

    # Marginesy bezpieczeństwa
    mx = max(0, int(w * SAFE_MARGIN_X_FRAC))
    my = max(0, int(h * SAFE_MARGIN_Y_FRAC))
    bg = np.zeros((h, w, 3), np.uint8)
    avail_w = max(1, w - 2*mx)
    avail_h = max(1, h - 2*my)
    scale = min(avail_w / base_small.shape[1], avail_h / base_small.shape[0], 1.0)
    new_w = max(1, int(base_small.shape[1] * scale))
    new_h = max(1, int(base_small.shape[0] * scale))
    content = cv2.resize(base_small, (new_w, new_h)) if (new_w, new_h) != base_small.shape[1::-1] else base_small
    x0 = mx + (avail_w - new_w)//2
    y0 = my + (avail_h - new_h)//2
    bg[y0:y0+new_h, x0:x0+new_w] = content
    base = bg

    # Oczy
    eye_width = int(w * EYE_WIDTH_FRAC)
    ipd_px    = int(w * IPD_FRAC)
    cx = w // 2
    left_src  = base[:, cx - eye_width//2 - ipd_px//2 : cx + eye_width//2 - ipd_px//2]
    right_src = base[:, cx - eye_width//2 + ipd_px//2 : cx + eye_width//2 + ipd_px//2]

    left  = barrel_distort(left_src)  if DISTORT else left_src
    right = barrel_distort(right_src) if DISTORT else right_src
    left  = perspective_tilt(left, 'left')
    right = perspective_tilt(right, 'right')

    return np.hstack((left, right))

# ================== MEDIA PIPE (ASYNC) ==================
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

latest_result = None
def mp_callback(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result

mp_options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=mp_callback
)
landmarker = PoseLandmarker.create_from_options(mp_options)

# ================== RTSP SERVER (appsrc) ==================
rtsp_appsrc = None  # wypełni się po starcie sesji
def on_media_configure(factory, media):
    global rtsp_appsrc
    element = media.get_element()
    appsrc = element.get_child_by_name("mysrc")
    rtsp_appsrc = appsrc
    # ustaw framerate przez property, PTS będzie nadawany w capture_thread
    if appsrc:
        appsrc.set_property("is-live", True)
        appsrc.set_property("format", Gst.Format.TIME)

def start_rtsp_server(width, height, fps):
    GObject.threads_init()
    Gst.init(None)

    server = GstRtspServer.RTSPServer()
    server.props.service = str(RTSP_PORT)
    factory = GstRtspServer.RTSPMediaFactory()
    factory.set_shared(True)

    # appsrc (BGR) -> videoconvert -> x264enc(low-latency) -> rtph264pay
    launch = (
        f"( appsrc name=mysrc is-live=true block=true format=GST_FORMAT_TIME "
        f"caps=video/x-raw,format=BGR,width={width},height={height},framerate={fps}/1 "
        f"! videoconvert "
        f"! x264enc tune=zerolatency speed-preset=veryfast bitrate=2000 key-int-max={fps} "
        f"! rtph264pay name=pay0 pt=96 )"
    )
    factory.set_launch(launch)
    factory.connect("media-configure", on_media_configure)

    mounts = server.get_mount_points()
    mounts.add_factory(RTSP_PATH, factory)
    server.attach(None)
    print(f"[RTSP] uruchomiono: rtsp://<host>:{RTSP_PORT}{RTSP_PATH}")

# ================== WĄTKI ==================

def capture_thread():
    """1) Kamerka → bufor + push do RTSP (appsrc)."""
    global rtsp_appsrc
    cap = Picamera2Cap(size=CAM_SIZE, fps=CAM_FPS)
    h, w = CAM_SIZE[1], CAM_SIZE[0]
    frame_duration_ns = int(1e9 / CAM_FPS)
    pts_ns = 0

    try:
        while True:
            ok, frame_bgr, frame_rgb = cap.read_both()
            if not ok:
                continue
            with buffers["lock"]:
                buffers["raw_bgr"] = frame_bgr
                buffers["raw_rgb"] = frame_rgb
                if buffers["frame_size"] is None:
                    buffers["frame_size"] = (frame_bgr.shape[0], frame_bgr.shape[1])

            # RTSP push
            appsrc = rtsp_appsrc
            if appsrc is not None:
                data = frame_bgr.tobytes()
                buf = Gst.Buffer.new_allocate(None, len(data), None)
                buf.fill(0, data)
                buf.pts = pts_ns
                buf.dts = pts_ns
                buf.duration = frame_duration_ns
                pts_ns += frame_duration_ns
                try:
                    appsrc.emit("push-buffer", buf)
                except Exception as e:
                    # gdy klient się rozłączy – nic strasznego
                    pass
    finally:
        cap.close()

def mask_thread():
    """2) Co ANALYSIS_INTERVAL_MS liczy maskę na bazie najnowszej klatki."""
    global latest_result
    start_time = time.time()
    target = BodyParts.randomPart()
    place = np.random.randint(0, 3)
    random_part = np.random.randint(0, 4)

    while True:
        t0 = time.time()

        with buffers["lock"]:
            rgb = None if buffers["raw_rgb"] is None else buffers["raw_rgb"].copy()
            bgr = None if buffers["raw_bgr"] is None else buffers["raw_bgr"].copy()

        if rgb is None or bgr is None:
            time.sleep(0.005)
            continue

        # MediaPipe async
        timestamp_ms = int(time.time() * 1000)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        landmarker.detect_async(mp_image, timestamp_ms)

        h, w = bgr.shape[:2]
        mask = np.zeros((h, w, 4), np.uint8)

        # UI
        ImageUtils.draw_gallery_background(
            mask, -1, [ipmed_img], [helmet_img, staza_img, gaza_img],
            image_size=IMAGE_SIZE, margin=MARGIN
        )

        # wykorzystaj najnowszy wynik (z callbacku)
        res = latest_result
        if res and res.pose_landmarks:
            lms = res.pose_landmarks[0]
            for i in target:
                x, y = int(lms[i].x * w), int(lms[i].y * h)
                cv2.circle(mask, (x, y), 8, (0, 255, 0, 255), -1)
            try:
                cx, cy = ImageUtils.getHeartCoords(lms, bgr.shape)
                ImageUtils.draw_rgba(mask, heart_img, cx, cy, size=(50, 50))
            except:
                pass

            elapsed = int(time.time() - start_time)
            remaining = max(0, TIME_LIMIT - elapsed)
            ImageUtils.draw_timer(mask, remaining)

            Injures.noPart_simple(bgr, mask, random_part, place, lms)

        with buffers["lock"]:
            buffers["mask_rgba"] = mask

        # regulacja okresu
        dt = (time.time() - t0)
        sleep_s = max(0.0, (ANALYSIS_INTERVAL_MS / 1000.0) - dt)
        time.sleep(sleep_s)

def display_thread():
    """3) Scala surową klatkę + maskę i wyświetla SBS VR fullscreen."""
    # pełny ekran
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    dt = 1.0 / SCREEN_FPS
    while True:
        t0 = time.time()
        with buffers["lock"]:
            bgr = None if buffers["raw_bgr"] is None else buffers["raw_bgr"].copy()
            msk = None if buffers["mask_rgba"] is None else buffers["mask_rgba"].copy()
        if bgr is None:
            time.sleep(0.005)
            continue

        if msk is not None and msk.shape[:2] == bgr.shape[:2]:
            a = msk[:, :, 3:4].astype(np.float32) / 255.0
            blended = cv2.convertScaleAbs(bgr * (1 - a) + msk[:, :, :3] * a)
        else:
            blended = bgr

        sbs = create_sbs(blended)
        cv2.imshow(WINDOW_NAME, sbs)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        rem = dt - (time.time() - t0)
        if rem > 0:
            time.sleep(rem)

    cv2.destroyAllWindows()

# ================== MAIN ==================
def main():
    # Start RTSP
    start_rtsp_server(CAM_SIZE[0], CAM_SIZE[1], CAM_FPS)

    # Wątki
    t1 = threading.Thread(target=capture_thread, daemon=True)
    t2 = threading.Thread(target=mask_thread,     daemon=True)
    t3 = threading.Thread(target=display_thread,  daemon=False)

    t1.start()
    t2.start()
    t3.start()   # blokuje do zamknięcia okna

if __name__ == "__main__":
    main()
