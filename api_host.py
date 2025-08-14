"""VR Camera — Optimized SBS VR for Cardboard/BoboVR
====================================================

FullHD + max quality.

- Receives JPEG frames via POST /upload_frame
- Runs MediaPipe Pose + custom mask graphics in a background thread
- Composites RGBA mask onto the frame
- Applies barrel warp & perspective (trapezoidal) tilt
- Shifts IPD for true stereoscopic SBS
- Streams MJPEG on /stream.mjpg (~30fps) and serves static SBS on /sbs.jpg
"""

import ssl, threading, time
from pathlib import Path
from typing import Dict, Tuple
import cv2, mediapipe as mp, numpy as np
from flask import Flask, make_response, render_template_string, request
from utils import ImageUtils, BodyParts, Injures
from load_images import ipmed_img, helmet_img, staza_img, gaza_img, heart_img

# ================== CONFIG ==================
SAFE_MARGIN_X_FRAC = 0.2  # ułamek szerokości na czarne marginesy po bokach (np. 0.05 = 5%)
SAFE_MARGIN_Y_FRAC = 0.04  # ułamek wysokości na margines u góry i dołu

IMAGE_SIZE = (75, 75)
MARGIN = 20
TIME_LIMIT = 60
ANALYSIS_INTERVAL_MS = 100        # ms — częstszy upload!
STREAM_FPS = 20
JPEG_Q = 90                       # Wyższa jakość
zoom = 0.72                        # Oddalone, szeroki kadr



HEADSET_PRESETS: Dict[str, Tuple[float, float]] = {
    "cardboard_v1":   (0.441, 0.156),
    "cardboard_v2":   (0.34,  0.55),
    "bobovr_z4":      (0.30,  0.02),
    "fisheye_strong": (0.55,  0.10),
}
CURRENT_HEADSET = "cardboard_v1"
BARREL_K1, BARREL_K2 = HEADSET_PRESETS[CURRENT_HEADSET]
DISTORT   = True
SHIFT_PX  = 10                     # Dla FullHD i Cardboard/BoboVR
TILT_ENABLE = True
TILT_FRAC   = 0.01

# ================== BUFFERS ==================
def init_buffers():
    return {"lock": threading.Lock(), "raw": None, "mask_rgba": None, "frame_size": None}
buffers = init_buffers()

# ================== MediaPipe Pose ==================
mp_img = mp.Image
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOpts = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
landmarker = PoseLandmarker.create_from_options(
    PoseLandmarkerOpts(
        base_options=BaseOptions(model_asset_path="pose_landmarker_lite.task"),
        running_mode=VisionRunningMode.IMAGE,
    )
)

# ================== FLASK & HTML ==================
app = Flask(__name__)
html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset='utf-8'/>
  <meta name='viewport' content='width=device-width, initial-scale=1, viewport-fit=cover'/>
  <title>VR Cam SBS</title>
  <style>
    html, body {{ margin:0; height:100%; background:#000; }}
    #vr {{ width:100vw; height:100vh; object-fit:contain; }}
  </style>
</head>
<body>
  <img id='vr' src='/stream.mjpg' alt='VR Stream'/>
  <video id='v' autoplay playsinline muted style='display:none'></video>
  <script>
  (async ()=> {{
    const INTERVAL = {ANALYSIS_INTERVAL_MS};
    const video = document.getElementById('v');
    await new Promise(res=>setTimeout(res, 50));
    // --- FULL HD wymuszony ---
    const stream = await navigator.mediaDevices.getUserMedia({{
        video: {{
          facingMode: 'environment',
          width: {{ideal:1920, max:1920}},
          height: {{ideal:1080, max:1080}}
        }}
    }}).catch(()=>null);
    if (!stream) return alert('Camera unavailable');
    video.srcObject = stream;
    await video.play();
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    function resize() {{
      canvas.width = video.videoWidth || 1920;
      canvas.height = video.videoHeight || 1080;
    }}
    video.addEventListener('loadedmetadata', resize);
    resize();
    setInterval(() => {{
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      canvas.toBlob(blob => {{
        if (blob) fetch('/upload_frame', {{ method: 'POST', body: blob, headers: {{'Content-Type':'image/jpeg'}} }});
      }}, 'image/jpeg', 0.97);
    }}, INTERVAL);
  }})();
  </script>
</body>
</html>"""

# ================== OPTICS ==================
_barrel_cache: Dict[Tuple[int,int,float,float], Tuple[np.ndarray,np.ndarray]] = {}
def barrel_distort(img: np.ndarray, k1=BARREL_K1, k2=BARREL_K2) -> np.ndarray:
    h,w = img.shape[:2]; key = (h,w,k1,k2)
    if key not in _barrel_cache:
        K = np.array([[w,0,w/2],[0,w,h/2],[0,0,1]],np.float32)
        D = np.array([k1,k2,0,0,0],np.float32)
        _barrel_cache[key]=cv2.initUndistortRectifyMap(K,D,None,K,(w,h),cv2.CV_32FC1)
    m1,m2 = _barrel_cache[key]
    return cv2.remap(img,m1,m2,interpolation=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)

def perspective_tilt(img: np.ndarray, which: str) -> np.ndarray:
    if not TILT_ENABLE: return img
    h,w = img.shape[:2]; off = TILT_FRAC*w
    src = np.float32([[0,0],[w,0],[0,h],[w,h]])
    if which=='left':  dst = np.float32([[0,0],[w-off,off],[0,h],[w-off,h-off]])
    else:              dst = np.float32([[off,off],[w,0],[off,h-off],[w,h]])
    M = cv2.getPerspectiveTransform(src,dst)
    return cv2.warpPerspective(img,M,(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)

def create_sbs(base: np.ndarray) -> np.ndarray:
    global zoom
    h, w = base.shape[:2]

    # --- Zoom out treści, zanim dodamy marginesy ---
    base_small = cv2.resize(base, (int(w * zoom), int(h * zoom)))

    # --- Marginesy bezpieczeństwa jako ułamki rozmiaru ramki ---
    mx = max(0, int(w * SAFE_MARGIN_X_FRAC))  # lewy/prawy
    my = max(0, int(h * SAFE_MARGIN_Y_FRAC))  # góra/dół

    # Płótno docelowe (pełny rozmiar ramki) z czarnym tłem
    bg = np.zeros((h, w, 3), np.uint8)

    # Obszar dostępny na obraz po odjęciu marginesów
    avail_w = max(1, w - 2 * mx)
    avail_h = max(1, h - 2 * my)

    # Jeśli base_small nie mieści się w obszarze, dopasuj (bez powiększania)
    scale = min(avail_w / base_small.shape[1], avail_h / base_small.shape[0], 1.0)
    new_w = max(1, int(base_small.shape[1] * scale))
    new_h = max(1, int(base_small.shape[0] * scale))
    content = cv2.resize(base_small, (new_w, new_h)) if (new_w, new_h) != base_small.shape[1::-1] else base_small

    # Wycentruj treść w środku obszaru roboczego (z marginesami po bokach i górze/dole)
    x0 = mx + (avail_w - new_w) // 2
    y0 = my + (avail_h - new_h) // 2
    bg[y0:y0+new_h, x0:x0+new_w] = content

    # Od tej chwili "base" ma pełny rozmiar z czarną ramką
    base = bg

    # --- Eye crop (szerzej) + IPD ---
    eye_width = int(w * 0.6)
    ipd_px = int(w * 0.065)    # 6–7% szerokości dla ~62–67 mm
    cx = w // 2

    left_src  = base[:, cx - eye_width//2 - ipd_px//2 : cx + eye_width//2 - ipd_px//2]
    right_src = base[:, cx - eye_width//2 + ipd_px//2 : cx + eye_width//2 + ipd_px//2]

    # Zniekształcenia i tilt
    left  = barrel_distort(left_src)  if DISTORT else left_src
    right = barrel_distort(right_src) if DISTORT else right_src
    left  = perspective_tilt(left, 'left')
    right = perspective_tilt(right, 'right')

    return np.hstack((left, right))


# ================== ROUTES ==================
@app.route('/')
def index():
    return render_template_string(html)

@app.route('/upload_frame',methods=['POST'])
def upload_frame():
    data = request.get_data(cache=False)
    if not data: return 'Bad Request', 400
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if frame is None: return 'Unsupported Media', 415
    h, w = frame.shape[:2]
    with buffers['lock']:
        buffers['raw'] = frame
        if buffers['frame_size'] is None:
            buffers['frame_size'] = (h, w)
    return '', 204

@app.route('/stream.mjpg')
def stream_mjpg():
    def gen():
        dt=1.0/STREAM_FPS
        while True:
            t0 = time.time()
            with buffers['lock']:
                base = buffers['raw'].copy() if buffers['raw'] is not None else None
                msk  = buffers['mask_rgba'].copy() if buffers['mask_rgba'] is not None else None
            if base is None:
                time.sleep(dt); continue
            if msk is not None and msk.shape[:2]==base.shape[:2]:
                a = msk[:,:,3:4]/255.0
                base = cv2.convertScaleAbs(base*(1-a)+msk[:,:,:3]*a)
            sbs = create_sbs(base)
            ok, jpg = cv2.imencode('.jpg', sbs, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_Q])
            if ok: yield(b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'+jpg.tobytes()+b'\r\n')
            rem = dt-(time.time()-t0); time.sleep(rem if rem>0 else 0)
    return app.response_class(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/sbs.jpg')
def sbs_jpg():
    with buffers['lock']:
        b = buffers['raw'].copy() if buffers['raw'] is not None else None
        m = buffers['mask_rgba'].copy() if buffers['mask_rgba'] is not None else None
    if not b: return 'No frame',404
    if m is not None and m.shape[:2]==b.shape[:2]:
        a = m[:,:,3:4]/255.0
        b = cv2.convertScaleAbs(b*(1-a)+m[:,:,:3]*a)
    ok,jp=cv2.imencode('.jpg',create_sbs(b),[int(cv2.IMWRITE_JPEG_QUALITY),JPEG_Q])
    return make_response(jp.tobytes(),{'Content-Type':'image/jpeg'})

@app.route('/mask.png')
def mask_png():
    with buffers['lock']:
        m = buffers['mask_rgba'].copy() if buffers['mask_rgba'] is not None else None
        sz = buffers['frame_size']
    if m is None:
        h,w = sz if sz else (1080,1920)
        m = np.zeros((h,w,4),np.uint8)
    ok,p=cv2.imencode('.png',m)
    return make_response(p.tobytes(),{'Content-Type':'image/png'})

# ================== WORKER ==================
def process_loop():
    st = time.time(); targ = BodyParts.randomPart(); idx = np.random.randint(len(targ)); rndv = np.random.randint(3)
    while True:
        with buffers['lock']:
            frm = buffers['raw'].copy() if buffers['raw'] is not None else None
        if frm is None:
            time.sleep(0.01); continue
        try:
            h,w = frm.shape[:2]; mask = np.zeros((h,w,4),np.uint8)
            ImageUtils.draw_gallery_background(mask,-1,[ipmed_img],[helmet_img,staza_img,gaza_img],image_size=(int(0.15*h),int(0.15*w)),margin=MARGIN)
            res=landmarker.detect(mp_img(image_format=mp.ImageFormat.SRGB,data=frm))
            if res and res.pose_landmarks:
                lms=res.pose_landmarks[0]
                for i in targ: x,y=int(lms[i].x*w),int(lms[i].y*h); cv2.circle(mask,(x,y),8,(0,255,0,255),-1)
                try: cx,cy=ImageUtils.getHeartCoords(lms,frm.shape); ImageUtils.draw_rgba(mask,heart_img,cx,cy,size=(50,50))
                except: pass
                Injures.noPart_simple(frm,mask,rndv,idx,lms)
            
            rem= max(0,TIME_LIMIT-int(time.time()-st)); ImageUtils.draw_timer(mask,rem,1.5,6)

            with buffers['lock']: buffers['mask_rgba']=mask

        except Exception as e:
            print(e)

        time.sleep(0.01)

if __name__ == '__main__':
    threading.Thread(target=process_loop,daemon=True).start()
    cert=Path('~/Downloads/localhost.pem').expanduser(); key=Path('~/Downloads/localhost-key.pem').expanduser()
    if cert.exists() and key.exists():
        ctx=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(str(cert),str(key))
    else: ctx=None
    app.run(host='0.0.0.0',port=5000,ssl_context=ctx,threaded=True)
