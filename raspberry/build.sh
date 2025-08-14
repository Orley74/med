#!/usr/bin/env bash
set -euo pipefail

# Przejdź do katalogu, w którym leży skrypt (bez sztywnych ścieżek)
cd "$(dirname "$0")"

# Pakiety systemowe (Picamera2 i OpenCV z repo RPi)
sudo apt install -y python3-gi gir1.2-gst-rtsp-server-1.0 \
    gstreamer1.0-rtsp python3-opencv python3-numpy \
    gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad gstreamer1.0-libav
    
sudo apt install -y python3-venv python3-picamera2 python3-opencv

# Wirtualne środowisko z dostępem do site-packages systemowych
python3 -m venv .venv --system-site-packages
source .venv/bin/activate

# Narzędzia PIP + MediaPipe (w venv)
python -m pip install --upgrade pip setuptools wheel
pip install mediapipe


