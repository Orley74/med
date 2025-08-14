#!/usr/bin/env bash
set -euo pipefail

# Przejdź do katalogu, w którym leży skrypt (bez sztywnych ścieżek)
cd "$(dirname "$0")"

# Pakiety systemowe (Picamera2 i OpenCV z repo RPi)
sudo apt update
sudo apt install -y python3-venv python3-picamera2 python3-opencv

# Wirtualne środowisko z dostępem do site-packages systemowych
python3 -m venv .venv --system-site-packages
source .venv/bin/activate

# Narzędzia PIP + MediaPipe (w venv)
python -m pip install --upgrade pip setuptools wheel
pip install mediapipe


