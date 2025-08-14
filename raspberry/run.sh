#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Aktywuj venv
source .venv/bin/activate

# Uruchom aplikację
exec python3 raspberry_VR.py
