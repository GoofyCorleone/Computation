#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Empaqueta ThorCam_App.py como aplicación nativa de macOS (.app)
# ─────────────────────────────────────────────────────────────────────────────
set -e

cd "$(dirname "$0")"
source venv/bin/activate

echo "→ Limpiando builds anteriores..."
rm -rf build dist *.spec

echo "→ Construyendo ThorCam.app con PyInstaller..."
pyinstaller \
  --name "ThorCam" \
  --windowed \
  --noconfirm \
  --clean \
  --osx-bundle-identifier "com.cam.thorcam" \
  --hidden-import PySide6.QtCore \
  --hidden-import PySide6.QtGui \
  --hidden-import PySide6.QtWidgets \
  --collect-submodules tifffile \
  ThorCam_App.py

echo ""
echo "✓ App creada: dist/ThorCam.app"
echo ""
echo "Para instalarla en /Applications:"
echo "  mv dist/ThorCam.app /Applications/"
echo ""
echo "Para ejecutarla:"
echo "  open dist/ThorCam.app"
