#!/usr/bin/env bash
# Downloads the two offline models Frend needs into ./models (about 45 MB total).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p models

NANODET=models/object_detection_nanodet_2022nov.onnx
if [ ! -f "$NANODET" ]; then
    echo "Downloading NanoDet-Plus object detector (3.8 MB)..."
    curl -fL -o "$NANODET" \
        https://huggingface.co/opencv/object_detection_nanodet/resolve/main/object_detection_nanodet_2022nov.onnx
fi

VOSK=models/vosk-model-small-en-us-0.15
if [ ! -d "$VOSK" ]; then
    echo "Downloading Vosk small English speech model (40 MB)..."
    curl -fL -o models/vosk.zip https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
    python3 -m zipfile -e models/vosk.zip models/
    rm models/vosk.zip
fi

echo "Models ready in ./models"
