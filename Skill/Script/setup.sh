#!/usr/bin/env bash
# One-time setup for the Gemini Watermark Remover skill.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "== checking ffmpeg =="
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg not found. Install it:  brew install ffmpeg"
else
  echo "ffmpeg: $(ffmpeg -version | head -1)"
fi

echo "== installing python deps =="
python3 -m pip install -r "$HERE/../requirements.txt"

echo "== pre-fetching the LaMa model (~196 MB, one time) =="
python3 - <<'PY' || echo "(model will download automatically on first run)"
try:
    from simple_lama_inpainting import SimpleLama
    SimpleLama()   # downloads big-lama.pt to the torch cache (CUDA-load may warn; ignored)
except Exception as e:
    import os
    ok = os.path.exists(os.path.expanduser("~/.cache/torch/hub/checkpoints/big-lama.pt"))
    print("model present:", ok)
PY

echo "== done. Try: =="
echo "  python3 \"$HERE/remove_watermark.py\" your_clip.mp4 -o your_clip_clean.mp4"
