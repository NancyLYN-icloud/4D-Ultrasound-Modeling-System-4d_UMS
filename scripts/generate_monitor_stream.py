"""Generate monitor_stream.npz from image folder. img生成monitor_stream.npz
- Reads images from data/Datawithimage/image_monitor
- Resizes to 64x64 float32
- Builds timestamps from 0 to 60s across frames
- Computes feature_trace as mean intensity
- Saves data/raw/monitor_stream.npz with keys: frames, timestamps, feature_trace
"""
import os
from pathlib import Path
import numpy as np
from PIL import Image

SRC = Path("data/Datawithimage/image_monitor")
OUT = Path("data/raw") / "monitor_stream.npz"
OUT.parent.mkdir(parents=True, exist_ok=True)

files = sorted([p for p in SRC.iterdir() if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.tif', '.tiff')])
if not files:
    raise SystemExit(f"No images found in {SRC}")

imgs = []
for p in files:
    im = Image.open(p).convert('L')
    im = im.resize((64,64), Image.BILINEAR)
    arr = np.asarray(im, dtype=np.float32) / 255.0
    imgs.append(arr)

frames = np.stack(imgs, axis=0).astype(np.float32)  # (N,64,64)
N = frames.shape[0]
# timestamps from 0 to 60s inclusive
timestamps = np.linspace(0.0, 60.0, N, dtype=np.float32)
# simple feature: mean intensity per frame
feature_trace = frames.reshape(N, -1).mean(axis=1).astype(np.float32)

np.savez_compressed(OUT, frames=frames, timestamps=timestamps, feature_trace=feature_trace)
print(f"Wrote {OUT} with {N} frames, timestamps 0..60s")
