"""Generate scanner_sequence.npz from image folder.
- Reads images from the external data root Datawithimage/images
- Resizes to 512x512 float64
- Builds timestamps from 0 to 180s across frames
- Synthesizes positions (N,3) and orientations (N,3,3)
- Saves raw/scanner_sequence.npz with keys: frames, timestamps, positions, orientations
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.paths import data_path

SRC = data_path("Datawithimage", "images")
OUT = data_path("raw", "scanner_sequence.npz")
OUT.parent.mkdir(parents=True, exist_ok=True)

files = sorted([p for p in SRC.iterdir() if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.tif', '.tiff')])
if not files:
    raise SystemExit(f"No images found in {SRC}")

imgs = []
for p in files:
    im = Image.open(p).convert('L')
    im = im.resize((512,512), Image.BILINEAR)
    arr = np.asarray(im, dtype=np.float64) / 255.0
    imgs.append(arr)

frames = np.stack(imgs, axis=0).astype(np.float64)  # (N,512,512)
N = frames.shape[0]
# timestamps from 0 to 180s inclusive
timestamps = np.linspace(0.0, 180.0, N, dtype=np.float64)

import math

# synthesize positions and orientations to mimic a free-hand ultrasound sweep
rng = np.random.default_rng(42)
t = np.linspace(0.0, 1.0, N)

# Smooth free-hand trajectory: combine slow oscillations + drift
positions = np.zeros((N, 3), dtype=np.float64)
positions[:, 0] = 20.0 * np.sin(2 * math.pi * 0.5 * t) + 5.0 * np.sin(2 * math.pi * 0.05 * t)  # x jitter
positions[:, 1] = 10.0 * np.cos(2 * math.pi * 0.4 * t) + 3.0 * np.sin(2 * math.pi * 0.07 * t)  # y jitter
# z progresses forward (depth) with slight irregularity
positions[:, 2] = np.linspace(0.0, 150.0, N) + 5.0 * rng.normal(scale=1.0, size=N)

# orientations: small rotations around all axes that vary smoothly
def euler_to_matrix(alpha, beta, gamma):
    # rotations around X (alpha), Y (beta), Z (gamma) in degrees
    a = math.radians(alpha)
    b = math.radians(beta)
    g = math.radians(gamma)
    Rx = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    Ry = np.array([[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]])
    Rz = np.array([[math.cos(g), -math.sin(g), 0], [math.sin(g), math.cos(g), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

alphas = 5.0 * np.sin(2 * math.pi * 0.2 * t) + 2.0 * rng.normal(scale=0.5, size=N)
betas = 7.0 * np.sin(2 * math.pi * 0.15 * t + 0.5) + 2.0 * rng.normal(scale=0.5, size=N)
gammas = 3.0 * np.sin(2 * math.pi * 0.25 * t - 0.3) + 1.0 * rng.normal(scale=0.3, size=N)

orientations = np.zeros((N, 3, 3), dtype=np.float64)
for i in range(N):
    orientations[i] = euler_to_matrix(alphas[i], betas[i], gammas[i])

np.savez_compressed(OUT, frames=frames, timestamps=timestamps, positions=positions, orientations=orientations)
print(f"Wrote {OUT} with {N} frames, timestamps 0..180s, positions shape {positions.shape}, orientations shape {orientations.shape}")
