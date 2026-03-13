from __future__ import annotations

from pathlib import Path
import math
import sys
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.paths import data_path


TEST_DIR = data_path("test", "image")
OUT_DIR = data_path("test", "preview")


def build_montage(source_dir: Path, out_path: Path, *, rows: int, cols: int, thumb_size: tuple[int, int], title: str) -> None:
    files = sorted(source_dir.glob("*.png"))
    if not files:
        raise FileNotFoundError(f"No PNG images found in {source_dir}")

    count = rows * cols
    if len(files) >= count:
        step = (len(files) - 1) / max(count - 1, 1)
        selected = [files[min(int(round(i * step)), len(files) - 1)] for i in range(count)]
    else:
        selected = files + [files[-1]] * (count - len(files))

    thumb_w, thumb_h = thumb_size
    margin = 12
    header_h = 42
    canvas_w = cols * thumb_w + (cols + 1) * margin
    canvas_h = rows * thumb_h + (rows + 1) * margin + header_h
    canvas = Image.new("L", (canvas_w, canvas_h), color=245)
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 12), title, fill=20)

    for idx, path in enumerate(selected):
        image = Image.open(path).convert("L").resize((thumb_w, thumb_h), Image.BILINEAR)
        row = idx // cols
        col = idx % cols
        x = margin + col * (thumb_w + margin)
        y = header_h + margin + row * (thumb_h + margin)
        canvas.paste(image, (x, y))
        draw.rectangle((x, y, x + thumb_w, y + thumb_h), outline=90, width=1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main() -> None:
    build_montage(
        TEST_DIR / "monitor",
        OUT_DIR / "monitor_montage.png",
        rows=3,
        cols=4,
        thumb_size=(128, 128),
        title="Monitor stream preview (selected phases)",
    )
    build_montage(
        TEST_DIR / "scanner",
        OUT_DIR / "scanner_montage.png",
        rows=3,
        cols=4,
        thumb_size=(160, 160),
        title="Freehand scanner stream preview (selected sweep frames)",
    )
    print(f"Wrote {OUT_DIR / 'monitor_montage.png'}")
    print(f"Wrote {OUT_DIR / 'scanner_montage.png'}")


if __name__ == "__main__":
    main()
