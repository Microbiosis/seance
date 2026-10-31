#!/usr/bin/env python3
"""Make the figures small enough to be served from the repo.

The link that uploads this project can only carry ~12 KB per request, so a
57 KB PNG can never be pushed to GitHub -- and the post links the figures as
raw URLs, which have to resolve to a real file.  So the figures are re-rendered
smaller and palette-quantised until each one fits.

    python scripts/shrink_figures.py
"""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image

RESULTS = Path(r"E:\seance\results")
LIMIT = 10_500  # bytes, comfortably under the ~12 KB the link carries


def shrink(src: Path, dst: Path) -> None:
    im = Image.open(src).convert("RGB")
    best = None
    for scale in (1.0, 0.8, 0.66, 0.55):
        w = max(320, int(im.width * scale))
        h = max(200, int(im.height * scale))
        small = im.resize((w, h), Image.LANCZOS)
        for colors in (256, 64, 32):
            pal = small.convert("P", palette=Image.ADAPTIVE, colors=colors)
            buf = io.BytesIO()
            pal.save(buf, format="PNG", optimize=True)
            data = buf.getvalue()
            if best is None or len(data) < len(best[0]):
                best = (data, w, h, colors)
            if len(data) <= LIMIT:
                dst.write_bytes(data)
                print(f"[fig] {src.name}: {w}x{h} palette{colors} -> {len(data)/1024:.1f} KB")
                return
    data, w, h, colors = best
    dst.write_bytes(data)
    print(f"[fig] {src.name}: {w}x{h} palette{colors} -> {len(data)/1024:.1f} KB  (over the limit)")


def main() -> None:
    for name in ("fig_evidence", "fig_levels", "fig_audit"):
        src = RESULTS / f"{name}.png"
        if src.exists():
            shrink(src, RESULTS / f"{name}.png")


if __name__ == "__main__":
    main()
