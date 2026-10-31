#!/usr/bin/env python3
"""Pack the Space assets into small parts.

Every asset is gzipped, base64-encoded and split into parts that fit the
upload link, with a manifest describing how to put them back together.  The
console reads only through the manifest, so the local run and the deployed
Space load exactly the same bytes.

    python scripts/pack_assets.py --assets app/assets --out space/assets
"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path

# base64 characters per part.  The parts are sized for the worst case the
# upload link was measured under (see the README, "How the assets ship, and
# why"): a 5000 character part becomes a ~6.7 KB request body.
PART_CHARS = 5000

FILES = ("probe.json", "features.json", "corpus.json", "audit.json", "figures.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default="app/assets")
    ap.add_argument("--out", default="space/assets")
    args = ap.parse_args()

    src = Path(args.assets)
    dst = Path(args.out)
    if dst.exists():
        for stale in dst.iterdir():
            if stale.is_file():
                stale.unlink()
    dst.mkdir(parents=True, exist_ok=True)

    manifest = {
        "note": (
            "Assets are gzipped, base64-encoded and split into parts. Reassemble "
            "with scripts/pack_assets.py (or app/app.py's loader)."
        ),
        "encoding": "gzip+base64",
        "assets": {},
    }

    total_parts = 0
    for name in FILES:
        path = src / name
        if not path.exists():
            print(f"[pack] missing {path}")
            continue
        raw = path.read_bytes()
        # mtime=0: the same input has to pack to the same bytes, or every
        # rebuild dirties the whole asset directory
        blob = base64.b64encode(gzip.compress(raw, 9, mtime=0)).decode("ascii")
        stem = name.replace(".json", "")
        parts = []
        for i in range(0, len(blob), PART_CHARS):
            chunk = blob[i:i + PART_CHARS]
            part_name = f"{stem}.p{i // PART_CHARS:03d}"
            (dst / part_name).write_text(chunk, encoding="ascii")
            parts.append(part_name)
        manifest["assets"][name] = {
            "parts": parts,
            "raw_bytes": len(raw),
            "packed_bytes": len(blob),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        total_parts += len(parts)
        print(f"[pack] {name}: {len(raw)/1024:.1f} KB -> {len(parts)} parts")

    (dst / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[pack] {total_parts} parts + manifest.json -> {dst}")

    # every file must fit the link
    biggest = max((p.stat().st_size, p.name) for p in dst.iterdir() if p.is_file())
    print(f"[pack] biggest file: {biggest[1]} ({biggest[0]} bytes)")


if __name__ == "__main__":
    main()
