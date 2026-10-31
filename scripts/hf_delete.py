"""Remove the leftover probe files from the Space.

The Space was created with a static scaffold and then used for a series of
upload probes, so the repository still carries those: fourteen "lad_*"
ladder probes, five "probe_*" samples, a stray "probe.txt", the scaffold's
index.html and style.css, and four unpacked JSON assets that the packed
parts replaced.

The deletions go up in small batches.  One request carrying all of them
was refused on this link, while a request carrying six was answered in
three seconds -- so the batch size is small and each batch is retried.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ENDPOINT = "https://huggingface.co"

LAD = [f"lad_{p}{n}.txt" for p in ("c", "ctl", "s") for n in (0, 3000, 5000, 7000, 9000)]
PROBE = ["probe.txt", "probe_32k_off_0.txt", "probe_a.txt", "probe_c.txt",
         "probe_ctl_0.txt", "probe_t_0.txt"]
SCAFFOLD = ["index.html", "style.css", "_t.txt"]
OLD_ASSETS = ["assets/audit.json", "assets/corpus.json",
              "assets/features.json", "assets/figures.json"]


def token() -> str:
    home = Path(os.environ.get("HF_HOME") or (Path.home() / ".cache" / "huggingface"))
    value = (home / "token").read_text(encoding="utf-8").strip()
    if value:
        return value
    raise SystemExit("no HF token")


def send(paths: list[str], tok: str, repo: str, tries: int = 8) -> bool:
    body = Path(tempfile.gettempdir()) / f"hfdel_{os.getpid()}.json"
    body.write_text(json.dumps({"summary": "remove the upload probes and the scaffold",
                                 "deletedFile": paths}), "utf-8")
    out = Path(tempfile.gettempdir()) / f"hfdel_{os.getpid()}.out"
    url = f"{ENDPOINT}/api/spaces/{repo}/commit/main"
    for attempt in range(1, tries + 1):
        if out.exists():
            out.unlink()
        cmd = ["curl.exe", "-s", "--http1.1", "-o", str(out), "-w", "%{http_code}",
               "--max-time", "90", "-X", "POST",
               "-H", f"Authorization: Bearer {tok}", "-H", "Expect:",
               "-H", "Content-Type: application/json"]
        proxy = os.environ.get("SEANCE_PROXY")
        if proxy:
            cmd += ["-x", proxy]
        cmd += ["--data-binary", f"@{body}", url]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        code = p.stdout.strip()
        if code in ("200", "201"):
            print(f"[delete] {len(paths)} paths removed -> {code}", flush=True)
            return True
        detail = ""
        try:
            detail = out.read_text("utf-8", errors="replace")[:200]
        except OSError:
            pass
        print(f"[delete] attempt {attempt}/{tries} got {code!r} {detail}", flush=True)
        time.sleep(min(3 * attempt, 12))
    return False


def main() -> int:
    repo = sys.argv[1] if len(sys.argv) > 1 else "redmingwei/seance"
    tok = token()
    paths = LAD + PROBE + SCAFFOLD + OLD_ASSETS
    print(f"[delete] {len(paths)} paths to remove from {repo}", flush=True)
    # One request for all of them.  The Hub builds the new tree from the
    # current head, so a second batch of deletions can put back what the
    # first batch removed -- the body is small enough to send in one go.
    ok = send(paths, tok, repo, tries=12)
    print("[delete] done" if ok else "[delete] failed", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
