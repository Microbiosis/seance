#!/usr/bin/env python3
"""Upload the Space to the Hub with several workers.

The Hub's commit endpoint is one commit per request, and this link drops a
good share of them no matter how small the body is -- measured on the day:
a 20 KB body went through four times out of six and a 100 KB body three
times out of six, while a 55 KB body was refused twelve times in a row.
The refusal is not a function of size, so the answer is to keep trying, and
the way to keep trying inside the deadline is to try in parallel.

Each worker owns its own slice of the file list, so no two workers ever
touch the same path.  A group that is refused is split in two and each half
is sent on its own, down to a single file.

    python scripts/hf_upload_parallel.py --repo redmingwei/seance --folder space --workers 6
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ENDPOINT = "https://huggingface.co"


def token() -> str:
    env = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if env:
        return env.strip()
    home = Path(os.environ.get("HF_HOME") or (Path.home() / ".cache" / "huggingface"))
    token_file = home / "token"
    if token_file.exists():
        value = token_file.read_text(encoding="utf-8").strip()
        if value:
            return value
    raise SystemExit("no HF token")


def curl(method: str, url: str, tok: str, body_file: Path, content_type: str,
         proxy: str | None) -> tuple[int, str]:
    out = Path(tempfile.gettempdir()) / f"hfp_{os.getpid()}_{threading_id()}.out"
    if out.exists():
        out.unlink()
    cmd = ["curl.exe", "-s", "--http1.1", "-o", str(out), "-w", "%{http_code}",
           "--max-time", "90", "--connect-timeout", "15",
           "-X", method, "-H", f"Authorization: Bearer {tok}", "-H", "Expect:",
           "-H", f"Content-Type: {content_type}"]
    proxy = proxy or os.environ.get("SEANCE_PROXY") or os.environ.get("HTTPS_PROXY")
    if proxy:
        cmd += ["-x", proxy]
    cmd += ["--data-binary", f"@{body_file}", url]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    code = p.stdout.strip()
    body = ""
    try:
        body = out.read_text("utf-8", errors="replace")
    except OSError:
        pass
    if not code.isdigit():
        code = "0"
    return int(code), body


def threading_id() -> int:
    import threading
    return threading.get_ident()


class Uploader:
    def __init__(self, repo: str, folder: Path, tok: str, message: str,
                 tries: int, proxy: str | None):
        self.repo = repo
        self.folder = folder
        self.tok = tok
        self.message = message
        self.tries = tries
        self.proxy = proxy
        self.tmp = Path(tempfile.gettempdir()) / f"hfp_body_{os.getpid()}_{threading_id()}.ndjson"
        self.done = 0

    def send(self, group: list[Path]) -> bool:
        lines = [{"key": "header", "value": {"summary": self.message}}]
        for p in group:
            data = p.read_bytes()
            lines.append({"key": "file", "value": {
                "content": base64.b64encode(data).decode("ascii"),
                "path": p.relative_to(self.folder).as_posix(),
                "encoding": "base64"}})
        ndjson = "\n".join(json.dumps(x) for x in lines) + "\n"
        # binary write on purpose: the Hub's ndjson parser rejects a stray CR
        self.tmp.write_bytes(ndjson.encode("utf-8"))

        url = f"{ENDPOINT}/api/spaces/{self.repo}/commit/main"
        for attempt in range(1, self.tries + 1):
            code, body = curl("POST", url, self.tok, self.tmp,
                              "application/x-ndjson", self.proxy)
            if code in (200, 201):
                names = ", ".join(p.relative_to(self.folder).as_posix() for p in group)
                print(f"[{threading_id() % 1000:>4}] ok {len(group)} file(s) "
                      f"{len(ndjson)/1024:.0f} KB: {names[:70]}", flush=True)
                self.done += len(group)
                return True
            time.sleep(min(2 * attempt, 10))
        return False

    def send_group(self, group: list[Path]) -> None:
        if not group:
            return
        if self.send(group):
            return
        if len(group) == 1:
            name = group[0].relative_to(self.folder).as_posix()
            print(f"[{threading_id() % 1000:>4}] GAVE UP on {name}", flush=True)
            raise RuntimeError(f"{name} could not be sent")
        mid = len(group) // 2
        print(f"[{threading_id() % 1000:>4}] group of {len(group)} refused; splitting",
              flush=True)
        self.send_group(group[:mid])
        self.send_group(group[mid:])


def remote_paths(repo: str, tok: str, proxy: str | None) -> set[str]:
    """The paths the Space already has, so an interrupted run can resume."""
    url = f"{ENDPOINT}/api/spaces/{repo}/tree/main?recursive=true"
    cmd = ["curl.exe", "-s", "--http1.1", "--max-time", "60",
           "-H", f"Authorization: Bearer {tok}"]
    proxy = proxy or os.environ.get("SEANCE_PROXY") or os.environ.get("HTTPS_PROXY")
    if proxy:
        cmd += ["-x", proxy]
    cmd.append(url)
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    try:
        entries = json.loads(p.stdout)
    except Exception:
        return set()
    return {e["path"] for e in entries if isinstance(e, dict) and e.get("type") == "file"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="redmingwei/seance")
    ap.add_argument("--folder", default="space")
    ap.add_argument("--message", default="seance console: live probe + recorded transcripts")
    ap.add_argument("--group-bytes", type=int, default=12000)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--tries", type=int, default=8)
    ap.add_argument("--resume", action="store_true",
                    help="skip the files the Space already has")
    args = ap.parse_args()

    tok = token()
    folder = Path(args.folder)
    files = [p for p in sorted(folder.rglob("*")) if p.is_file() and ".git" not in p.parts]

    if args.resume:
        have = remote_paths(args.repo, tok, None)
        if have:
            before = len(files)
            files = [p for p in files if p.relative_to(folder).as_posix() not in have]
            print(f"[parallel] resuming: {before - len(files)} of {before} files are "
                  f"already on the Space", flush=True)

    print(f"[parallel] {len(files)} files from {folder}", flush=True)

    groups: list[list[Path]] = []
    current: list[Path] = []
    current_bytes = 0
    for p in sorted(files, key=lambda x: (x.stat().st_size, str(x))):
        size = len(base64.b64encode(p.read_bytes()))
        if current and current_bytes + size > args.group_bytes:
            groups.append(current)
            current = []
            current_bytes = 0
        current.append(p)
        current_bytes += size
    if current:
        groups.append(current)
    print(f"[parallel] {len(groups)} groups of about {args.group_bytes} base64 bytes, "
          f"{args.workers} workers", flush=True)

    # the workers get disjoint slices, so two of them never write the same path
    slices: list[list[list[Path]]] = [[] for _ in range(args.workers)]
    for i, group in enumerate(groups):
        slices[i % args.workers].append(group)

    up = Uploader(args.repo, folder, tok, args.message, args.tries, None)
    errors: list[str] = []

    def run_slice(slice_: list[list[Path]]) -> None:
        for group in slice_:
            try:
                up.send_group(group)
            except Exception as exc:  # noqa: BLE001
                errors.append(str(exc))

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(run_slice, slices))
    dt = time.time() - t0
    print(f"[parallel] {up.done}/{len(files)} files sent in {dt:.0f}s", flush=True)
    if errors:
        print(f"[parallel] {len(errors)} failures:", flush=True)
        for e in errors[:10]:
            print(f"  {e}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
