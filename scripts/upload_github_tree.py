"""Upload the repository to GitHub with as few requests as possible.

The blob-by-blob uploader needs one request per file.  This one leans on the
fact that the trees API accepts a file's *content* inline: text files go into
a tree directly, so a whole directory becomes a single request.  Only the
files that cannot be inlined -- the binary figures and the packed asset parts
-- need a blob of their own.

    python scripts/upload_github_tree.py --repo Microbiosis/seance

Request budget for this repository:

    15  asset parts (45 KB each, one blob each)
     4  figures (binary, one blob each)
     6  trees (code, scripts, docs, app, results, root)
     2  commit + branch update
    --
    27 requests instead of 55
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO / ".upload_cache.json"
AUTHOR = {
    "name": "Séance",
    "email": "seance@users.noreply.github.com",
    "date": "2026-10-30T00:00:00Z",
}
MESSAGE = "seance: a seance rig that reads a brain (linear probe + local Gemma 3 4B)"

# a directory becomes one tree; its text files are inlined, its binary files
# are referenced by the blob they already have
TREES = {
    "seance": None,
    "scripts": None,
    "docs": None,
    "results": None,
    "app": None,
}
ROOT_FILES = [".gitattributes", ".gitignore", "LICENSE", "README.md"]
ASSET_DIR = "app/assets"


def gh_token() -> str:
    out = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True)
    value = out.stdout.strip()
    if not value:
        raise SystemExit("no gh token available")
    return value


def curl(method: str, url: str, payload: dict | None, token: str, tries: int = 6) -> dict:
    """One REST call.  The body is the thing that decides success: this link
    truncates the connection after a good answer often enough that curl
    reports no status code."""
    body_file = REPO / ".tree_body.json"
    out_file = Path(str(body_file) + ".out")
    if payload is not None:
        body_file.write_bytes(json.dumps(payload).encode())
    cmd = [
        "curl.exe", "-s", "-o", str(out_file), "-w", "%{http_code}",
        "--max-time", "60", "-X", method,
        "-H", f"Authorization: Bearer {token}",
        "-H", "Accept: application/vnd.github+json",
        "-H", "X-GitHub-Api-Version: 2022-11-28",
        "-H", "User-Agent: seance-uploader",
    ]
    if payload is not None:
        cmd += ["-H", "Content-Type: application/json", "--data-binary", f"@{body_file}"]
    cmd.append(url)

    for attempt in range(tries):
        if out_file.exists():
            out_file.unlink()
        subprocess.run(cmd, capture_output=True, text=True)
        raw = ""
        try:
            raw = out_file.read_text("utf-8", "replace")
        except OSError:
            pass
        answer = None
        if raw.strip():
            try:
                answer = json.loads(raw)
            except json.JSONDecodeError:
                answer = None
        if isinstance(answer, dict) and ("sha" in answer or "tree" in answer):
            return answer
        if isinstance(answer, dict) and answer.get("message"):
            if attempt == tries - 1:
                raise RuntimeError(f"{method} {url} -> {answer['message']}")
            time.sleep(4.0 * (attempt + 1))
            continue
        time.sleep(4.0 * (attempt + 1))
    raise RuntimeError(f"{method} {url} -> no usable answer")


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "-C", str(REPO), "ls-files"], capture_output=True, text=True)
    return sorted(p for p in out.stdout.splitlines() if p.strip())


def is_text(path: str) -> bool:
    raw = (REPO / path).read_bytes()
    if b"\0" in raw[:4096]:
        return False
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def blob_sha(raw: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


def upload_blob(api: str, token: str, path: str, cache: dict) -> str:
    if path in cache:
        return cache[path]
    raw = (REPO / path).read_bytes()
    payload = {"content": base64.b64encode(raw).decode(), "encoding": "base64"}
    for attempt in range(1, 26):
        try:
            answer = curl("POST", f"{api}/git/blobs", payload, token)
            break
        except RuntimeError as exc:
            wait = min(15 * attempt, 45)
            print(f"  [{path}] attempt {attempt}/25 failed ({exc}); waiting {wait}s", flush=True)
            time.sleep(wait)
    else:
        raise RuntimeError(f"{path}: no answer after 25 attempts")
    expected = blob_sha(raw)
    if answer.get("sha") != expected:
        raise RuntimeError(f"{path}: Hub answered sha {answer.get('sha')}, expected {expected}")
    cache[path] = expected
    CACHE.write_text(json.dumps(cache, indent=1, sort_keys=True), "utf-8")
    print(f"  blob {path} -> {expected[:10]}", flush=True)
    return expected


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Microbiosis/seance")
    ap.add_argument("--delay", type=float, default=6.0)
    ap.add_argument("--skip-assets", action="store_true",
                    help="leave app/assets out (the packed demo assets)")
    args = ap.parse_args()

    token = gh_token()
    api = f"https://api.github.com/repos/{args.repo}"
    files = tracked_files()
    if args.skip_assets:
        files = [f for f in files if not f.startswith("app/assets/")]
    cache: dict = {}
    if CACHE.exists():
        try:
            cache = json.loads(CACHE.read_text("utf-8"))
        except json.JSONDecodeError:
            cache = {}

    # 1. the files that cannot be inlined: binary ones, and the big packed parts
    blobs: dict[str, str] = {}
    for path in files:
        big = (REPO / path).stat().st_size > 20_000
        if big or not is_text(path):
            blobs[path] = upload_blob(api, token, path, cache)
            if args.delay:
                time.sleep(args.delay)
    print(f"{len(blobs)} blobs uploaded")

    # 2. one tree per directory, built from the top down
    def make_tree(prefix: str) -> str:
        entries = []
        subdirs: dict[str, list[str]] = {}
        for path in files:
            rest = path[len(prefix) + 1:] if prefix else path
            if "/" in rest:
                subdirs.setdefault(rest.split("/")[0], []).append(path)
                continue
            if path in blobs:
                entries.append({"path": rest, "mode": "100644", "type": "blob", "sha": blobs[path]})
            else:
                entries.append({"path": rest, "mode": "100644", "type": "blob",
                                "content": (REPO / path).read_text("utf-8")})
        for sub in subdirs:
            entries.append({"path": sub, "mode": "040000", "type": "tree",
                            "sha": make_tree(f"{prefix}/{sub}" if prefix else sub)})
        answer = curl("POST", f"{api}/git/trees", {"base_tree": None, "tree": entries}, token)
        print(f"  tree {prefix or '/'} -> {answer['sha']} ({len(entries)} entries)", flush=True)
        if args.delay:
            time.sleep(args.delay)
        return answer["sha"]

    root_sha = make_tree("")

    # 3. one commit, one branch move
    commit = curl("POST", f"{api}/git/commits",
                  {"message": MESSAGE, "tree": root_sha, "parents": [], "author": AUTHOR,
                   "committer": AUTHOR}, token)
    print(f"  commit {commit['sha']}")
    ref = curl("PATCH", f"{api}/git/refs/heads/main", {"sha": commit["sha"], "force": True}, token)
    print(f"  main -> {ref['object']['sha']}")
    print(f"done: https://github.com/{args.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
