"""Upload this repository to GitHub through the REST git-data API.

The link that reaches GitHub from this machine has two hard limits:

  * any HTTP request body larger than roughly 100 KB is dropped mid-flight, so
    the git smart-HTTP protocol (`git-receive-pack`) never gets an answer --
    the pack goes out and the response never comes back;
  * a burst of POSTs from one exit address gets the connection closed without
    a reply after a handful of requests.

Plain REST requests survive both, as long as each body is small and the
requests are spaced out.  So the repository goes up as ordinary REST calls:

  1. one `POST /git/blobs` per file   (small base64 bodies, paced)
  2. one `POST /git/trees`             (one entry per file)
  3. one `POST /git/commits`           (the single commit)
  4. one `PATCH /git/refs/heads/main`  (point the branch at it)

Requests are issued with curl, which is what this link answers, and the
upload is resumable: every blob sha is cached on disk, so a run that dies
half-way can be restarted without spending the budget twice.

    python scripts/upload_github_rest.py --repo Microbiosis/seance --delay 12
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO / ".upload_cache.json"

# fixed identity and timestamp: the commit the API creates must have the same
# sha as the commit git builds locally from the same tree and metadata
AUTHOR = {
    "name": "Séance",
    "email": "seance@users.noreply.github.com",
    "date": "2026-10-30T00:00:00Z",
}


def gh_token() -> str:
    out = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True)
    value = out.stdout.strip()
    if not value:
        raise SystemExit("no gh token available")
    return value


def curl(method: str, url: str, payload: dict | None, token: str, tries: int = 5) -> dict:
    """One REST call.  curl is used because this link only answers curl.

    The link sometimes drops the connection right after the answer has been
    written, so curl reports no status code even though the body arrived
    intact.  The body is therefore the thing that decides whether the call
    worked.
    """
    body_file = REPO / ".upload_body.json"
    if payload is not None:
        body_file.write_bytes(json.dumps(payload).encode())
    cmd = [
        "curl.exe", "-s", "-o", str(body_file) + ".out", "-w", "%{http_code}",
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
        # the answer file must not survive a failed request: curl leaves the
        # previous body in place when the connection dies, and a stale body
        # looks exactly like a fresh answer
        out_file = Path(str(body_file) + ".out")
        if out_file.exists():
            out_file.unlink()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        code = proc.stdout.strip()
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
        # the body wins over the status line: this link truncates the
        # connection after a good answer more often than not
        if isinstance(answer, dict) and "sha" in answer:
            return answer
        if isinstance(answer, dict) and answer.get("message") and "Not Found" not in str(answer.get("message")):
            if attempt == tries - 1:
                raise RuntimeError(f"{method} {url} -> {answer.get('message')}")
            time.sleep(3.0 * (attempt + 1))
            continue
        time.sleep(3.0 * (attempt + 1))
    raise RuntimeError(f"{method} {url} -> no usable answer (last body: {raw[:200]})")


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "-C", str(REPO), "ls-files"], capture_output=True, text=True)
    return [p for p in out.stdout.splitlines() if p.strip()]


def load_cache() -> dict:
    if CACHE.exists():
        try:
            return json.loads(CACHE.read_text("utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def save_cache(cache: dict) -> None:
    CACHE.write_text(json.dumps(cache, indent=1, sort_keys=True), "utf-8")


def upload_blobs(repo: str, token: str, files: list[str], delay: float, inline: bool) -> dict:
    cache = load_cache()
    todo = [f for f in files if f not in cache]
    print(f"{len(files)} files, {len(files) - len(todo)} already cached, {len(todo)} to send")
    api = f"https://api.github.com/repos/{repo}"
    for index, path in enumerate(todo, start=1):
        raw = (REPO / path).read_bytes()
        payload = {"content": base64.b64encode(raw).decode(), "encoding": "base64"}
        body = json.dumps(payload).encode()
        if len(body) > 90_000:
            raise RuntimeError(f"{path}: body {len(body)} B is too large for one request")
        answer = None
        for attempt in range(1, 26):
            try:
                answer = curl("POST", f"{api}/git/blobs", payload, token)
                break
            except RuntimeError as exc:
                wait = min(15 * attempt, 45)
                print(f"  [{path}] attempt {attempt}/25 failed ({exc}); waiting {wait}s", flush=True)
                time.sleep(wait)
        if answer is None:
            raise RuntimeError(f"{path}: no answer after 6 attempts")
        expected = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        if answer.get("sha") != expected:
            # the answer does not belong to this file: the connection died and
            # the body of an earlier request was read instead
            raise RuntimeError(f"{path}: Hub answered sha {answer.get('sha')}, expected {expected}")
        cache[path] = answer["sha"]
        save_cache(cache)
        if index % 5 == 0 or index == len(todo):
            print(f"  blobs {index}/{len(todo)} (last {path})", flush=True)
        if delay:
            time.sleep(delay)
    return cache


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Microbiosis/seance")
    ap.add_argument("--message", default="seance: a seance rig that reads a brain (linear probe + local Gemma 3 4B)")
    ap.add_argument("--delay", type=float, default=12.0)
    ap.add_argument("--skip", default="", help="comma separated paths to leave out")
    ap.add_argument("--finish", action="store_true", help="only do tree/commit/ref")
    args = ap.parse_args()

    token = gh_token()
    api = f"https://api.github.com/repos/{args.repo}"
    skip = {s for s in args.skip.split(",") if s}

    cache = load_cache()
    if not args.finish:
        files = [f for f in tracked_files() if f not in skip]
        cache = upload_blobs(args.repo, token, files, args.delay, False)

    missing = [f for f in tracked_files() if f not in skip and f not in cache]
    if missing:
        print(f"missing {len(missing)} blobs, not finishing: {missing[:5]}", file=sys.stderr)
        return 1

    tree = {
        "base_tree": None,
        "tree": [
            {"path": p, "mode": "100644", "type": "blob", "sha": cache[p]}
            for p in sorted(cache) if p not in skip
        ],
    }
    tree_ans = curl("POST", f"{api}/git/trees", tree, token)
    print(f"  tree {tree_ans['sha']}")

    commit_ans = curl(
        "POST",
        f"{api}/git/commits",
        {
            "message": args.message,
            "tree": tree_ans["sha"],
            "parents": [],
            "author": AUTHOR,
            "committer": AUTHOR,
        },
        token,
    )
    commit_sha = commit_ans["sha"]
    print(f"  commit {commit_sha}")

    ref = curl("PATCH", f"{api}/git/refs/heads/main", {"sha": commit_sha, "force": True}, token)
    print(f"  main -> {ref['object']['sha']}")
    print(f"done: https://github.com/{args.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
