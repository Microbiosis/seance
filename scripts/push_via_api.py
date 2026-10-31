#!/usr/bin/env python3
"""Push the repository to GitHub through the REST API.

`git push` over HTTPS is unreliable from this machine (the connection is cut
mid-transfer), while the REST API answers fine.  The tree here is small, so we
upload it file by file with the contents API.

    python scripts/push_via_api.py --repo Microbiosis/seance
"""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = "https://api.github.com"


def get_token() -> str:
    tok = None
    try:
        tok = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True,
                             timeout=30).stdout.strip()
    except Exception:  # noqa: BLE001
        tok = None
    if not tok:
        import os

        tok = os.environ.get("GITHUB_TOKEN", "")
    if not tok:
        raise SystemExit("no GitHub token: run `gh auth login` or set GITHUB_TOKEN")
    return tok


def request(method: str, url: str, token: str, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    req.add_header("User-Agent", "seance-push")
    if data:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = resp.read().decode()
    return json.loads(body) if body else {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Microbiosis/seance")
    ap.add_argument("--branch", default="main")
    ap.add_argument("--message", default="seance: console, audit and results")
    args = ap.parse_args()

    token = get_token()
    files = subprocess.run(["git", "ls-files"], capture_output=True, text=True,
                           cwd=ROOT, timeout=30).stdout.split()
    files = [f for f in files if f]
    print(f"[push] {len(files)} files -> {args.repo}")

    ok = 0
    for path in files:
        content = (ROOT / path).read_bytes()
        b64 = base64.b64encode(content).decode()
        url = f"{API}/repos/{args.repo}/contents/{path}"
        payload = {"message": f"update {path}", "content": b64, "branch": args.branch}
        # create-or-update needs the current sha when the file exists
        try:
            existing = request("GET", url, token)
            if "sha" in existing:
                payload["sha"] = existing["sha"]
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                print(f"[push] {path}: GET failed {exc.code}")
        for attempt in range(3):
            try:
                request("PUT", url, token, payload)
                ok += 1
                print(f"[push]   ok {path}")
                break
            except urllib.error.HTTPError as exc:
                body = exc.read().decode()[:200]
                print(f"[push]   retry {path}: {exc.code} {body}")
                if exc.code in (401, 403, 404, 422):
                    break
            except Exception as exc:  # noqa: BLE001
                print(f"[push]   retry {path}: {type(exc).__name__}: {exc}")
        else:
            print(f"[push] FAILED {path}")
    print(f"[push] uploaded {ok}/{len(files)}")


if __name__ == "__main__":
    main()
