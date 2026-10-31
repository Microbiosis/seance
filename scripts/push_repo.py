"""Push the repository to GitHub through the REST git-data API.

One blob per file, then one tree, one commit and one branch move.  Files are
sent in the order that matters (the figures the post links to first, the
packed demo assets last), the upload is resumable through a cache of blob
shas, and every answer is checked against the sha git computes for the same
bytes.

The REST API rather than `git push` because this path is resumable: a link
that drops large bodies does not have to be restarted from the beginning, and
every blob is verified on its own.  (The link drops bodies only when the
traffic leaves through the proxy; routed directly, `git push` is fine -- the
Space is pushed that way, see scripts/deploy_space.py.)

    python scripts/push_repo.py --repo Microbiosis/seance
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


def local_head() -> tuple[str, str]:
    """The message and the date of the local commit being published.

    A fixed message makes every published commit read the same, which tells a
    reader nothing about what changed between them; a fixed date makes the
    history lie about when the work happened.  Both come from the local
    commit instead.
    """
    out = subprocess.run(["git", "-C", str(REPO), "log", "-1",
                           "--pretty=%aI%x00%s%x00%b"], capture_output=True, text=True)
    parts = out.stdout.split("\0")
    date = parts[0].strip() if parts else ""
    subject = parts[1].strip() if len(parts) > 1 else ""
    body = parts[2].strip() if len(parts) > 2 else ""
    message = subject + (f"\n\n{body}" if body else "")
    return message or MESSAGE, date or AUTHOR["date"]


def gh_token() -> str:
    out = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True)
    value = out.stdout.strip()
    if not value:
        raise SystemExit("no gh token available")
    return value


def curl(method: str, url: str, payload: dict | None, token: str, tries: int = 3) -> dict:
    """One REST call.

    `Expect:` is what curl adds for bodies over 1 KB, and the middlebox on
    this link swallows that handshake, so it is switched off explicitly.
    """
    body_file = REPO / ".push_body.json"
    out_file = Path(str(body_file) + ".out")
    if payload is not None:
        body_file.write_bytes(json.dumps(payload).encode())
    cmd = [
        "curl.exe", "-s", "-o", str(out_file), "-w", "%{http_code}",
        "--max-time", "45", "-X", method,
        "-H", f"Authorization: Bearer {token}",
        "-H", "Accept: application/vnd.github+json",
        "-H", "X-GitHub-Api-Version: 2022-11-28",
        "-H", "User-Agent: seance-uploader",
        "-H", "Expect:",
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
        # a blob answer carries "sha", a tree answer carries "tree", and a
        # ref answer carries the sha under "object" -- check for all three,
        # or a successful update-ref is mistaken for a silent failure
        if isinstance(answer, dict) and (
            "sha" in answer or "tree" in answer or "object" in answer
        ):
            return answer
        if isinstance(answer, dict) and answer.get("message"):
            raise RuntimeError(f"{method} {url} -> {answer['message']}")
        time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"{method} {url} -> no usable answer")


def tracked() -> list[str]:
    out = subprocess.run(["git", "-C", str(REPO), "ls-files"], capture_output=True, text=True)
    return sorted(p for p in out.stdout.splitlines() if p.strip())


def order(files: list[str]) -> list[str]:
    """Figures first (the post links to them), then the code, then the assets."""
    def rank(path: str) -> tuple[int, str]:
        if path.startswith("results/") and path.endswith(".png"):
            return (0, path)
        if path.startswith("results/"):
            return (1, path)
        if path.startswith("app/assets/"):
            return (4, path)
        if path in ("README.md", "docs/submission.md"):
            return (2, path)
        return (3, path)
    return sorted(files, key=rank)


def blob_sha(raw: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


def head_sha(api: str, token: str) -> str | None:
    """The commit `main` currently points at, if there is one."""
    try:
        answer = curl("GET", f"{api}/git/ref/heads/main", None, token)
    except RuntimeError:
        return None
    return answer.get("object", {}).get("sha")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Microbiosis/seance")
    ap.add_argument("--wait", type=float, default=12.0)
    ap.add_argument("--tries", type=int, default=40)
    ap.add_argument("--skip-assets", action="store_true")
    ap.add_argument("--finish-only", action="store_true")
    args = ap.parse_args()

    token = gh_token()
    api = f"https://api.github.com/repos/{args.repo}"
    files = tracked()
    if args.skip_assets:
        files = [f for f in files if not f.startswith("app/assets/")]
    files = order(files)

    cache: dict = {}
    if CACHE.exists():
        try:
            cache = json.loads(CACHE.read_text("utf-8"))
        except json.JSONDecodeError:
            cache = {}

    if not args.finish_only:
        for path in files:
            raw = (REPO / path).read_bytes()
            expected = blob_sha(raw)
            if cache.get(path) == expected:
                continue
            payload = {"content": base64.b64encode(raw).decode(), "encoding": "base64"}
            for attempt in range(1, args.tries + 1):
                try:
                    answer = curl("POST", f"{api}/git/blobs", payload, token)
                except RuntimeError as exc:
                    print(f"  [{path}] {exc}", flush=True)
                    time.sleep(args.wait)
                    continue
                if answer.get("sha") != expected:
                    print(f"  [{path}] stale answer {answer.get('sha')}", flush=True)
                    time.sleep(args.wait)
                    continue
                cache[path] = expected
                CACHE.write_text(json.dumps(cache, indent=1, sort_keys=True), "utf-8")
                print(f"  blob {path} ({len(raw)} B) ok", flush=True)
                break
            else:
                print(f"  [{path}] gave up", flush=True)
                return 2
            time.sleep(1.5)

    missing = [f for f in files if cache.get(f) != blob_sha((REPO / f).read_bytes())]
    if missing:
        print(f"missing {len(missing)} blobs: {missing[:3]}", flush=True)
        return 1

    tree = {
        "base_tree": None,
        "tree": [{"path": p, "mode": "100644", "type": "blob", "sha": cache[p]} for p in files],
    }
    for attempt in range(1, args.tries + 1):
        try:
            tree_ans = curl("POST", f"{api}/git/trees", tree, token)
            break
        except RuntimeError as exc:
            print(f"  tree attempt {attempt}: {exc}", flush=True)
            time.sleep(args.wait)
    else:
        return 1
    print(f"  tree {tree_ans['sha']}", flush=True)

    # the new commit hangs off the one that is already there, so the published
    # repository keeps a history instead of being a single snapshot
    head = head_sha(api, token)
    print(f"  parent {head or '(none: first commit)'}", flush=True)

    message, date = local_head()
    author = {**AUTHOR, "date": date}
    print(f"  message: {message.splitlines()[0]}", flush=True)

    for attempt in range(1, args.tries + 1):
        try:
            commit = curl("POST", f"{api}/git/commits",
                          {"message": message, "tree": tree_ans["sha"],
                           "parents": [head] if head else [],
                           "author": author, "committer": author}, token)
            break
        except RuntimeError as exc:
            print(f"  commit attempt {attempt}: {exc}", flush=True)
            time.sleep(args.wait)
    else:
        return 1
    print(f"  commit {commit['sha']}", flush=True)

    for attempt in range(1, args.tries + 1):
        try:
            ref = curl("PATCH", f"{api}/git/refs/heads/main",
                       {"sha": commit["sha"], "force": True}, token)
            break
        except RuntimeError as exc:
            print(f"  ref attempt {attempt}: {exc}", flush=True)
            time.sleep(args.wait)
    else:
        return 1
    print(f"  main -> {ref.get('object', {}).get('sha', ref.get('sha'))}", flush=True)
    print(f"done: https://github.com/{args.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
