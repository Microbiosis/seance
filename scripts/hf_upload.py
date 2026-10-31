#!/usr/bin/env python3
"""Upload the Space to the Hub without huggingface_hub's HTTP client.

Why: on this machine `huggingface_hub`'s upload path kept getting its
connection reset by the local network path (the same channel that produced
empty replies on `git push` and curl 52/56 on HF downloads), while `curl
--http1.1` against the same endpoints works.  So we speak the Hub's commit
protocol directly:

    1. POST /api/spaces/<repo>/preupload/<rev>   -> upload mode per file
    2. LFS files: POST <repo>.git/info/lfs/objects/batch -> presigned PUT
    3. POST /api/spaces/<repo>/commit/<rev> with an ndjson body:
       regular files are inlined as base64, LFS files are referenced by oid

    python scripts/hf_upload.py --repo redmingwei/seance --folder space
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import time
from pathlib import Path

ENDPOINT = "https://huggingface.co"


def token() -> str:
    """The Hub token, without touching the network if we can help it.

    `hf auth token` contacts the Hub to validate the token, which takes a
    minute on this link -- and its stdout can carry a hint line, which would
    end up inside the Authorization header and get the connection dropped.
    So: the local token file first, the CLI only as a fallback, and only the
    line that looks like a token.
    """
    env = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if env:
        return env.strip()

    home = Path(os.environ.get("HF_HOME") or (Path.home() / ".cache" / "huggingface"))
    token_file = home / "token"
    if token_file.exists():
        value = token_file.read_text(encoding="utf-8").strip()
        if value:
            return value

    out = subprocess.run(["hf", "auth", "token"], capture_output=True, text=True,
                         timeout=60).stdout.strip()
    for line in reversed(out.splitlines()):
        line = line.strip()
        if line.startswith("hf_"):
            return line
    raise SystemExit("no HF token")


def curl(method: str, url: str, tok: str, body_file: Path | None = None,
         content_type: str | None = None, extra: list[str] | None = None,
         proxy: str | None = None):
    import os
    import re
    import tempfile

    out = Path(tempfile.gettempdir()) / f"hfcurl_{os.getpid()}_{abs(hash(url)) % 10**8}.out"
    # the answer file must not survive a failed request: curl leaves the
    # previous body in place when the connection dies, and a stale body looks
    # exactly like a fresh answer
    if out.exists():
        out.unlink()
    # "Expect:" disables curl's 100-continue handshake for bodies over 1 KB.
    # The status-code format deliberately has no newline: a literal newline
    # inside a Windows command-line argument is treated as a delimiter and
    # splits the argument, which makes curl fetch the format string as a URL.
    cmd = ["curl.exe", "-sS", "--http1.1", "-o", str(out), "-w", "%{http_code}",
           "--max-time", "40", "--connect-timeout", "10",
           "-X", method, "-H", f"Authorization: Bearer {tok}", "-H", "Expect:"]
    # The Hub is reached through the local proxy: the campus resolver sends
    # huggingface.co to a blackhole address, and a DIRECT route cannot
    # resolve the tunnel's fake-ip either.  Set SEANCE_PROXY to override.
    proxy = proxy or os.environ.get("SEANCE_PROXY") or os.environ.get("HTTPS_PROXY")
    if proxy:
        cmd += ["-x", proxy]
    if content_type:
        cmd += ["-H", f"Content-Type: {content_type}"]
    if body_file is not None:
        cmd += ["--data-binary", f"@{body_file}"]
    if extra:
        cmd += list(extra)
    cmd.append(url)
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if res.returncode != 0:
        raise RuntimeError(f"curl {method} failed: {res.stderr.strip()[:300]}")
    stdout = res.stdout.strip()
    code = int(stdout) if stdout.isdigit() else 0
    body = out.read_text(encoding="utf-8", errors="replace") if out.exists() else ""
    if not body.strip():
        body = stdout if code == 0 else ""
    return code, body


def curl_retry(method: str, url: str, tok: str, *args, tries: int = 12, **kw):
    """This machine's network channel drops connections mid-request (empty
    replies on the Hub's own endpoints).  Retry with backoff.

    NOTE: this tool deliberately does **not** touch the machine's network
    adapter.  An earlier version called `net-auto-recover.ps1 -Force` when a
    request failed, which (a) is a loop -- the tool is triggered by "the
    network is down" and it *makes* the network go down, so retry -> cycle ->
    retry never converges -- and (b) changes the user's machine state without
    asking, and (c) does not even fix this failure mode, which is a
    connection-level drop rather than a dead link.

    If every retry fails, the caller is expected to report that the network
    looks broken and leave the recovery decision to the user:
        powershell -NoProfile -ExecutionPolicy Bypass -File tools/net-auto-recover.ps1 -Force
    """
    last = None
    for attempt in range(1, tries + 1):
        try:
            code, body = curl(method, url, tok, *args, **kw)
            if code == 0:
                raise RuntimeError("no http status")
            return code, body
        except Exception as exc:  # noqa: BLE001
            last = exc
            print(f"[upload]   retry {attempt}/{tries} after {type(exc).__name__}: {str(exc)[:160]}")
            time.sleep(min(5 * attempt, 20))
    print("[upload] the network looks broken (every attempt got no HTTP reply).")
    print("[upload] this tool will not change your network state on its own.")
    print("[upload] if you want the WiFi cycle, run it yourself:")
    print("[upload]   powershell -NoProfile -ExecutionPolicy Bypass -File tools/net-auto-recover.ps1 -Force")
    raise RuntimeError(f"{method} {url} failed after {tries} tries: {last}")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="redmingwei/seance")
    ap.add_argument("--folder", default="space")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--message", default="seance console: live probe + recorded transcripts")
    ap.add_argument("--delete", nargs="*", default=[],
                    help="paths to remove from the repo (e.g. leftover static scaffold)")
    ap.add_argument("--delete-prefix", nargs="*", default=[],
                    help="paths to remove by prefix (e.g. probe_ leftovers)")
    ap.add_argument("--group-bytes", type=int, default=8000,
                    help="max base64 bytes per commit (the link refuses large bodies)")
    ap.add_argument("--tries", type=int, default=6,
                    help="attempts per commit before the group is split in two")
    ap.add_argument("--no-preupload", action="store_true",
                    help="skip the LFS preupload query (all files are small text blobs)")
    args = ap.parse_args()

    tok = token()
    folder = Path(args.folder)
    files = [p for p in sorted(folder.rglob("*")) if p.is_file() and ".git" not in p.parts]
    print(f"[upload] {len(files)} files from {folder}")

    # 1. preupload: ask the Hub which files are LFS and which are regular.
    # Two constraints shaped this: the payload carries one entry per file and
    # this link can only take ~12 KB per request, and the endpoint answers
    # slowly for a long list -- which is when the connection gets cut.  So the
    # list goes up in small batches.
    def preupload_batch(batch: list[Path]) -> dict:
        payload = {"files": []}
        for p in batch:
            data = p.read_bytes()
            payload["files"].append({
                "path": p.relative_to(folder).as_posix(),
                "sample": base64.b64encode(data[:32]).decode("ascii"),
                "size": len(data),
            })
        tmp = folder.parent / ".preupload.json"
        tmp.write_bytes(json.dumps(payload).encode("utf-8"))
        code, body = curl_retry("POST", f"{ENDPOINT}/api/spaces/{args.repo}/preupload/{args.revision}",
                                tok, tmp, "application/json")
        if code != 200:
            print(body[:500])
            raise SystemExit(1)
        return json.loads(body)

    modes: dict[str, dict] = {}
    BATCH = 12
    if args.no_preupload:
        print("[upload] skipping preupload (every file is a small text blob)")
    else:
        for i in range(0, len(files), BATCH):
            batch = files[i:i + BATCH]
            for f in preupload_batch(batch).get("files", []):
                modes[f["path"]] = f
            print(f"[upload] preupload {i + len(batch)}/{len(files)}")
        print(f"[upload] preupload done, {len(modes)} files")

    # 2. every file must be an ordinary git blob (see the LFS note above)
    lfs_files = [p for p in files if modes.get(p.relative_to(folder).as_posix(), {}).get("uploadMode") == "lfs"]
    if lfs_files:
        # The Hub's LFS decision is per-file and not fully predictable (two of
        # our three PNGs came back "regular", one came back "lfs").  LFS needs
        # a presigned-S3 PUT through a second endpoint that this machine's
        # network cannot reach -- and one LFS file is enough to send the whole
        # upload down that path.  So: refuse loudly instead of hanging.
        names = ", ".join(p.relative_to(folder).as_posix() for p in lfs_files)
        print(f"[upload] the Hub wants these files in LFS: {names}")
        print("[upload] LFS needs a presigned-S3 upload that does not work on this")
        print("[upload] network.  Ship such files as base64 inside a JSON asset")
        print("[upload] instead, so every file stays an ordinary git blob.")
        raise SystemExit(2)

    # 3. commits.  The link drops request bodies over roughly 100 KB, so the
    #    files are grouped into commits that stay under the limit instead of
    #    one commit per file.  Regular files are inlined as base64.
    groups: list[list[Path]] = []
    current: list[Path] = []
    current_bytes = 0
    # small files first: the console's own code matters more than the packed
    # assets, and a small request is the one this link answers
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

    def send_group(group: list[Path], depth: int = 0) -> None:
        """Commit one group of files, halving it if the link refuses it.

        The Hub's commit endpoint takes an ndjson body, and the size of that
        body is what this link will not carry: a 55 KB body was refused
        twelve times in a row while a small one went through every time, and
        the ceiling moves with the load rather than sitting at a fixed
        number.  So when a group is refused, the group is split and each
        half is sent on its own -- a commit that fails is retried as two
        smaller commits, down to a single file.
        """
        if not group:
            return
        lines = [{"key": "header", "value": {"summary": args.message}}]
        for p in group:
            rel = p.relative_to(folder).as_posix()
            data = p.read_bytes()
            lines.append({"key": "file", "value": {
                "content": base64.b64encode(data).decode("ascii"),
                "path": rel, "encoding": "base64"}})
        ndjson = "\n".join(json.dumps(x) for x in lines) + "\n"
        ntmp = folder.parent / ".commit.ndjson"
        # Binary write on purpose: the Hub's ndjson parser does not tolerate a
        # stray CR, and Python's text mode rewrites "\n" as "\r\n" on Windows
        # (which the Hub answers with "Unexpected end of JSON input").
        ntmp.write_bytes(ndjson.encode("utf-8"))
        names = ", ".join(p.relative_to(folder).as_posix() for p in group)
        try:
            code, body = curl_retry("POST", f"{ENDPOINT}/api/spaces/{args.repo}/commit/{args.revision}",
                                    tok, ntmp, "application/x-ndjson", tries=args.tries)
        except RuntimeError:
            if len(group) == 1:
                print(f"[upload] {names} could not be sent at all")
                raise
            mid = len(group) // 2
            print(f"[upload] group of {len(group)} refused; splitting it in two")
            send_group(group[:mid], depth + 1)
            send_group(group[mid:], depth + 1)
            return
        print(f"[upload] commit {names} -> {code} ({len(ndjson)/1024:.0f} KB)")
        if code not in (200, 201):
            print(body[:400])
            raise SystemExit(1)

    for group in groups:
        send_group(group)

    # 4. deletions (e.g. the static scaffold the Space was created with)
    deletions = list(args.delete) + [p for p in args.delete_prefix
                                    if p not in args.delete]
    if deletions:
        lines = [{"key": "header", "value": {"summary": args.message}}]
        for rel in deletions:
            lines.append({"key": "deletedFile", "value": {"path": rel}})
        ndjson = "\n".join(json.dumps(x) for x in lines) + "\n"
        ntmp = folder.parent / ".commit.ndjson"
        ntmp.write_bytes(ndjson.encode("utf-8"))
        code, body = curl_retry("POST", f"{ENDPOINT}/api/spaces/{args.repo}/commit/{args.revision}",
                                tok, ntmp, "application/x-ndjson")
        print(f"[upload] delete {len(deletions)} files -> {code}")
        if code not in (200, 201):
            print(body[:400])

    print(f"[upload] done: https://huggingface.co/spaces/{args.repo}")


if __name__ == "__main__":
    main()
