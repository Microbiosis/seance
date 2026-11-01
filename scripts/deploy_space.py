#!/usr/bin/env python3
"""Rebuild the Space's git tree and push it.

The Space went up through the Hub's commit endpoint, one commit per batch of
files.  That endpoint builds each new tree from the current head, so a batch
of deletions made after a batch of uploads can put back what the deletions
removed -- the probe files kept coming back.  Git does not have that
problem: one tree, one commit, one push.

So the staging directory is rebuilt from scratch -- only the console's own
files and the packed assets -- and pushed with --force, which replaces the
history on the remote with exactly this tree.

    python scripts/deploy_space.py
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SPACE = REPO / "space"
ASSETS = REPO / "app" / "assets"
APP = REPO / "app" / "app.py"
PROXY = "http://127.0.0.1:7897"
# the Space's own files live in the repository, not in this staging directory:
# app.py, requirements.txt, the front-matter README and the attributes file.
# A Space built from a directory the repository does not own cannot be rebuilt
# from a clone, and its dependency list is where a missing package hides.
SPACE_FILES = (
    (APP, "app.py"),
    (REPO / "app" / "requirements.txt", "requirements.txt"),
    (REPO / "app" / "README.md", "README.md"),
    (REPO / ".gitattributes", ".gitattributes"),
)


def run(args: list[str], cwd: Path | None = None) -> tuple[int, str]:
    p = subprocess.run(args, cwd=str(cwd) if cwd else None, capture_output=True, text=True)
    return p.returncode, (p.stdout + p.stderr)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote", default="origin")
    ap.add_argument("--branch", default="main")
    ap.add_argument("--message", default="seance console: live probe, recorded transcripts, packed assets")
    ap.add_argument("--no-push", action="store_true")
    args = ap.parse_args()

    # the packed assets are the ones the console reads; keep the two copies
    # identical so the Space and the repository ship the same bytes
    if (SPACE / "assets").exists():
        shutil.rmtree(SPACE / "assets")
    shutil.copytree(ASSETS, SPACE / "assets")

    # the console code has one home: app/app.py in the repository.  Copy it in,
    # so the Space can never run a stale copy of the app.
    for src, name in SPACE_FILES:
        if not src.exists():
            raise SystemExit(f"missing Space file: {src}")
        shutil.copyfile(src, SPACE / name)

    # a fresh index: only the files the Space should have
    code, out = run(["git", "read-tree", "--empty"], cwd=SPACE)
    if code != 0:
        print(out)
        return code
    code, out = run(["git", "add", "--force", ".gitattributes", "README.md", "app.py",
                     "requirements.txt", "assets"], cwd=SPACE)
    if code != 0:
        print(out)
        return code
    code, out = run(["git", "status", "--short"], cwd=SPACE)
    print(out.strip()[:2000] or "(tree matches the staging directory)")

    code, out = run(["git", "commit", "-q", "-m", args.message], cwd=SPACE)
    if code != 0:
        # nothing changed is not a failure: the tree is what it should be
        if "nothing to commit" not in out:
            print(out)
            return code
        print("(nothing changed; the existing commit is the tree we want)")
    code, out = run(["git", "log", "--oneline", "-1"], cwd=SPACE)
    print(out.strip())

    code, out = run(["git", "ls-tree", "-r", "--name-only", "HEAD"], cwd=SPACE)
    names = out.split()
    print(f"[deploy] the tree has {len(names)} entries")
    for n in names:
        if not (n.startswith("assets/") or n in (".gitattributes", "README.md", "app.py",
                                                 "requirements.txt")):
            print(f"[deploy] UNEXPECTED: {n}")

    if args.no_push:
        return 0

    run(["git", "config", "http.proxy", PROXY], cwd=SPACE)
    run(["git", "config", "http.postBuffer", "524288000"], cwd=SPACE)
    code, out = run(["git", "push", "--force", args.remote, f"HEAD:{args.branch}"], cwd=SPACE)
    print(out.strip()[-1500:])
    if code != 0:
        return code
    print(f"[deploy] pushed: https://huggingface.co/spaces/redmingwei/seance")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
