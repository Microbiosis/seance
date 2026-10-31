"""Smoke test: can we load the local ghost and make it speak?

    python scripts/smoke_ghost.py [path/to/gemma-3-4b-it-bnb-4bit]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from seance.ghost import Ghost, Sitting, build_prompt  # noqa: E402

DEFAULT_MODEL = ROOT / "models" / "gemma-3-4b-it-bnb-4bit"


def main() -> None:
    model_dir = sys.argv[1] if len(sys.argv) > 1 else str(DEFAULT_MODEL)
    t0 = time.time()
    ghost = Ghost(model_dir, max_new_tokens=80, seed=0)
    print(f"[smoke] loaded in {time.time()-t0:.1f}s", flush=True)

    s = Sitting(
        key="k", subject=1, image_idx=0, rep=60,
        true_concept="aardvark", sighted_concept="aardvark", clarity=0.42,
    )
    t1 = time.time()
    msg = ghost.speak(build_prompt(s))
    print(f"[smoke] {time.time()-t1:.1f}s for one message:", flush=True)
    print("-" * 60)
    print(msg)
    print("-" * 60)

    s2 = Sitting(key="k2", subject=1, image_idx=1, rep=61,
                 true_concept="airplane", sighted_concept="airplane", clarity=0.8)
    msg2 = ghost.speak(build_prompt(s2))
    print(msg2)


if __name__ == "__main__":
    main()
