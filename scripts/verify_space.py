#!/usr/bin/env python3
"""Render the live Space and fail if the app raises.

The Space can be RUNNING and still be broken: a missing package in the
container only shows up when the script runs, long after the health check
says "ok".  This drives the Space over the same websocket the browser uses
and reports what the app actually rendered.

    python scripts/verify_space.py                       # the published Space
    python scripts/verify_space.py redmingwei-seance.hf.space

Needs `websockets` (the protobuf definitions come with streamlit itself).
"""
from __future__ import annotations

import asyncio
import sys

import websockets
from streamlit.proto.BackMsg_pb2 import BackMsg
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg

DEFAULT_HOST = "redmingwei-seance.hf.space"
# what the console must show; these are the audited numbers
EXPECTED_METRICS = {
    "single-trial top-1": "4.5%",
    "chance": "0.5%",
    "shuffled-label control": "2.0%",
}
# text that only appears if the séance panel really rendered
REQUIRED_TEXT = (
    "Hold the séance",
    "Impression",
    "The ghost",
    "control: the medium read somebody else's brain",
    "control: the medium read nothing at all",
    "the signal behind the impression",
)


def _messages(node):
    """Every sub-message of a protobuf tree, whatever streamlit calls them.

    The field holding the element moved between streamlit versions
    (`new_element`, `add`, `add_block`, ...), so walk the tree instead of
    naming a field.
    """
    if not hasattr(node, "ListFields"):
        return
    for _field, value in node.ListFields():
        if type(value).__name__.startswith("Repeated"):
            for item in value:
                if hasattr(item, "ListFields"):
                    yield item
        elif hasattr(value, "ListFields"):
            yield value


def collect(node, out: list[str], depth: int = 0) -> None:
    """Every human-readable string in a protobuf tree."""
    if depth > 14:
        return
    for field, value in node.ListFields():
        if isinstance(value, str):
            if value.strip():
                out.append(value.strip())
        elif type(value).__name__.startswith("Repeated"):
            for item in value:
                if isinstance(item, str):
                    if item.strip():
                        out.append(item.strip())
                elif hasattr(item, "ListFields"):
                    collect(item, out, depth + 1)
        elif hasattr(value, "ListFields"):
            collect(value, out, depth + 1)


def _metric_pair(metric) -> tuple[str, str]:
    """(label, value) of a metric message.

    streamlit renamed the value field to body, but both are field 2 of the
    same message -- read the field number, not the name.
    """
    label = value = ""
    for field, v in metric.ListFields():
        if not isinstance(v, str):
            continue
        if field.number == 1:
            label = v
        elif field.number == 2:
            value = v
    return label, value


def metrics_in(node, out: list[tuple[str, str]], depth: int = 0) -> None:
    """Pick every metric element out of the tree, wherever it is nested."""
    if depth > 14:
        return
    name = node.DESCRIPTOR.name
    if name == "Element":
        if node.WhichOneof("type") == "metric":
            out.append(_metric_pair(node.metric))
            return
    elif name == "Metric":
        out.append(_metric_pair(node))
        return
    for sub in _messages(node):
        metrics_in(sub, out, depth + 1)


async def render(host: str) -> tuple[list[str], list[tuple[str, str]], object | None]:
    texts: list[str] = []
    metrics: list[tuple[str, str]] = []
    exception = None
    url = f"wss://{host}/_stcore/stream"
    async with websockets.connect(url, subprotocols=["stream"],
                                  open_timeout=90, max_size=None) as ws:
        back = BackMsg()
        back.rerun_script.page_name = ""
        back.rerun_script.query_string = ""
        await ws.send(back.SerializeToString())
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 180
        while loop.time() < deadline:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=60)
            except asyncio.TimeoutError:
                break
            msg = ForwardMsg()
            try:
                msg.ParseFromString(raw)
            except Exception:
                continue
            kind = msg.WhichOneof("type")
            if kind == "exception":
                exception = msg.exception
            elif kind == "delta":
                collect(msg.delta, texts)
                metrics_in(msg.delta, metrics)
            elif kind == "scriptFinished":
                break
    return texts, metrics, exception


def main() -> int:
    host = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_HOST
    print(f"rendering https://{host}/")
    texts, metrics, exception = asyncio.run(render(host))

    if exception is not None:
        print("\nTHE APP RAISED:")
        lines: list[str] = []
        collect(exception, lines)
        for line in lines[:30]:
            print("   |", line[:140])
        return 1

    seen: list[str] = []
    for t in texts:
        if t not in seen:
            seen.append(t)
    print(f"\nno exception; {len(seen)} distinct strings rendered")
    for label, value in metrics:
        print(f"   metric | {label} = {value}")
    missing = [p for p in REQUIRED_TEXT if not any(p in t for t in seen)]
    for probe in REQUIRED_TEXT:
        print(f"   {'has' if probe not in missing else 'MISSING'} | {probe}")

    bad = [(l, v) for l, v in metrics
           if l in EXPECTED_METRICS and v != EXPECTED_METRICS[l]]
    if missing:
        print("\nmissing on the page:", ", ".join(missing))
    if bad:
        print("\nwrong numbers on the page:")
        for l, v in bad:
            print(f"   {l}: expected {EXPECTED_METRICS[l]}, page says {v}")
    if missing or bad:
        return 1
    print("\nOK: the live console renders and shows the audited numbers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
