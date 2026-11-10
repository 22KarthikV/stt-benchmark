"""File-replay client: connects to the demo server and prints partial/final lines."""

import json
from collections.abc import Callable
from urllib.parse import urlencode

import websockets


async def replay(
    server: str, provider: str, file: str, out: Callable[[str], None] = print
) -> dict | None:
    """Return the final summary dict, or None if the server reported an error."""
    url = f"{server.rstrip('/')}/ws?{urlencode({'provider': provider, 'file': file})}"
    summary = None
    async with websockets.connect(url) as ws:
        async for raw in ws:
            msg = json.loads(raw)
            kind = msg["kind"]
            if kind == "info":
                tag = " [FAKE: scripted output, not real ASR]" if msg.get("fake") else ""
                out(f"provider={msg['provider']} model={msg['model']}{tag}")
            elif kind in ("partial", "final"):
                out(f"[{msg['t']:7.2f}s] {kind:7s} {msg['text']}")
            elif kind == "error":
                out(f"error: {msg['message']}")
                return None
            elif kind == "done":
                summary = msg["summary"]
                out(f"summary: {json.dumps(summary)}")
    return summary
