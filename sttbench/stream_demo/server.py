"""FastAPI app: replays a WAV clip over a WebSocket to a provider's stream() in real time."""

import asyncio
import json
import os
import time
from collections.abc import Callable
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

from ..audio import AudioError, iter_pcm_chunks, read_pcm, read_wav, require_stream_format
from ..providers.base import Provider, StreamEvent
from ..providers.registry import build_providers
from .metrics import compute_stream_metrics, summarize_stream

CHUNK_MS = 100
STATIC = Path(__file__).parent / "static"


def env_provider_factory(name: str) -> Provider:
    providers, skipped = build_providers([name], os.environ)
    if name in skipped:
        raise ValueError(skipped[name])
    return providers[name]


def create_app(
    provider_factory: Callable[[str], Provider],
    clips_dir: Path,
    log_path: Path,
    *,
    realtime: bool = True,
) -> FastAPI:
    app = FastAPI(title="stt-benchmark streaming demo")
    clips_dir = Path(clips_dir)
    chunk_s = CHUNK_MS / 1000

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/clips")
    def clips():
        return JSONResponse(sorted(p.name for p in clips_dir.glob("*.wav")))

    @app.websocket("/ws")
    async def ws(websocket: WebSocket, provider: str = "fake", file: str = ""):
        await websocket.accept()

        async def fail(message: str) -> None:
            await websocket.send_json({"kind": "error", "message": message})
            await websocket.close()

        if not file or Path(file).name != file or not file.endswith(".wav"):
            return await fail("invalid file name")
        path = clips_dir / file
        if not path.is_file():
            return await fail(f"no such clip: {file}")
        try:
            require_stream_format(read_wav(path))
            pcm = read_pcm(path)
            prov = provider_factory(provider)
        except (AudioError, ValueError, KeyError) as e:
            return await fail(str(e))

        sent: list[float] = []
        t0 = time.monotonic()

        async def paced_chunks():
            for i, chunk in enumerate(iter_pcm_chunks(pcm, CHUNK_MS)):
                if realtime:
                    delay = t0 + i * chunk_s - time.monotonic()
                    if delay > 0:
                        await asyncio.sleep(delay)
                sent.append(time.monotonic())
                yield chunk

        events: list[StreamEvent] = []
        try:
            await websocket.send_json(
                {"kind": "info", "provider": provider, "model": prov.model, "fake": not prov.live}
            )
            async for ev in prov.stream(paced_chunks()):
                events.append(ev)
                await websocket.send_json(
                    {"kind": ev.kind, "text": ev.text, "t": round(ev.t_recv - t0, 3)}
                )
            metrics = compute_stream_metrics(sent, events, chunk_s)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as f:
                for m in metrics:
                    row = {"provider": provider, "file": file, "fake": not prov.live, **m}
                    f.write(json.dumps(row) + "\n")
            await websocket.send_json({"kind": "done", "summary": summarize_stream(metrics)})
            await websocket.close()
        except WebSocketDisconnect:
            return
        except Exception as e:
            await fail(f"{type(e).__name__}: {e}")

    return app
