import json

import pytest
from fastapi.testclient import TestClient

from sttbench.providers.base import StreamEvent
from sttbench.providers.fake import FakeProvider
from sttbench.stream_demo.metrics import compute_stream_metrics, summarize_stream
from sttbench.stream_demo.server import create_app
from tests.conftest import make_wav


def ev(kind, t, start=None, end=None, text="x"):
    return StreamEvent(kind=kind, text=text, t_recv=t, audio_start_s=start, audio_end_s=end)


def test_metrics_with_provider_audio_offsets():
    # 20 chunks of 0.1 s sent at t = 100.0, 100.1, ...
    sent = [100.0 + 0.1 * i for i in range(20)]
    events = [
        ev("partial", 100.35, start=0.0),
        ev("final", 101.0, start=0.0, end=0.8),
        ev("partial", 101.2, start=0.8),
        ev("final", 102.5, start=0.8, end=1.9),
    ]
    m = compute_stream_metrics(sent, events, chunk_s=0.1)
    assert len(m) == 2
    # utt 1: first byte sent t=100.0, first partial 100.35, last audio chunk (idx 7) sent 100.7
    assert m[0]["time_to_first_partial"] == pytest.approx(0.35)
    assert m[0]["final_latency"] == pytest.approx(101.0 - 100.7)
    # utt 2: start chunk idx 8 sent 100.8; end 1.9 -> idx 18 sent 101.8
    assert m[1]["time_to_first_partial"] == pytest.approx(101.2 - 100.8)
    assert m[1]["final_latency"] == pytest.approx(102.5 - 101.8)


def test_metrics_fallback_without_offsets():
    sent = [10.0 + 0.1 * i for i in range(10)]
    events = [ev("partial", 10.25), ev("final", 11.5)]
    m = compute_stream_metrics(sent, events, chunk_s=0.1)
    assert m[0]["time_to_first_partial"] == pytest.approx(0.25)
    # last chunk sent before the final arrived (idx 9, t=10.9)
    assert m[0]["final_latency"] == pytest.approx(11.5 - 10.9)


def test_metrics_utterance_without_partial():
    sent = [0.0, 0.1]
    m = compute_stream_metrics(sent, [ev("final", 0.5, start=0.0, end=0.2)], chunk_s=0.1)
    assert m[0]["time_to_first_partial"] is None


def test_summarize_p50():
    s = summarize_stream(
        [
            {"time_to_first_partial": 0.1, "final_latency": 1.0},
            {"time_to_first_partial": 0.3, "final_latency": 3.0},
            {"time_to_first_partial": None, "final_latency": 2.0},
        ]
    )
    assert s["utterances"] == 3
    assert s["final_latency_p50"] == pytest.approx(2.0)
    assert s["time_to_first_partial_p50"] == pytest.approx(0.2)


@pytest.fixture
def app_client(tmp_path):
    clips = tmp_path / "clips"
    make_wav(clips / "demo.wav", 1.0)
    log = tmp_path / "stream_log.jsonl"

    def factory(name):
        if name != "fake":
            raise ValueError(name)
        return FakeProvider(utterances=["hello there"], chunks_per_utterance=5)

    app = create_app(factory, clips, log, realtime=False)
    return TestClient(app), log


def collect(ws):
    msgs = []
    while True:
        m = ws.receive_json()
        msgs.append(m)
        if m["kind"] in ("done", "error"):
            return msgs


def test_ws_streams_partials_and_finals(app_client):
    client, log = app_client
    with client.websocket_connect("/ws?provider=fake&file=demo.wav") as ws:
        msgs = collect(ws)
    kinds = [m["kind"] for m in msgs]
    assert "partial" in kinds and "final" in kinds and kinds[-1] == "done"
    assert msgs[-1]["summary"]["utterances"] >= 1
    assert msgs[0].get("fake") is True or any(m.get("fake") for m in msgs)
    lines = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines()]
    assert lines and "final_latency" in lines[0]


def test_ws_unknown_provider_errors(app_client):
    client, _ = app_client
    with client.websocket_connect("/ws?provider=nope&file=demo.wav") as ws:
        assert collect(ws)[-1]["kind"] == "error"


def test_ws_rejects_path_traversal_and_missing_file(app_client):
    client, _ = app_client
    for f in ("../secret.wav", "missing.wav"):
        with client.websocket_connect(f"/ws?provider=fake&file={f}") as ws:
            assert collect(ws)[-1]["kind"] == "error"


def test_index_page_served(app_client):
    client, _ = app_client
    r = client.get("/")
    assert r.status_code == 200 and "partial" in r.text.lower()
