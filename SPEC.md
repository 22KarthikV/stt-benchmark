# stt-benchmark — SPEC (draft for approval)

Status: **APPROVED (2026-11-10).** Decisions in §2 and §14 are final.

## 1. Purpose

A small harness that (1) scores speech-to-text providers on real audio (WER, latency),
(2) streams audio over WebSockets to show partial/final transcripts with measured latency,
and (3) validates reference transcripts *before* scoring so bad truth files fail loudly.
Portfolio project: correctness and honesty over feature count.

## 2. Decisions already made

| Topic | Decision |
|---|---|
| Providers | AssemblyAI, Deepgram, **Vosk (local, open source)** — proposed third, see §2a |
| Dataset | LibriSpeech `test-clean` subset, CC BY 4.0, fetched by a script |
| Streaming input | File replay only, paced at real time. No browser mic in v1. |
| Keys | Read from environment variables only (`.env`, git-ignored). CI and unit tests use fakes. |
| Cost | **Free only. No paid subscription, no card on file.** See §2a. |
| Tooling | Python 3.11+ (developed on 3.15), pip + venv (uv optional), ruff, pytest |

## 2a. Free-only policy

- **AssemblyAI**: per its [billing docs](https://assemblyai.com/docs/billing-and-pricing), new accounts get $50 free credit, no card needed; when it runs out and no card/auto-pay is set, API access pauses (no surprise charges).
- **Deepgram**: third-party sources report a $200 signup credit with no card; sources disagree on expiry and I did not reach Deepgram's official page. **Verify on signup** before relying on it.
- **Vosk** (Apache 2.0, offline Kaldi-based, `pip install vosk`, small English model from Alpha Cephei): costs nothing, needs no key, and supports true incremental partial/final results, so the streaming demo is meaningful. Model licence and size to be confirmed on the official model page. It is also a useful contrast (small local model vs. cloud APIs). Alternative: `faster-whisper` (MIT, CPU int8), but it is batch-oriented, so streaming would be simulated; not proposed.
- A full run is ~30 clips x ~10 s ≈ 5 min of audio per provider, a tiny fraction of free credit. The harness prints audio minutes sent per provider so you can watch usage.
- Free credits can expire or change; the README states the date terms were checked and that they may change. Vosk/AssemblyAI/Deepgram models and credits are never assumed available in CI (CI uses fakes only).
- Vosk is an optional extra (`uv sync --extra vosk`) so the core install stays light.

## 3. Honesty rules (hard requirements)

- Every number in README comes from `results/results.json` via `python -m sttbench report --update-readme`. No hand-typed or estimated figures.
- If `results/results.json` does not exist, README shows "not yet run".
- Results carry: run date (UTC), dataset name + clip count, provider model identifiers, normaliser version, harness git SHA.
- Latency is wall-clock measured by the harness; the definition is documented (§6).
- Anything not verified (e.g. an adapter never run against the live API) is labelled "untested against live API" in README.

## 4. Package layout

```
sttbench/
  __init__.py
  __main__.py        # CLI entry (argparse)
  normalize.py       # text normalisation (versioned)
  wer.py             # jiwer wrapper using normalize
  truth.py           # truth-file validator
  audio.py           # WAV read/validate, PCM chunker
  providers/
    base.py          # Provider ABC, Transcript, StreamEvent, errors
    fake.py          # deterministic fake for tests / offline demo
    assemblyai.py
    deepgram.py
    <third>.py
    registry.py
  retry.py           # backoff helper
  runner.py          # benchmark loop, per-clip error isolation
  report.py          # results.json -> results.md, README table block
  stream_demo/
    server.py        # FastAPI app + /ws
    client.py        # file-replay client, latency logging
    static/index.html  # minimal page showing partial/final lines
tests/  data/{clips,truth,README.md,fetch_librispeech.py}  results/  docs/
```

## 5. Provider interface

```python
class Provider(ABC):
    name: str

    async def transcribe_file(self, path: Path) -> Transcript: ...
    def stream(self, pcm_chunks: AsyncIterator[bytes]) -> AsyncIterator[StreamEvent]: ...
```

- `Transcript(text, provider, model, duration_s, latency_s)`.
- `StreamEvent(kind: "partial"|"final", text, t_recv)`; `t_recv` is monotonic time stamped by the harness on receipt.
- Errors: `TransientProviderError` (retried: 429, 5xx, network timeouts) vs `PermanentProviderError` (4xx auth/bad request; not retried).
- Keys read from env only (`ASSEMBLYAI_API_KEY`, `DEEPGRAM_API_KEY`, ...). Missing key → provider reported as "skipped: no key", never silently scored as 0.

### Adapter facts gathered from current official docs (via docs search, this session)

These must be **re-verified against the live docs at implementation time**; model names in particular change often (AssemblyAI docs currently show several different model names across pages).

- **AssemblyAI file**: `POST /v2/upload` (raw bytes) → `upload_url`; `POST https://api.assemblyai.com/v2/transcript` with `audio_url` and `speech_models` (plural; singular `speech_model` is deprecated there); poll `GET /v2/transcript/{id}` until `status` is `completed` or `error`; auth header `authorization: <key>`.
- **AssemblyAI streaming**: `wss://streaming.assemblyai.com/v3/ws?sample_rate=16000&speech_model=<model>`; header `Authorization: <key>`; binary PCM16 frames (docs say 50–1000 ms per chunk); server messages `Begin`, `Turn` (`transcript`, `end_of_turn`), `Termination`; client sends `{"type":"Terminate"}` at end.
- **Deepgram file**: `POST https://api.deepgram.com/v1/listen?model=nova-3&smart_format=true`, header `Authorization: Token <key>`, body = audio bytes with `Content-Type: audio/wav`; transcript at `results.channels[0].alternatives[0].transcript`.
- **Deepgram streaming**: `wss://api.deepgram.com/v1/listen?encoding=linear16&sample_rate=16000`; raw audio requires `encoding` and `sample_rate`; `{"type":"KeepAlive"}` and `{"type":"CloseStream"}` control messages; Results carry `is_final`.
- Model names are configurable via env (`ASSEMBLYAI_MODEL`, `DEEPGRAM_MODEL`) and recorded in results.

## 6. Benchmark

CLI: `python -m sttbench run --providers a,b,c --clips data/clips [--out results] [--concurrency 1]`
Also: `validate` (truth check only), `report` (regenerate md/README block), `demo` (streaming client).

- Order: validate all truth files **first**; any validation error aborts before any provider call (exit code 2, message lists every problem).
- Per clip per provider: `transcribe_file` with retries (default 4 attempts, exponential backoff 1s/2s/4s + jitter). A clip that still fails is recorded `{status:"failed", error}` and the run continues.
- **File latency** = wall-clock from request start to final transcript, including upload/polling for AssemblyAI (documented, since providers differ in shape). Reported as p50/p95.
- **WER** per provider = corpus-level (total edits / total reference words over successful clips) via jiwer; per-clip WER also stored. Failed clips are excluded from WER and counted in `failures`; table shows both.
- Output `results/results.json` (raw per-clip data, metadata) and `results/results.md` (table: provider, WER, clips, failures, p50, p95, model).
- With only 30-ish clips, p95 is nearly the max; the report notes n beside it.

## 7. Normalisation (documented in data/README.md and README)

Applied identically to reference and hypothesis, version string `norm-v1`:
1. Unicode NFKC, lowercase.
2. Expand numbers to words (e.g. `3` → `three`) via a small deterministic rule set; LibriSpeech references are already spelled out, so this mainly fixes provider output.
3. Remove punctuation except intra-word apostrophes; hyphens → space.
4. Collapse whitespace.
Not done: spelling variants (colour/color), contractions, filler words. These choices change WER and are listed under limitations. Final number-handling library choice (own rules vs. `num2words`) decided in implementation; keeps deps minimal.

## 8. Truth-file validation (`truth.py`)

Checks, each producing a named error code and message with file path:
1. `MISSING_TRUTH` — clip without `data/truth/<stem>.txt` (and orphan truth without clip, warning→error in strict mode).
2. `EMPTY_TRANSCRIPT` — empty / whitespace-only after normalisation.
3. `DURATION_MISMATCH` — words-per-second outside plausible bounds for speech (default 0.5–6.0 wps, configurable) given WAV duration; catches truncated or wrong-length text.
4. `MISALIGNED` — **optional cross-check**: heuristic checks that don't need a provider: non-speech characters ratio, repeated-token ratio, non-alphabetic text, language-script mismatch. A stronger check (ASR sanity pass vs. a provider; flags WER > threshold) is available as `--sanity-provider`, off by default because it costs API calls and cannot run in CI.
5. `BAD_AUDIO` — unreadable WAV, wrong sample rate/channels for the stream demo.
Honest limit: heuristics catch gross errors (wrong clip's text, truncation), not a single wrong word. Documented as such.
Required test: a deliberately mislabelled fixture (text of clip B attached to audio A) must be rejected. With no ASR in CI, this is caught by the duration/length heuristic, so the fixture pairs a long clip with a very short transcript; the spec admits that a same-length mislabel would pass the offline heuristic and that only `--sanity-provider` could catch it.

## 9. Streaming demo

- FastAPI app: `GET /` page, `WS /ws?provider=<name>&file=<clip>`; server reads WAV, converts/validates to 16 kHz mono 16-bit PCM, sends **100 ms chunks = 1600 samples = 3200 bytes**, paced at real time, to the provider's `stream()`, and relays `partial`/`final` JSON to the browser.
- Chunker: `iter_pcm_chunks(pcm, chunk_ms=100)`; last chunk may be shorter; tested for exact byte sizes.
- Metrics logged per utterance (JSON lines to `results/stream_log.jsonl`): `time_to_first_partial` = first partial `t_recv` − time the first audio byte of that utterance was sent; `final_latency` = final `t_recv` − time the last audio byte of that utterance was sent. Summary prints p50 of `final_latency` and time-to-first-partial.
- Utterance boundaries come from the provider's final events; audio-send timestamps are kept per chunk to attribute them. Stated caveat: file replay measures provider latency plus network, not microphone/capture latency.
- Runs offline against the fake provider (used in tests and for a no-key demo, clearly labelled "FAKE").

## 10. Tests (written first, run in CI with no network/keys)

- WER known strings (identical = 0, one sub / ins / del, empty hypothesis) + normalisation edge cases (case, punctuation, apostrophes, hyphens, numbers, unicode, whitespace).
- Truth validator: one test per error code + mislabelled fixture rejected + valid fixture accepted + aggregated error reporting.
- Provider contract with fake provider: file + stream, error classification, retry-with-backoff (injected sleep), per-clip failure doesn't abort run, missing key → skipped.
- Chunker frame sizes (exact multiples, remainder, empty input, odd byte count rejected).
- Report: results.json → md produces expected table; README block says "not yet run" when no results.
- Adapters: request construction and response parsing tested against recorded-shape fixtures I write from the docs (mock transport, e.g. `httpx.MockTransport`). These prove parsing logic only, **not** live-API correctness; README says so until a real run happens.
- Streaming server: WebSocket test with fake provider via FastAPI TestClient.

## 11. Dependencies

Runtime: `httpx`, `websockets`, `fastapi`, `uvicorn`, `jiwer`. Optional extra `vosk`. Dev: `pytest`, `pytest-asyncio`, `ruff`. Third-provider SDKs avoided; raw HTTP/WS so behaviour matches docs.

## 12. CI

`.github/workflows/ci.yml`: on push/PR → `uv sync`, `ruff check`, `ruff format --check`, `pytest`. No secrets.

## 13. Environment

Development machine has network access, so `fetch_librispeech.py`, the Vosk model download and real benchmark runs are executed locally. API keys come from environment variables only. CI never touches the network or keys. Demo clips for unit tests are synthetic WAVs (generated tones/silence) with fake transcripts, marked as test fixtures and kept out of `data/clips`.

## 14. Decisions (resolved)

1. Third provider: Vosk (local). 2. Subset: 30 clips. 3. Fetch and benchmark run locally. 4. Own number rules, no `num2words`.

## 15. Commit plan

spec → scaffold/CI → tests (red) → normalize/wer → truth → audio/chunker → provider base/fake/retry/runner → report/CLI → adapters → streaming demo → data fetch script/docs → README.
