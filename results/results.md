Run 2026-11-11T00:11:31Z (UTC) · dataset LibriSpeech test-clean (subset) (30 clips) · normaliser norm-v1 · harness 49ef53f

| Provider | WER | Clips | Failures | p50 (s) | p95 (s) | Model |
|---|---|---|---|---|---|---|
| deepgram | n/a | 0 | n/a | n/a | n/a | skipped: no key (DEEPGRAM_API_KEY) |
| vosk | 11.13% | 30 | 0 | 1.14 | 1.83 | vosk:vosk-model-small-en-us-0.15 |
| assemblyai | 2.49% | 30 | 0 | 3.76 | 4.08 | universal-3-5-pro |

- vosk: latency percentiles over n=30 successful clips (linear interpolation; at this sample size p95 is close to the maximum).
- assemblyai: latency percentiles over n=30 successful clips (linear interpolation; at this sample size p95 is close to the maximum).
