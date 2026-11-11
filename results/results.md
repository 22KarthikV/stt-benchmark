Run 2026-11-11T13:13:20Z (UTC) · dataset LibriSpeech test-clean (subset) (30 clips) · normaliser norm-v1 · harness df65c3e

| Provider | WER | Clips | Failures | p50 (s) | p95 (s) | Model |
|---|---|---|---|---|---|---|
| vosk | 11.13% | 30 | 0 | 1.83 | 2.98 | vosk:vosk-model-small-en-us-0.15 |
| assemblyai | 2.49% | 30 | 0 | 3.76 | 5.33 | universal-3-5-pro |
| deepgram | 2.99% | 30 | 0 | 0.92 | 1.48 | nova-3 |

- vosk: latency percentiles over n=30 successful clips (linear interpolation; at this sample size p95 is close to the maximum).
- assemblyai: latency percentiles over n=30 successful clips (linear interpolation; at this sample size p95 is close to the maximum).
- deepgram: latency percentiles over n=30 successful clips (linear interpolation; at this sample size p95 is close to the maximum).
