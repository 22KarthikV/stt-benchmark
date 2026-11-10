"""CLI: python -m sttbench {validate,run,report,demo}"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from .normalize import NORM_VERSION
from .providers.registry import build_providers
from .report import render_markdown, update_readme
from .runner import run_benchmark
from .truth import has_errors, validate_dataset


def load_dotenv(path: Path) -> dict[str, str]:
    """Minimal KEY=VALUE reader for the git-ignored .env; real environment variables win."""
    values: dict[str, str] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                values[k.strip()] = v.strip().strip("\"'")
    return values


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() or "unknown" if out.returncode == 0 else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _validate(args) -> list:
    issues = validate_dataset(
        Path(args.clips),
        Path(args.truth),
        strict=args.strict,
        min_wps=args.min_wps,
        max_wps=args.max_wps,
        stream_format=args.stream_format,
    )
    for issue in issues:
        print(issue, file=sys.stderr)
    return issues


def cmd_validate(args) -> int:
    issues = _validate(args)
    if has_errors(issues):
        print(f"{sum(i.severity == 'error' for i in issues)} validation error(s)", file=sys.stderr)
        return 2
    print("truth files OK")
    return 0


def cmd_run(args) -> int:
    if has_errors(_validate(args)):
        print("aborting before any provider call: fix the truth files first", file=sys.stderr)
        return 2
    env = {**load_dotenv(Path(args.env_file)), **os.environ}
    names = [n.strip() for n in args.providers.split(",") if n.strip()]
    try:
        providers, skipped = build_providers(names, env)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 2
    for name, reason in skipped.items():
        print(f"{name}: {reason}", file=sys.stderr)
    if not providers:
        print("no provider could run; nothing written", file=sys.stderr)
        return 1
    n_clips = len(list(Path(args.clips).glob("*.wav")))
    meta = {
        "run_date": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset": args.dataset_name,
        "n_clips": n_clips,
        "norm_version": NORM_VERSION,
        "git_sha": git_sha(),
    }
    results = asyncio.run(
        run_benchmark(
            providers,
            skipped,
            Path(args.clips),
            Path(args.truth),
            meta=meta,
            attempts=args.attempts,
            concurrency=args.concurrency,
        )
    )
    for name, p in results["providers"].items():
        if p["status"] == "ok":
            print(f"{name}: sent {p['audio_s_sent'] / 60:.2f} min of audio")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    md = render_markdown(results)
    (out / "results.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


def cmd_report(args) -> int:
    path = Path(args.results)
    results = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    if args.update_readme:
        readme = Path(args.readme)
        readme.write_text(
            update_readme(readme.read_text(encoding="utf-8"), results), encoding="utf-8"
        )
        print(f"updated {readme}")
    elif results is None:
        print("not yet run")
    else:
        md = render_markdown(results)
        path.with_name("results.md").write_text(md, encoding="utf-8")
        print(md)
    return 0


def cmd_demo(args) -> int:
    if args.serve:
        import uvicorn

        from .stream_demo.server import create_app, env_provider_factory

        for k, v in load_dotenv(Path(args.env_file)).items():
            os.environ.setdefault(k, v)
        app = create_app(env_provider_factory, Path(args.clips), Path(args.log))
        uvicorn.run(app, host=args.host, port=args.port)
        return 0
    from .stream_demo.client import replay

    summary = asyncio.run(replay(args.server, args.provider, args.file))
    return 0 if summary is not None else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sttbench")
    sub = p.add_subparsers(dest="cmd", required=True)

    def data_args(sp):
        sp.add_argument("--clips", default="data/clips")
        sp.add_argument("--truth", default="data/truth")
        sp.add_argument("--strict", action="store_true")
        sp.add_argument("--stream-format", action="store_true")
        sp.add_argument("--min-wps", type=float, default=0.5)
        sp.add_argument("--max-wps", type=float, default=6.0)

    v = sub.add_parser("validate", help="check truth files only")
    data_args(v)
    v.set_defaults(fn=cmd_validate)

    r = sub.add_parser("run", help="run the benchmark")
    data_args(r)
    r.add_argument("--providers", default="fake")
    r.add_argument("--out", default="results")
    r.add_argument("--concurrency", type=int, default=1)
    r.add_argument("--attempts", type=int, default=4)
    r.add_argument("--dataset-name", default="LibriSpeech test-clean (subset)")
    r.add_argument("--env-file", default=".env")
    r.set_defaults(fn=cmd_run)

    rp = sub.add_parser("report", help="regenerate results.md / README block")
    rp.add_argument("--results", default="results/results.json")
    rp.add_argument("--readme", default="README.md")
    rp.add_argument("--update-readme", action="store_true")
    rp.set_defaults(fn=cmd_report)

    d = sub.add_parser("demo", help="streaming demo (--serve starts the server)")
    d.add_argument("--serve", action="store_true")
    d.add_argument("--host", default="127.0.0.1")
    d.add_argument("--port", type=int, default=8000)
    d.add_argument("--clips", default="data/clips")
    d.add_argument("--log", default="results/stream_log.jsonl")
    d.add_argument("--env-file", default=".env")
    d.add_argument("--server", default="ws://127.0.0.1:8000")
    d.add_argument("--provider", default="fake")
    d.add_argument("--file", default="")
    d.set_defaults(fn=cmd_demo)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
