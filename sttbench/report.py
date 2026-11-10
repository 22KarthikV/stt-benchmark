"""results.json -> results.md and the README results block. No number is typed by hand."""

from typing import Any

README_START = "<!-- RESULTS:START -->"
README_END = "<!-- RESULTS:END -->"

NOT_YET_RUN = (
    "Results: **not yet run.** Run `python -m sttbench run ...` and then "
    "`python -m sttbench report --update-readme` to fill this table."
)


def percentile(values: list[float], pct: float) -> float:
    """Linear-interpolation percentile (same convention as numpy's default)."""
    if not values:
        raise ValueError("percentile of empty list")
    xs = sorted(values)
    pos = (len(xs) - 1) * pct / 100
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def summarize(results: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for name, p in results["providers"].items():
        if p["status"] != "ok":
            rows.append({"provider": name, "status": p["status"], "reason": p.get("reason", "")})
            continue
        ok = [c for c in p["clips"] if c["status"] == "ok"]
        row: dict[str, Any] = {
            "provider": name,
            "status": "ok",
            "model": p.get("model", "?"),
            "live": p.get("live", True),
            "clips": len(ok),
            "failures": len(p["clips"]) - len(ok),
            "wer": None,
            "p50": None,
            "p95": None,
        }
        if ok:
            row["wer"] = sum(c["edits"] for c in ok) / sum(c["ref_words"] for c in ok)
            lat = [c["latency_s"] for c in ok]
            row["p50"], row["p95"] = percentile(lat, 50), percentile(lat, 95)
        rows.append(row)
    return rows


def _fmt(value: float | None, fmt: str) -> str:
    return "n/a" if value is None else format(value, fmt)


def render_markdown(results: dict[str, Any]) -> str:
    meta = results.get("meta", {})
    rows = summarize(results)
    lines = [
        "| Provider | WER | Clips | Failures | p50 (s) | p95 (s) | Model |",
        "|---|---|---|---|---|---|---|",
    ]
    notes = []
    for r in rows:
        if r["status"] != "ok":
            lines.append(f"| {r['provider']} | n/a | 0 | n/a | n/a | n/a | {r['reason']} |")
            continue
        model = r["model"]
        if r["provider"] == "fake":
            model += " (FAKE: scripted output, not real ASR)"
        elif not r["live"]:
            model += " (untested against live API)"
        wer_cell = "n/a" if r["wer"] is None else f"{r['wer'] * 100:.2f}%"
        lines.append(
            f"| {r['provider']} | {wer_cell} | {r['clips']} | {r['failures']} | "
            f"{_fmt(r['p50'], '.2f')} | {_fmt(r['p95'], '.2f')} | {model} |"
        )
        if r["clips"]:
            notes.append(
                f"- {r['provider']}: latency percentiles over n={r['clips']} successful clips "
                "(linear interpolation; at this sample size p95 is close to the maximum)."
            )
    header = (
        f"Run {meta.get('run_date', '?')} (UTC) · dataset {meta.get('dataset', '?')} "
        f"({meta.get('n_clips', '?')} clips) · normaliser {meta.get('norm_version', '?')} · "
        f"harness {meta.get('git_sha', 'unknown')}"
    )
    return "\n".join([header, "", *lines, "", *notes]).rstrip() + "\n"


def update_readme(readme: str, results: dict[str, Any] | None) -> str:
    start, end = readme.find(README_START), readme.find(README_END)
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"README is missing the {README_START} / {README_END} markers")
    body = NOT_YET_RUN if results is None else render_markdown(results)
    return readme[:start] + f"{README_START}\n{body.strip()}\n" + readme[end:]
