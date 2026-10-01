"""
Evaluation runner — score the detection pipeline against the gold corpus.

Usage (from backend/):
    python -m eval.run_eval                     # full mode
    python -m eval.run_eval --modes regex,ner,full
    python -m eval.run_eval --errors            # also print per-doc mismatches
    python -m eval.run_eval --save              # write a versioned JSON result

What it measures: DETECTION quality (result.spans vs gold). The output mask
policy is irrelevant here — we score what the pipeline detects, which is exactly
the quantity the policy is designed NOT to change.

Each run is stamped with ENGINE_VERSION; saving accumulates a history row so the
per-version error-rate curve (the thesis's before/after evidence) can be plotted.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Turkish text + table glyphs need UTF-8; the default Windows console is cp1252.
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except Exception:
    pass

from app.masking.pipeline import pipeline, ENGINE_VERSION
from eval.dataset import GoldDoc, load_dataset
from eval.scorer import Report, score_doc, format_report, report_to_dict

RESULTS_DIR = Path(__file__).resolve().parent / "results"


async def evaluate(docs: list[GoldDoc], mode: str, show_errors: bool) -> Report:
    report = Report(mode=mode)
    for doc in docs:
        result = await pipeline.run(doc.text, mode=mode)  # type: ignore[arg-type]
        pred = result.spans
        score_doc(report, doc.spans, pred)
        if show_errors:
            _print_doc_errors(doc, pred)
    return report


def _print_doc_errors(doc: GoldDoc, pred) -> None:
    gold_keys = {(g.start, g.end, g.label) for g in doc.spans}
    pred_keys = {(p.start, p.end, p.label) for p in pred}
    fps = [p for p in pred if (p.start, p.end, p.label) not in gold_keys]
    fns = [g for g in doc.spans if (g.start, g.end, g.label) not in pred_keys]
    if not fps and not fns:
        return
    print(f"\n  ── {doc.doc_id} ──")
    for g in fns:
        print(f"    FN  {g.label:<14} {g.text!r}")
    for p in fps:
        print(f"    FP  {p.label:<14} {p.text!r}  ({p.source})")


def _save(reports: dict[str, Report]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()
    payload = {
        "engine_version": ENGINE_VERSION,
        "timestamp": ts,
        "modes": {mode: report_to_dict(r) for mode, r in reports.items()},
    }
    out = RESULTS_DIR / f"eval_v{ENGINE_VERSION}_{ts.replace(':', '-')}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # Append a flat history row per mode for the version curve.
    history = RESULTS_DIR / "history.csv"
    new = not history.exists()
    with history.open("a", encoding="utf-8") as fh:
        if new:
            fh.write("timestamp,engine_version,mode,micro_f1,micro_precision,"
                     "micro_recall,leak_rate,n_docs\n")
        for mode, r in reports.items():
            p, rc, f = r.micro(strict=True)
            fh.write(f"{ts},{ENGINE_VERSION},{mode},{f:.4f},{p:.4f},{rc:.4f},"
                     f"{r.leak_rate:.4f},{r.n_docs}\n")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="MaskLayer detection eval")
    ap.add_argument("--modes", default="full",
                    help="comma-separated: regex,ner,full (default: full)")
    ap.add_argument("--errors", action="store_true", help="print per-doc FP/FN")
    ap.add_argument("--save", action="store_true", help="write versioned result JSON + history row")
    args = ap.parse_args()

    docs = load_dataset()
    if not docs:
        raise SystemExit("No gold documents found in eval/gold/")

    gold_spans = sum(len(d.spans) for d in docs)
    print(f"Loaded {len(docs)} gold docs, {gold_spans} annotated spans. "
          f"Engine v{ENGINE_VERSION}.")

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    reports: dict[str, Report] = {}
    for mode in modes:
        report = asyncio.run(evaluate(docs, mode, args.errors))
        reports[mode] = report
        print()
        print(format_report(report))

    if args.save:
        out = _save(reports)
        print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
