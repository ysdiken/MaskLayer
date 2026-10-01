"""
Scoring: gold spans vs. predicted spans.

Three complementary views, because "did we mask the PII?" and "did we mask it
cleanly?" are different questions:

  1. STRICT span match — predicted (start, end, label) must equal a gold span
     exactly. The headline detection metric; penalises boundary errors.

  2. RELAXED span match — same label + any character overlap, one-to-one.
     Credits the model for finding an entity even with imperfect boundaries.

  3. CHARACTER level (label-agnostic) — of all gold PII characters, how many
     were covered by any predicted span. `leak_rate = 1 - char_recall` is the
     privacy-relevant metric for KVKK: a character left uncovered is leaked PII,
     regardless of whether the label was right. Over-masking shows up as lower
     char precision, the safe failure mode.

All three aggregate across documents by summing counts, then computing
micro (pooled tp/fp/fn) and macro (mean of per-label F1) figures.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Protocol


class _SpanLike(Protocol):
    start: int
    end: int
    label: str


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


@dataclass
class LabelScore:
    label: str
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def support(self) -> int:
        return self.tp + self.fn

    @property
    def prf(self) -> tuple[float, float, float]:
        return _prf(self.tp, self.fp, self.fn)


@dataclass
class Report:
    mode: str = ""
    strict: dict[str, LabelScore] = field(default_factory=dict)
    relaxed: dict[str, LabelScore] = field(default_factory=dict)
    n_docs: int = 0
    # Character-level totals (label-agnostic)
    gold_chars: int = 0
    pred_chars: int = 0
    covered_chars: int = 0

    # ── Aggregates ────────────────────────────────────────────────────────────
    def micro(self, strict: bool = True) -> tuple[float, float, float]:
        table = self.strict if strict else self.relaxed
        tp = sum(s.tp for s in table.values())
        fp = sum(s.fp for s in table.values())
        fn = sum(s.fn for s in table.values())
        return _prf(tp, fp, fn)

    def macro(self, strict: bool = True) -> tuple[float, float, float]:
        table = self.strict if strict else self.relaxed
        if not table:
            return 0.0, 0.0, 0.0
        ps, rs, fs = zip(*(s.prf for s in table.values()))
        n = len(table)
        return sum(ps) / n, sum(rs) / n, sum(fs) / n

    @property
    def char_recall(self) -> float:
        return self.covered_chars / self.gold_chars if self.gold_chars else 1.0

    @property
    def char_precision(self) -> float:
        return self.covered_chars / self.pred_chars if self.pred_chars else 1.0

    @property
    def leak_rate(self) -> float:
        return 1.0 - self.char_recall


def _accumulate(table: dict[str, LabelScore], label: str, *, tp=0, fp=0, fn=0) -> None:
    s = table.setdefault(label, LabelScore(label))
    s.tp += tp
    s.fp += fp
    s.fn += fn


def _char_set(spans: Iterable[_SpanLike]) -> set[int]:
    chars: set[int] = set()
    for s in spans:
        chars.update(range(s.start, s.end))
    return chars


def score_doc(report: Report, gold: list[_SpanLike], pred: list[_SpanLike]) -> None:
    """Score one document into `report` (mutates the running totals)."""
    report.n_docs += 1

    # ── Strict: exact (start, end, label) ─────────────────────────────────────
    gold_keys = {(g.start, g.end, g.label) for g in gold}
    pred_keys = {(p.start, p.end, p.label) for p in pred}
    for key in pred_keys:
        label = key[2]
        if key in gold_keys:
            _accumulate(report.strict, label, tp=1)
        else:
            _accumulate(report.strict, label, fp=1)
    for key in gold_keys:
        if key not in pred_keys:
            _accumulate(report.strict, key[2], fn=1)

    # ── Relaxed: same label + any overlap, greedy one-to-one ──────────────────
    matched_pred: set[int] = set()
    for g in gold:
        hit = None
        for i, p in enumerate(pred):
            if i in matched_pred:
                continue
            if p.label == g.label and p.start < g.end and g.start < p.end:
                hit = i
                break
        if hit is not None:
            matched_pred.add(hit)
            _accumulate(report.relaxed, g.label, tp=1)
        else:
            _accumulate(report.relaxed, g.label, fn=1)
    for i, p in enumerate(pred):
        if i not in matched_pred:
            _accumulate(report.relaxed, p.label, fp=1)

    # ── Character level (label-agnostic) ──────────────────────────────────────
    gchars = _char_set(gold)
    pchars = _char_set(pred)
    report.gold_chars += len(gchars)
    report.pred_chars += len(pchars)
    report.covered_chars += len(gchars & pchars)


def format_report(report: Report) -> str:
    """Human-readable table: per-label strict P/R/F1, aggregates, leak rate."""
    lines: list[str] = []
    title = f"  EVAL — mode={report.mode or '?'}  ({report.n_docs} docs)"
    lines.append("=" * 64)
    lines.append(title)
    lines.append("=" * 64)
    lines.append(f"{'Label':<16}{'P':>8}{'R':>8}{'F1':>8}{'TP':>6}{'FP':>6}{'FN':>6}")
    lines.append("-" * 64)
    for label in sorted(report.strict, key=lambda l: report.strict[l].support, reverse=True):
        s = report.strict[label]
        p, r, f = s.prf
        lines.append(
            f"{label:<16}{p:>8.3f}{r:>8.3f}{f:>8.3f}{s.tp:>6}{s.fp:>6}{s.fn:>6}"
        )
    lines.append("-" * 64)
    mp, mr, mf = report.micro(strict=True)
    Mp, Mr, Mf = report.macro(strict=True)
    rmp, rmr, rmf = report.micro(strict=False)
    lines.append(f"{'micro (strict)':<16}{mp:>8.3f}{mr:>8.3f}{mf:>8.3f}")
    lines.append(f"{'macro (strict)':<16}{Mp:>8.3f}{Mr:>8.3f}{Mf:>8.3f}")
    lines.append(f"{'micro (relaxed)':<16}{rmp:>8.3f}{rmr:>8.3f}{rmf:>8.3f}")
    lines.append("-" * 64)
    lines.append(
        f"char-level: recall={report.char_recall:.3f}  "
        f"precision={report.char_precision:.3f}  "
        f"LEAK RATE={report.leak_rate:.3f}  "
        f"({report.covered_chars}/{report.gold_chars} PII chars masked)"
    )
    lines.append("=" * 64)
    return "\n".join(lines)


def report_to_dict(report: Report) -> dict:
    """Serialisable summary for versioned result files / the error-rate curve."""
    mp, mr, mf = report.micro(strict=True)
    Mp, Mr, Mf = report.macro(strict=True)
    return {
        "mode": report.mode,
        "n_docs": report.n_docs,
        "micro_strict": {"precision": mp, "recall": mr, "f1": mf},
        "macro_strict": {"precision": Mp, "recall": Mr, "f1": Mf},
        "micro_relaxed": dict(zip(("precision", "recall", "f1"), report.micro(strict=False))),
        "char": {
            "recall": report.char_recall,
            "precision": report.char_precision,
            "leak_rate": report.leak_rate,
            "gold_chars": report.gold_chars,
            "covered_chars": report.covered_chars,
        },
        "per_label": {
            label: {
                "precision": s.prf[0], "recall": s.prf[1], "f1": s.prf[2],
                "tp": s.tp, "fp": s.fp, "fn": s.fn, "support": s.support,
            }
            for label, s in sorted(report.strict.items())
        },
    }
