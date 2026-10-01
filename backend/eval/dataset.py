"""
Gold dataset loading via inline annotation markup.

Authoring PII spans by character offset is error-prone, so gold documents are
written as plain Turkish text with the PII wrapped in markup:

    TC Kimlik No [[10000000146|TC_No]] sahibi [[Ahmet Yılmaz|Person]] ...

`parse_markup()` strips the markup, returns the clean text, and computes the
exact character offsets of each annotated span — so authors never touch offsets.

Markup grammar:
    [[<surface text>|<Label>]]
  - <Label> is a taxonomy label (Person, TC_No, Company, …).
  - <surface text> may contain spaces and punctuation; the LAST '|' separates
    it from the label, so a '|' inside the surface text is tolerated.
  - Annotate ONLY true PII the system is expected to mask. Do NOT annotate
    statute names, public platforms, contract aliases, or document-type words —
    leaving them unannotated is how the gold set rewards precision filters.

Boundary conventions (match the masking pipeline so strict scoring is fair):
  - Money_Amount includes the currency symbol/code ("₺12.500,00", "1.000 TL").
  - Keyword-anchored numbers (Customer_No, SGK_No, Policy_No, Contract_No,
    Invoice_No, Reference_No) annotate ONLY the number, not the Turkish label.
  - Company includes the legal-form suffix if attached ("… A.Ş.").
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


# [[ surface | Label ]] — surface is non-greedy, label is a bare identifier.
_MARKUP = re.compile(r"\[\[(.+?)\|([A-Za-z_][A-Za-z0-9_]*)\]\]", re.DOTALL)

GOLD_DIR = Path(__file__).resolve().parent / "gold"


@dataclass(frozen=True)
class GoldSpan:
    start: int
    end: int
    label: str
    text: str


@dataclass
class GoldDoc:
    doc_id: str
    text: str               # clean text, markup removed
    spans: list[GoldSpan]


def parse_markup(raw: str) -> tuple[str, list[GoldSpan]]:
    """
    Convert markup-annotated text into (clean_text, gold_spans) with offsets
    that index into clean_text. Unannotated text is preserved verbatim.
    """
    out: list[str] = []
    spans: list[GoldSpan] = []
    pos = 0          # offset into the clean text being built
    last = 0         # offset into raw, end of last consumed region

    for m in _MARKUP.finditer(raw):
        before = raw[last:m.start()]
        out.append(before)
        pos += len(before)

        surface = m.group(1)
        label = m.group(2)
        start = pos
        out.append(surface)
        pos += len(surface)
        spans.append(GoldSpan(start=start, end=pos, label=label, text=surface))

        last = m.end()

    tail = raw[last:]
    out.append(tail)
    return "".join(out), spans


def load_doc(path: Path) -> GoldDoc:
    raw = path.read_text(encoding="utf-8")
    text, spans = parse_markup(raw)
    return GoldDoc(doc_id=path.stem, text=text, spans=spans)


def load_dataset(directory: Path | None = None) -> list[GoldDoc]:
    """Load every *.txt gold document in `directory` (default eval/gold), sorted by id."""
    directory = directory or GOLD_DIR
    docs = [load_doc(p) for p in sorted(directory.glob("*.txt"))]
    return docs
