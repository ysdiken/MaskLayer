# Real-Document Case Study — the synthetic-vs-real gap

> The single most important evaluation finding. **Synthetic-corpus F1 (0.997)
> drastically overstates real-world performance.** On a real Turkish bank
> contract, precision collapsed to 0.15 — a ~6× drop — for a reason synthetic
> data structurally cannot reproduce.

## The document

`tuketici_kredisi_filled.txt` — a real 9-page vehicle-loan contract from a large Turkish bank
(`Tüketici Kredisi Sözleşmesi`), extracted with the project's own PDF pipeline,
with the customer fields filled with fictional but realistic data (valid TC/IBAN
checksums). The **legal language, layout, clauses, bank header, and signatures
are real** — only the customer values are synthetic. `tuketici_kredisi_masked.txt`
is the pipeline output.

> **Data availability:** the contract texts (`*_filled.txt`, `*_masked*.txt`)
> are **not distributed** in this repository — the templates are copyrighted by
> the issuing bank. Only the analysis and numbers are published here. To
> reproduce, extract a publicly available bank contract template with the
> project's PDF pipeline and fill the customer fields with fictional data.
>
> **Anonymisation note:** in the local copies, the names of the bank officers
> printed as signatories were replaced with fictional names of the same shape
> (`Given SURNAME [SURNAME]`) after the masking runs. The masked outputs were
> edited with the same substitution rather than regenerated, so the numbers
> below reflect the original runs.

This is `n=1` — a qualitative case study, not a corpus-level F1. But one real
contract was enough to expose a failure mode 22 synthetic documents missed.

## What happened (v0.5.0, before fixes)

| metric | synthetic (22 docs) | **real contract** |
|--------|--------------------:|------------------:|
| precision (customer PII) | 1.000 | **0.150** |
| recall | 0.994 | 0.870 |
| F1 | 0.997 | **~0.26** |
| spans detected | — | 133 (**94 garbage**) |

The NER model tagged the word **`Kredi`** ("loan") as a company **64 times**,
plus tax abbreviations (`KKDF`, `BSMV`), capitalised legal jargon (`Olur`,
`Akdi`, `Temerrüt`), fee names, and 6 public institutions. The masked contract
was unreadable: `Kredi Hesap Numarası` → `{Company_1} Hesap Numarası`.

**Root cause:** real Turkish contracts define and then capitalise key terms
throughout — `Taşıt Kredisi'nin ("Kredi")`, `("Biz")`, `("Siz")`,
`("Sözleşme")`. The out-of-domain WikiANN NER model reads any capitalised token
as a proper noun → ORG. My synthetic docs never repeated a capitalised defined
term 60×, so the synthetic F1 never saw this.

## Fixes (v0.5.1 — generalizable, not doc-specific)

1. **Bare parenthetical alias extraction** — `("X")` defined terms are now
   recognised and suppressed (killed all 64 `Kredi` FPs).
2. **Broadened institution suppression** — case-suffix tolerant
   (`mahkemesine`), + `Birliği` / `Sistemi` / hakem heyeti / tüketici mahkemesi.
3. **Finance-jargon blocklist** — KKDF, BSMV, ÖTV, Olur, Akdi, Temerrüt, …

| | spans | precision | F1 |
|--|------:|----------:|---:|
| v0.5.0 (before) | 133 | 0.150 | 0.26 |
| v0.5.1 (after)  | 53  | **0.377** (0.50 excl. bank-public info) | **0.53** |

Synthetic F1 stayed **0.997** (no regression) — the fixes only removed real-doc
noise.

## What remains (honest limitations)

Residual FPs are **inherent to the out-of-domain NER model** and resist
post-processing: product/contract names that *contain* a defined term
(`Taşıt Kredisi`, `Bağlı Kredi`), fee names (`… Ücreti`), and stray capitalised
nouns. Recall is also imperfect — one bank executive (a three-part name, `Given SURNAME SURNAME`) was
caught only partially. These are the case **for domain-adaptation fine-tuning**
(deferred), not more hand-written rules.

## Reproduce

```bash
cd backend
python -X utf8 - <<'PY'
import asyncio; from app.masking.pipeline import pipeline
t = open("eval/real/tuketici_kredisi_filled.txt", encoding="utf-8").read()
r = asyncio.run(pipeline.run(t, mode="full"))
open("eval/real/tuketici_kredisi_masked.txt","w",encoding="utf-8").write(r.masked_text)
print(len(r.spans), "spans")
PY
```

## Update — 5-document real corpus

Four more real templates from the same bank were filled + masked (kept locally,
see *Data availability* above):
`virman` (transfer instruction, 1pg), `platin_maden` (precious-metal deposit,
3pg), `kredi_bilgi_formu` (credit info form, 9pg), `kredi_sozlesme_sube` (branch
credit contract, 9pg).

| doc | spans | Company FPs | dominant junk |
|-----|------:|------------:|---------------|
| virman | 14 | ~1 | (clean — short doc) |
| platin_maden | 53 | ~27 | **`Platin` ×15** |
| kredi_bilgi_formu | 51 | ~29 | `Tüketici Kredisi`, `Bağlı Kredi`, fee/rate names |
| kredi_sozlesme_sube | 56 | ~30 | same product/fee names |

Two findings:
1. **The `("X")` fix generalized** — `Kredi`/`Biz`/`Siz`/`Sözleşme`/`Hesap` are
   suppressed across *all* credit docs (held up on unseen documents).
2. **The root problem recurs for any capitalized domain term not parenthetically
   defined** — `platin_maden` tags **`Platin` 15×** as a company, exactly like
   `Kredi` before. Product/fee names (`Tüketici Kredisi`, `Bağlı Kredi`,
   `Rehin Tesis Ücreti`) persist everywhere.

**Conclusion:** post-processing rules fix *specific* terms (parenthetical-defined,
known abbreviations) but cannot fix the *general* failure — the WikiANN NER model
tags capitalised Turkish common/domain nouns as organisations, and every new
contract domain introduces fresh such terms. Hand-rules have hit diminishing
returns. **This is the definitive case for domain-adaptation fine-tuning.** A
possible generalizable rule still worth trying first: glossary-aware suppression
(these contracts carry a `TERİMLER LİSTESİ` that defines exactly the over-tagged
jargon — Akdi, Temerrüt, Muaccel, BSMV, KKDF), but it won't cover product names
or `Platin`.

## Thesis takeaway

Report **two** numbers and never let the synthetic one stand alone:
- synthetic F1 0.997 = an *upper bound / regression sanity check*;
- real-contract precision 0.15→0.38 = the *honest* operating point and the
  motivation for fine-tuning.

Next: collect **more real contracts** (different banks/types) for a real
held-out test set — `n=1` is a case study, not a benchmark.
