# Detection Evaluation Harness

Measures the MaskLayer detection pipeline against a hand-annotated gold corpus
of synthetic Turkish documents. Produces per-label P/R/F1, the regex/NER/full
ablation, and a character-level **privacy leak rate**.

## Run

```bash
cd backend
python -m eval.run_eval                      # full mode (regex + NER)
python -m eval.run_eval --modes regex,ner,full   # full ablation table
python -m eval.run_eval --modes full --errors    # show every FP/FN per doc
python -m eval.run_eval --modes regex,ner,full --save   # write versioned JSON + history.csv
```

`--save` writes `results/eval_v<version>_<timestamp>.json` and appends a row to
`results/history.csv` (timestamp, engine_version, mode, micro_f1, P, R,
leak_rate, n_docs) — that file is the per-version error-rate curve for the thesis.

## Metrics

| Metric | Meaning |
|--------|---------|
| **Strict** P/R/F1 | predicted `(start, end, label)` must equal a gold span exactly. Headline detection metric; penalises boundary errors. |
| **Relaxed** P/R/F1 | same label + any character overlap (one-to-one). Credits near-misses. |
| **Char-level / leak rate** | of all gold PII characters, fraction left unmasked (`leak_rate = 1 − char_recall`). Label-agnostic — the KVKK-relevant privacy metric. |

## Adding / editing gold documents

Gold docs live in `gold/*.txt` as plain Turkish text with PII wrapped in markup:

```
TC Kimlik No [[10000000146|TC_No]] sahibi [[Ahmet Yılmaz|Person]] ...
```

`[[surface text|Label]]` — the loader strips the markup and computes offsets, so
you never touch character positions. Rules:

- **Annotate only true PII** the system should mask. Leave statute names
  ("6698 sayılı … Kanunu"), public platforms (Twitter), contract aliases
  ("Şirket"), and document-type words unannotated — that is how the corpus
  rewards the precision filters.
- **Boundary conventions** (so strict scoring is fair): Money_Amount includes the
  currency symbol; keyword-anchored numbers (Customer_No, SGK_No, Policy_No,
  Contract_No, Invoice_No, Reference_No) annotate only the number; Company
  includes an attached "A.Ş." suffix.
- Use valid checksums for *bare* TC/IBAN (e.g. TC `10000000146`, IBAN
  `TR330006100519786457841326`), or put them after a label
  (`TC Kimlik No: …`) so the keyword-anchored fallback catches them.

`tests/test_eval.py` asserts every gold span's offsets match its surface text,
so a malformed annotation fails CI immediately.

## Baseline (v0.5.0, 22 docs / 158 spans)

| mode | strict micro F1 | precision | recall | leak rate |
|------|----------------:|----------:|-------:|----------:|
| regex | 0.766 | 1.000 | 0.620 | 0.352 |
| ner   | 0.534 | 0.937 | 0.373 | 0.640 |
| **full** | **0.997** | **1.000** | 0.994 | **0.009** |

Regex + NER + gazetteer are complementary; combining them cuts the PII leak rate
from ~35% / ~64% to **0.9%** at **1.000 precision** (zero false positives over
158 spans). The v0.4.1→0.5.0 gain (full F1 0.975→0.997) came from the gazetteer
(company legal-form detector — fixes the `A.Ş.` boundary), public-institution
suppression (court names), and the Address inline-over-capture fix. Only
remaining miss: one `Mavi Ada Teknoloji` with no legal-form suffix (1/158) — a
genuine NER limitation.

> The corpus is synthetic — fictional names/numbers only. Expand toward 20+ docs
> and re-run `--save` after each engine change to grow the version curve.
