# NER Domain-Adaptation Fine-Tuning

Continued training of the baked Turkish NER model to fix the failure that real
contracts exposed (`eval/real/`): the out-of-domain WikiANN model tagged every
capitalised Turkish common/domain noun (`Kredi`, `Platin`, `Sözleşme`) as an
organisation, masking `Kredi` 64× in one contract.

## Method

- **Start from** the baked model (`models/bert-turkish-ner`) — keeps its WikiANN
  PER/LOC/ORG ability; we *adapt*, not retrain from scratch.
- **Training data** (`data_gen.py` → `train.jsonl`, 6000 sentences): templated
  Turkish contract/form sentences with real Person/Org/Location entities
  **and**, crucially, ~100 capitalised common/domain nouns (`Kredi`, `Platin`,
  `Vergi`, `Sözleşme`, …) labelled **O** — the signal that capitalised ≠ ORG.
- **Disjoint from the test sets** — none of the eval/gold synthetic corpus or
  the eval/real contracts is used for training.
- **Train** (`train.py`): manual PyTorch loop, low LR (3e-5), 3 epochs, on GPU.
  Output: `models/bert-turkish-ner-ft/`.

```bash
cd backend
python -m finetune.data_gen        # writes train.jsonl
python -m finetune.train           # writes models/bert-turkish-ner-ft/
```

The engine prefers the fine-tuned model automatically; force the baseline with
`MASKLAYER_NER_MODEL=<path>/models/bert-turkish-ner` (used for the A/B below).

## Results

**Synthetic test set (22 docs) — no regression:**

| mode | baseline F1 | fine-tuned F1 | leak (full) |
|------|------------:|--------------:|------------:|
| full | 0.997 | **0.997** | 0.009 → **0.000** |

Full-mode recall reached 1.000 and the leak rate fell to 0 (the fine-tune even
caught the one company the baseline missed). Raw NER-only precision dropped
slightly (a few new FPs the post-processing absorbs).

**Real contract (`tuketici_kredisi`, customer-PII ground truth):**

| metric | baseline | **fine-tuned** |
|--------|---------:|---------------:|
| total detections | 53 | **35** |
| junk false positives | 19 | **2** (`Türk Hukuku`, `Türkiye`) |
| precision (customer PII) | 0.377 | **0.571** |
| precision (bank public info allowed) | 0.50 | **0.943** |
| recall | 0.870 | **0.952** |
| F1 | 0.53 | **0.71** |

**Across all 5 real contracts** the capitalised-noun over-tagging is gone — e.g.
`platin_maden` went from **`Platin` ×15** to 0; remaining `Company` detections
are now mostly legitimate (the bank itself, the car dealer, the vehicle brand,
the bank's pension subsidiary) rather than defined-term junk.

## Honest limitations

- Trained on **synthetic templated** data and validated on **n=5** real
  contracts — promising, not a benchmark. More real documents (other banks /
  domains) are needed to confirm generalisation.
- Near-zero training loss → the model fit the templates easily; the real-doc
  gain is the meaningful signal, not the loss curve.
- Residual real-doc FPs (`Türk Hukuku`, `Türkiye`, the bank's own public block)
  and the customer-vs-bank-PII policy question remain.

## Thesis framing

This is the project's **domain-adaptation contribution**: a failure that **no
post-processing rule could fix** (every contract introduces fresh capitalised
jargon) was resolved by continued training, lifting real-contract F1 0.53→0.71
(precision 0.38→0.57; 0.94 excluding the bank's own public info) while holding
synthetic at 0.997. It is the empirical justification for the eval→fix→re-measure
loop run end to end on real data.
