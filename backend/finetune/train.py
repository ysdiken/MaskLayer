"""
Continue-train the baked Turkish NER model on the domain-adaptation data.

Manual PyTorch loop (no accelerate/datasets dependency). Starts from the
existing trained NER weights (preserves WikiANN PER/LOC/ORG knowledge) and
adapts with a low LR so it learns "capitalised common noun → O" without
catastrophic forgetting.

Usage (from backend/):
    python -m finetune.data_gen      # if train.jsonl not present
    python -m finetune.train
Outputs: backend/models/bert-turkish-ner-ft/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = ROOT / "models" / "bert-turkish-ner"
OUT_DIR = ROOT / "models" / "bert-turkish-ner-ft"
DATA = Path(__file__).resolve().parent / "train.jsonl"

MAXLEN = 64
EPOCHS = 3
BATCH = 16
LR = 3e-5


def main() -> None:
    if not DATA.exists():
        raise SystemExit("train.jsonl missing — run: python -m finetune.data_gen")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(str(BASE_DIR))
    model = AutoModelForTokenClassification.from_pretrained(str(BASE_DIR)).to(device)
    label2id = model.config.label2id
    print(f"device={device}  labels={label2id}")

    rows = [json.loads(l) for l in DATA.open(encoding="utf-8")]

    def encode(row: dict) -> dict:
        enc = tok(row["tokens"], is_split_into_words=True,
                  truncation=True, max_length=MAXLEN, padding="max_length")
        word_ids = enc.word_ids()
        labels, prev = [], None
        for wid in word_ids:
            if wid is None:
                labels.append(-100)
            elif wid != prev:
                labels.append(label2id[row["tags"][wid]])
            else:
                labels.append(-100)   # only first sub-token of a word is supervised
            prev = wid
        enc["labels"] = labels
        return enc

    class DS(Dataset):
        def __init__(self, rows):
            self.data = [encode(r) for r in rows]
        def __len__(self):
            return len(self.data)
        def __getitem__(self, i):
            e = self.data[i]
            return {k: torch.tensor(e[k]) for k in ("input_ids", "attention_mask", "labels")}

    loader = DataLoader(DS(rows), batch_size=BATCH, shuffle=True)
    opt = AdamW(model.parameters(), lr=LR)

    model.train()
    for epoch in range(EPOCHS):
        total = 0.0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            out.loss.backward()
            opt.step()
            opt.zero_grad()
            total += out.loss.item()
        print(f"epoch {epoch + 1}/{EPOCHS}  mean_loss={total / len(loader):.4f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(OUT_DIR))
    tok.save_pretrained(str(OUT_DIR))
    print(f"saved → {OUT_DIR}")


if __name__ == "__main__":
    main()
