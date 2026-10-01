"""
One-time script: download the Turkish NER model from HuggingFace and save
it to backend/models/bert-turkish-ner/ so the backend never needs internet.

Run once (with internet):
    cd backend
    python scripts/bake_model.py

After this, set TRANSFORMERS_OFFLINE=1 in .env and the model loads from disk.
The models/ directory should be committed to git-lfs or kept on the server —
it is intentionally excluded from regular git via .gitignore.
"""

from pathlib import Path
from transformers import AutoModelForTokenClassification, AutoTokenizer

MODEL_NAME = "savasy/bert-base-turkish-ner-cased"
SAVE_PATH  = Path(__file__).parent.parent / "models" / "bert-turkish-ner"

print(f"Downloading {MODEL_NAME} …")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)

print(f"Saving to {SAVE_PATH} …")
SAVE_PATH.mkdir(parents=True, exist_ok=True)
tokenizer.save_pretrained(str(SAVE_PATH))
model.save_pretrained(str(SAVE_PATH))

print("Done. Model is now fully local.")
print(f"Size: {sum(f.stat().st_size for f in SAVE_PATH.rglob('*') if f.is_file()) / 1e6:.1f} MB")
