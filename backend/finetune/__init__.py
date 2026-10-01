"""
Domain-adaptation fine-tuning for the Turkish NER model.

Continues training the baked BERTurk NER model on synthetically generated
Turkish contract sentences whose CAPITALISED COMMON/DOMAIN NOUNS (Kredi, Platin,
Sözleşme, …) are labelled O — directly targeting the over-tagging that real
contracts exposed (see eval/real/README.md).

Training data is generated here and is DISJOINT from the evaluation sets
(eval/gold synthetic corpus + eval/real contracts), which stay held-out.
"""
