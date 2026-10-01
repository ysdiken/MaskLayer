"""
Mask policy — a global, post-detection output filter.

WHAT THIS IS
    A per-deployment configuration of which entity *types* are actually
    rendered as placeholders in the masked output. Different organisations
    have different requirements: a law firm may not need to mask dates, a
    bank may not need to mask company names that are already public.

WHAT THIS IS NOT
    This does NOT change detection, the model, the corrections log, or the
    training-examples store. The pipeline always detects every entity it can,
    every span is still returned in MaskResponse.spans, and every analyst
    correction is still logged against the full detection. The policy only
    decides which detected spans get substituted in the *output text* and
    the Redis mapping. Flipping a label off later, or on again, has zero
    effect on what the model learned or what we recorded — it is purely a
    presentation filter applied after masking.

FAIL-SAFE DESIGN
    The policy is stored as an explicit label→bool map, but resolution is
    fail-safe: a label that is missing from the policy (e.g. a newly added
    entity type, or a custom manual-span label) is treated as ENABLED and
    therefore masked. The only way an entity is left unmasked is an explicit
    `false` for its label. Over-masking is a privacy-safe failure mode; a
    silent leak is not.

PERSISTENCE
    A small JSON file (MASK_POLICY_PATH env var, default backend/config/
    mask_policy.json), cached in memory. Deliberately file-based, not DB-based:
    masking is designed to keep working when PostgreSQL is down (see
    app/main.py), so the output filter must not depend on the DB either.
    When the file is absent or unreadable, the safe default — mask everything
    — applies.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Label catalog — every maskable entity type, with its Turkish display name.
# Mirrors the frontend TAXONOMY_LABELS, plus IP_Address / URL which the regex
# engine can emit but the manual-span picker does not offer.
# `group` matches the detector that produces the type ("ner" | "regex").
# ---------------------------------------------------------------------------

MASK_LABELS: list[dict[str, str]] = [
    # NER (semantic) entities
    {"label": "Person",        "display": "Kişi",        "group": "ner"},
    {"label": "Company",       "display": "Şirket",      "group": "ner"},
    {"label": "Location",      "display": "Konum",       "group": "ner"},
    {"label": "Title",         "display": "Ünvan",       "group": "ner"},
    {"label": "Facility",      "display": "Tesis",       "group": "ner"},
    # Regex (deterministic) entities
    {"label": "TC_No",         "display": "TC Kimlik",   "group": "regex"},
    {"label": "IBAN",          "display": "IBAN",        "group": "regex"},
    {"label": "Tax_No",        "display": "Vergi No",    "group": "regex"},
    {"label": "Card_No",       "display": "Kart No",     "group": "regex"},
    {"label": "Phone_No",      "display": "Telefon",     "group": "regex"},
    {"label": "Email",         "display": "E-posta",     "group": "regex"},
    {"label": "Date",          "display": "Tarih",       "group": "regex"},
    {"label": "Money_Amount",  "display": "Para",        "group": "regex"},
    {"label": "Account_No",    "display": "Hesap No",    "group": "regex"},
    {"label": "License_Plate", "display": "Plaka",       "group": "regex"},
    {"label": "Passport_No",   "display": "Pasaport",    "group": "regex"},
    {"label": "Case_No",       "display": "Dava No",     "group": "regex"},
    {"label": "Address",       "display": "Adres",       "group": "regex"},
    {"label": "Customer_No",   "display": "Müşteri No",  "group": "regex"},
    {"label": "Contract_No",   "display": "Sözleşme No", "group": "regex"},
    {"label": "Policy_No",     "display": "Poliçe No",   "group": "regex"},
    {"label": "Invoice_No",    "display": "Fatura No",   "group": "regex"},
    {"label": "SGK_No",        "display": "SGK No",      "group": "regex"},
    {"label": "Reference_No",  "display": "Referans No", "group": "regex"},
    {"label": "IP_Address",    "display": "IP Adresi",   "group": "regex"},
    {"label": "URL",           "display": "URL",         "group": "regex"},
]

_ALL_LABELS: tuple[str, ...] = tuple(item["label"] for item in MASK_LABELS)


def _policy_path() -> Path:
    env = os.getenv("MASK_POLICY_PATH")
    if env:
        return Path(env)
    # backend/app/masking/mask_policy.py → parents[2] == backend/
    return Path(__file__).resolve().parents[2] / "config" / "mask_policy.json"


# ---------------------------------------------------------------------------
# In-memory cache (loaded lazily, refreshed on write)
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_cache: dict[str, bool] | None = None


def _default_policy() -> dict[str, bool]:
    """Every known label enabled — the privacy-safe default (mask everything)."""
    return {label: True for label in _ALL_LABELS}


def _read_file() -> dict[str, bool]:
    """Load the policy file, overlaying it on the default. Missing/invalid → default."""
    base = _default_policy()
    path = _policy_path()
    try:
        if not path.exists():
            return base
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            logger.warning("mask_policy: %s is not a JSON object — using defaults.", path)
            return base
        for label, value in raw.items():
            # Only honour booleans; ignore anything else defensively.
            if isinstance(value, bool):
                base[label] = value
    except Exception as e:
        logger.warning("mask_policy: failed to read %s (%s) — using defaults.", path, e)
    return base


def _write_file(policy: dict[str, bool]) -> None:
    path = _policy_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(policy, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_policy() -> dict[str, bool]:
    """Return the full label→enabled map (cached). Always includes every catalog label."""
    global _cache
    with _lock:
        if _cache is None:
            _cache = _read_file()
        return dict(_cache)


def get_disabled_labels() -> frozenset[str]:
    """
    Labels explicitly turned OFF. The pipeline masks every span whose label is
    NOT in this set — so unknown/custom labels are always masked (fail-safe).
    """
    return frozenset(label for label, enabled in get_policy().items() if not enabled)


def update_policy(updates: dict[str, bool]) -> dict[str, bool]:
    """
    Merge boolean `updates` into the policy, persist, and return the new policy.
    Unknown labels in `updates` are ignored (keeps the file aligned to the catalog).
    """
    global _cache
    with _lock:
        current = _cache if _cache is not None else _read_file()
        merged = dict(current)
        for label, value in updates.items():
            if label in merged and isinstance(value, bool):
                merged[label] = value
        _write_file(merged)
        _cache = merged
        return dict(merged)


def reset_cache() -> None:
    """Drop the in-memory cache (forces a re-read on next access). For tests."""
    global _cache
    with _lock:
        _cache = None
