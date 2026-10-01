"""
Unit tests for ner_engine.py long-document chunking.

No ML model required — a stub tokenizer (one token per whitespace word, real
character offsets) and a stub pipeline (substring search for known entities)
replace the HuggingFace components, so tests run fast and offline.

Coverage:
  - _compute_chunks: short text → single full-span window.
  - _compute_chunks: core regions tile the document exactly (no gaps,
    no overlaps), windows respect max_tokens.
  - detect(): entity beyond the first 384-token window is found (the bug
    chunking fixes — previously truncated away).
  - detect(): entity inside the shared overlap region is emitted once.
  - detect(): offsets are global — text[start:end] == span.text.
  - _char_window_chunks fallback: same tiling guarantees.
  - detect() falls back to char windows when the tokenizer raises.
"""

import re

import pytest

from app.masking.ner_engine import (
    _CHUNK_MAX_TOKENS,
    _CHUNK_OVERLAP_TOKENS,
    NerEngine,
    _char_window_chunks,
    _compute_chunks,
)


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

class StubTokenizer:
    """Whitespace tokenizer: one token per word, real character offsets."""

    def __call__(self, text, **kwargs):
        return {
            "offset_mapping": [
                (m.start(), m.end()) for m in re.finditer(r"\S+", text)
            ]
        }


class RaisingTokenizer:
    """Simulates a slow (non-fast) tokenizer without offset support."""

    def __call__(self, text, **kwargs):
        raise NotImplementedError("offset mapping requires a fast tokenizer")


class StubNerPipeline:
    """
    Finds every occurrence of the configured names in the chunk it receives
    and emits HuggingFace-style entity dicts with chunk-relative offsets.
    """

    def __init__(self, names: dict[str, str], tokenizer=None):
        self.names = names  # surface form → entity_group (PER/LOC/ORG)
        self.tokenizer = tokenizer if tokenizer is not None else StubTokenizer()
        self.calls: list[str] = []

    def __call__(self, chunk: str):
        self.calls.append(chunk)
        ents = []
        for name, group in self.names.items():
            start = 0
            while True:
                idx = chunk.find(name, start)
                if idx == -1:
                    break
                ents.append({
                    "entity_group": group,
                    "score": 0.95,
                    "word": name,
                    "start": idx,
                    "end": idx + len(name),
                })
                start = idx + len(name)
        return ents


def make_engine(names: dict[str, str], tokenizer=None) -> NerEngine:
    eng = NerEngine()
    eng._pipeline = StubNerPipeline(names, tokenizer=tokenizer)
    eng._available = True
    eng._load_attempted = True
    return eng


def words(n: int, stem: str = "lorem") -> str:
    """n filler words that are neither entities nor blocklisted."""
    return " ".join(f"{stem}{i}" for i in range(n))


# ---------------------------------------------------------------------------
# _compute_chunks
# ---------------------------------------------------------------------------

class TestComputeChunks:

    def test_short_text_single_full_window(self):
        text = "Ahmet dün Ankara'ya gitti."
        chunks = _compute_chunks(text, StubTokenizer())
        assert chunks == [(0, len(text), 0, len(text))]

    def test_empty_text_single_window(self):
        assert _compute_chunks("", StubTokenizer()) == [(0, 0, 0, 0)]

    def test_long_text_produces_multiple_windows(self):
        text = words(_CHUNK_MAX_TOKENS + 100)
        chunks = _compute_chunks(text, StubTokenizer())
        assert len(chunks) >= 2

    def test_cores_tile_document_exactly(self):
        text = words(1000)
        chunks = _compute_chunks(text, StubTokenizer(), max_tokens=100, overlap_tokens=20)

        # First core starts at 0, last core ends at len(text)
        assert chunks[0][2] == 0
        assert chunks[-1][3] == len(text)
        # Each window's core ends exactly where the next window's core starts
        for (_, _, _, core_end), (_, _, next_core_start, _) in zip(chunks, chunks[1:]):
            assert core_end == next_core_start

    def test_windows_respect_max_tokens(self):
        text = words(1000)
        chunks = _compute_chunks(text, StubTokenizer(), max_tokens=100, overlap_tokens=20)
        for win_start, win_end, _, _ in chunks:
            n_tokens = len(re.findall(r"\S+", text[win_start:win_end]))
            assert n_tokens <= 100

    def test_core_lies_within_window(self):
        text = words(1000)
        chunks = _compute_chunks(text, StubTokenizer(), max_tokens=100, overlap_tokens=20)
        for win_start, win_end, core_start, core_end in chunks:
            assert win_start <= core_start < core_end <= win_end


# ---------------------------------------------------------------------------
# detect() with chunking
# ---------------------------------------------------------------------------

class TestDetectChunked:

    def test_entity_beyond_first_window_is_found(self):
        # Entity sits past the 384-token window — pre-chunking this was
        # silently truncated away.
        text = words(600) + " Ahmet Yılmaz " + words(50, stem="ipsum")
        eng = make_engine({"Ahmet Yılmaz": "PER"})

        spans = eng.detect(text)

        assert len(spans) == 1
        span = spans[0]
        assert span.label == "Person"
        assert span.text == "Ahmet Yılmaz"
        assert text[span.start: span.end] == "Ahmet Yılmaz"

    def test_entity_in_overlap_region_emitted_once(self):
        # Word index ~350 falls inside the overlap between window 1
        # (tokens 0–384) and window 2 (tokens 320–704): both windows see it,
        # exactly one must own it.
        overlap_word_idx = _CHUNK_MAX_TOKENS - _CHUNK_OVERLAP_TOKENS // 2
        text = (
            words(overlap_word_idx)
            + " Ahmet Yılmaz "
            + words(400, stem="ipsum")
        )
        eng = make_engine({"Ahmet Yılmaz": "PER"})

        spans = eng.detect(text)

        assert len(spans) == 1
        assert text[spans[0].start: spans[0].end] == "Ahmet Yılmaz"
        # Sanity: the doc was actually processed in 2+ windows
        assert len(eng._pipeline.calls) >= 2

    def test_entities_across_multiple_windows_all_found(self):
        text = (
            "Mehmet Demir " + words(700) + " Ayşe Kaya " + words(700, stem="ipsum")
        )
        eng = make_engine({"Mehmet Demir": "PER", "Ayşe Kaya": "PER"})

        spans = eng.detect(text)

        found = sorted(text[s.start: s.end] for s in spans)
        assert found == ["Ayşe Kaya", "Mehmet Demir"]

    def test_repeated_entity_found_at_every_occurrence(self):
        # Same name on "page 1" and "page 3" — each occurrence gets a span
        # with its own correct global offsets.
        text = (
            words(10) + " Mavi Ada Teknoloji " + words(700)
            + " Mavi Ada Teknoloji " + words(10, stem="ipsum")
        )
        eng = make_engine({"Mavi Ada Teknoloji": "ORG"})

        spans = eng.detect(text)

        assert len(spans) == 2
        assert spans[0].start != spans[1].start
        for s in spans:
            assert s.label == "Company"
            assert text[s.start: s.end] == "Mavi Ada Teknoloji"

    def test_short_text_still_works(self):
        text = "Sözleşmeyi Ahmet Yılmaz imzaladı."
        eng = make_engine({"Ahmet Yılmaz": "PER"})

        spans = eng.detect(text)

        assert len(spans) == 1
        assert text[spans[0].start: spans[0].end] == "Ahmet Yılmaz"
        # Single window → single pipeline call
        assert len(eng._pipeline.calls) == 1


# ---------------------------------------------------------------------------
# Character-window fallback
# ---------------------------------------------------------------------------

class TestCharWindowFallback:

    def test_short_text_single_full_window(self):
        text = "kısa metin"
        assert _char_window_chunks(text) == [(0, len(text), 0, len(text))]

    def test_cores_tile_document_exactly(self):
        text = "x" * 5000
        chunks = _char_window_chunks(text, max_chars=900, overlap_chars=180)

        assert chunks[0][2] == 0
        assert chunks[-1][3] == len(text)
        for (_, _, _, core_end), (_, _, next_core_start, _) in zip(chunks, chunks[1:]):
            assert core_end == next_core_start
        for win_start, win_end, core_start, core_end in chunks:
            assert win_end - win_start <= 900
            assert win_start <= core_start < core_end <= win_end

    def test_detect_falls_back_when_tokenizer_raises(self):
        # ~2000 chars of filler, entity near the end — past the first
        # 900-char fallback window.
        text = words(300) + " Ahmet Yılmaz " + words(20, stem="ipsum")
        assert len(text) > 1800
        eng = make_engine({"Ahmet Yılmaz": "PER"}, tokenizer=RaisingTokenizer())

        spans = eng.detect(text)

        assert len(spans) == 1
        assert text[spans[0].start: spans[0].end] == "Ahmet Yılmaz"
        assert len(eng._pipeline.calls) >= 2
