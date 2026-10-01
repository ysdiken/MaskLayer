"""
Tests for the post-detection output filter (mask_policy + pipeline integration).

The whole point of this feature: disabling an entity type changes ONLY the
masked output, never the detection. These tests pin that contract:
  - result.spans always contains the full detection (training/audit safe)
  - disabled labels keep their original text in masked_text + mapping
  - manual (user-confirmed) spans are always masked, policy notwithstanding
  - the policy persists to disk and resolves fail-safe (unknown → masked)

NER is never loaded here — every test uses mode="regex" with deterministic
entities, so the suite stays fast and offline.
"""

import pytest

from app.masking import mask_policy
from app.masking.pipeline import pipeline
from app.models.schemas import ManualSpanRequest
from app.api.admin_routes import (
    get_mask_policy,
    update_mask_policy,
    MaskPolicyUpdate,
)


# ---------------------------------------------------------------------------
# Isolate the policy file per test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def isolated_policy(tmp_path, monkeypatch):
    monkeypatch.setenv("MASK_POLICY_PATH", str(tmp_path / "mask_policy.json"))
    mask_policy.reset_cache()
    yield
    mask_policy.reset_cache()


SAMPLE = (
    "TC Kimlik No 10000000146 sahibi, IBAN TR330006100519786457841326, "
    "15 Ocak 2024 tarihinde 1.234,56 TL ödedi."
)


# ---------------------------------------------------------------------------
# Policy module
# ---------------------------------------------------------------------------

class TestPolicyModule:

    def test_default_all_enabled(self):
        policy = mask_policy.get_policy()
        assert all(policy.values())
        assert set(policy) == {item["label"] for item in mask_policy.MASK_LABELS}

    def test_default_no_disabled(self):
        assert mask_policy.get_disabled_labels() == frozenset()

    def test_update_persists_and_disables(self):
        mask_policy.update_policy({"Date": False})
        assert mask_policy.get_disabled_labels() == frozenset({"Date"})
        # Survives a cache drop (i.e. it was written to disk)
        mask_policy.reset_cache()
        assert mask_policy.get_disabled_labels() == frozenset({"Date"})

    def test_update_is_partial(self):
        mask_policy.update_policy({"Date": False})
        mask_policy.update_policy({"Money_Amount": False})
        assert mask_policy.get_disabled_labels() == frozenset({"Date", "Money_Amount"})

    def test_re_enable(self):
        mask_policy.update_policy({"Date": False})
        mask_policy.update_policy({"Date": True})
        assert mask_policy.get_disabled_labels() == frozenset()

    def test_unknown_label_ignored_on_write(self):
        mask_policy.update_policy({"NotARealLabel": False})
        assert "NotARealLabel" not in mask_policy.get_policy()
        assert mask_policy.get_disabled_labels() == frozenset()

    def test_corrupt_file_falls_back_to_default(self, tmp_path, monkeypatch):
        path = tmp_path / "broken.json"
        path.write_text("{ not json", encoding="utf-8")
        monkeypatch.setenv("MASK_POLICY_PATH", str(path))
        mask_policy.reset_cache()
        # Fail-safe: unreadable policy → mask everything
        assert mask_policy.get_disabled_labels() == frozenset()


# ---------------------------------------------------------------------------
# Pipeline output filter
# ---------------------------------------------------------------------------

class TestPipelineFilter:

    @pytest.mark.asyncio
    async def test_no_policy_masks_everything(self):
        res = await pipeline.run(SAMPLE, mode="regex")
        assert "{TC_No_1}" in res.masked_text
        assert "{Date_1}" in res.masked_text
        assert "{Money_Amount_1}" in res.masked_text

    @pytest.mark.asyncio
    async def test_disabled_label_not_masked(self):
        res = await pipeline.run(
            SAMPLE, mode="regex", disabled_labels=frozenset({"Date"})
        )
        # Date stays as original text; no placeholder, no mapping entry
        assert "{Date_1}" not in res.masked_text
        assert "15 Ocak 2024" in res.masked_text
        assert "Date" not in {ph.split("_")[0].strip("{") for ph in res.mapping}
        # Other entities still masked
        assert "{TC_No_1}" in res.masked_text
        assert "{IBAN_1}" in res.masked_text

    @pytest.mark.asyncio
    async def test_spans_always_complete(self):
        """Detection is unaffected — the disabled entity is still in result.spans."""
        res = await pipeline.run(
            SAMPLE, mode="regex", disabled_labels=frozenset({"Date"})
        )
        labels = {s.label for s in res.spans}
        assert "Date" in labels   # detected, just not rendered

    @pytest.mark.asyncio
    async def test_mapping_excludes_disabled(self):
        res = await pipeline.run(
            SAMPLE, mode="regex", disabled_labels=frozenset({"Money_Amount"})
        )
        assert all("Money_Amount" not in ph for ph in res.mapping)
        assert "1.234,56 TL" in res.masked_text

    @pytest.mark.asyncio
    async def test_multiple_disabled(self):
        res = await pipeline.run(
            SAMPLE, mode="regex",
            disabled_labels=frozenset({"Date", "Money_Amount"}),
        )
        assert "15 Ocak 2024" in res.masked_text
        assert "1.234,56 TL" in res.masked_text
        assert "{TC_No_1}" in res.masked_text

    @pytest.mark.asyncio
    async def test_manual_span_always_masked(self):
        """User-confirmed spans bypass the policy — explicit intent wins."""
        text = "Toplantı 15 Ocak 2024 tarihinde yapıldı."
        manual = [ManualSpanRequest(text="15 Ocak 2024", label="Date")]
        res = await pipeline.run(
            text, mode="regex", manual_spans=manual,
            disabled_labels=frozenset({"Date"}),
        )
        # Even though Date is disabled globally, the manual span is masked
        assert "{Date_1}" in res.masked_text
        assert "15 Ocak 2024" not in res.masked_text


# ---------------------------------------------------------------------------
# Admin endpoints (called directly — no server / lifespan needed)
# ---------------------------------------------------------------------------

class TestAdminEndpoints:

    @pytest.mark.asyncio
    async def test_get_returns_catalog_and_policy(self):
        resp = await get_mask_policy()
        assert len(resp.labels) == len(mask_policy.MASK_LABELS)
        assert resp.policy["Date"] is True
        # Catalog entries carry display + group for the UI
        person = next(l for l in resp.labels if l["label"] == "Person")
        assert person["display"] == "Kişi"
        assert person["group"] == "ner"

    @pytest.mark.asyncio
    async def test_put_updates_policy(self):
        resp = await update_mask_policy(MaskPolicyUpdate(policy={"Location": False}))
        assert resp.policy["Location"] is False
        assert mask_policy.get_disabled_labels() == frozenset({"Location"})

    @pytest.mark.asyncio
    async def test_put_then_get_roundtrip(self):
        await update_mask_policy(MaskPolicyUpdate(policy={"Company": False}))
        resp = await get_mask_policy()
        assert resp.policy["Company"] is False
