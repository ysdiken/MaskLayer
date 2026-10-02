"""
Unit tests for MappingStore.

Uses fakeredis.aioredis so no real Redis container is required.
All tests are async — pytest-asyncio handles the event loop.

Coverage:
  - save_mapping / get_all / get_original round-trip
  - TTL is set on save
  - unmask_text: full substitution, partial (unknown placeholders), empty mapping
  - delete removes the key
  - Regex engine → _apply_masks → MappingStore end-to-end
"""

import pytest
import pytest_asyncio
import fakeredis.aioredis

from app.masking.mapping_store import MappingStore, _job_key
from app.masking.pipeline import _apply_masks
from app.masking.regex_engine import engine as regex_engine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def redis():
    """In-memory Redis backed by fakeredis — no real server needed."""
    r = fakeredis.aioredis.FakeRedis()
    yield r
    await r.aclose()


@pytest_asyncio.fixture
async def store(redis):
    return MappingStore(redis, ttl_seconds=3600)


# ---------------------------------------------------------------------------
# save_mapping / get_all / get_original
# ---------------------------------------------------------------------------

class TestSaveAndRead:
    async def test_save_and_get_all(self, store):
        mapping = {"{TC_No_1}": "10000000146", "{Email_1}": "ali@example.com"}
        await store.save_mapping("job-1", mapping)
        result = await store.get_all("job-1")
        assert result == mapping

    async def test_get_original_single_key(self, store):
        await store.save_mapping("job-2", {"{IBAN_1}": "TR330006100519786457841326"})
        val = await store.get_original("job-2", "{IBAN_1}")
        assert val == "TR330006100519786457841326"

    async def test_get_original_missing_key_returns_none(self, store):
        await store.save_mapping("job-3", {"{TC_No_1}": "10000000146"})
        val = await store.get_original("job-3", "{IBAN_1}")  # not in mapping
        assert val is None

    async def test_get_all_missing_job_returns_empty_dict(self, store):
        result = await store.get_all("nonexistent-job")
        assert result == {}

    async def test_save_empty_mapping_is_noop(self, store):
        await store.save_mapping("job-4", {})           # should not raise
        result = await store.get_all("job-4")
        assert result == {}

    async def test_ttl_is_set_after_save(self, store, redis):
        await store.save_mapping("job-5", {"{Phone_No_1}": "05321234567"})
        ttl = await redis.ttl(_job_key("job-5"))
        # TTL should be close to 3600 (fixture value); definitely > 0
        assert ttl > 0
        assert ttl <= 3600

    async def test_multiple_placeholders_all_stored(self, store):
        mapping = {
            "{TC_No_1}": "10000000146",
            "{IBAN_1}": "TR330006100519786457841326",
            "{Email_1}": "ali@example.com",
            "{Phone_No_1}": "05321234567",
        }
        await store.save_mapping("job-6", mapping)
        result = await store.get_all("job-6")
        assert result == mapping


# ---------------------------------------------------------------------------
# unmask_text
# ---------------------------------------------------------------------------

class TestUnmaskText:
    async def test_full_round_trip(self, store):
        mapping = {"{TC_No_1}": "10000000146", "{Email_1}": "ali@example.com"}
        await store.save_mapping("job-rt", mapping)

        masked = "Müşteri {TC_No_1} numaralı kişi, {Email_1} adresine bildirim gönderildi."
        result = await store.unmask_text("job-rt", masked)

        assert "10000000146" in result
        assert "ali@example.com" in result
        assert "{TC_No_1}" not in result
        assert "{Email_1}" not in result

    async def test_unknown_placeholder_left_intact(self, store):
        await store.save_mapping("job-unk", {"{TC_No_1}": "10000000146"})
        masked = "TC: {TC_No_1}, IBAN: {IBAN_1}"   # {IBAN_1} not in mapping
        result = await store.unmask_text("job-unk", masked)
        assert "10000000146" in result
        assert "{IBAN_1}" in result   # unknown — left as-is

    async def test_no_placeholders_in_text_returns_unchanged(self, store):
        await store.save_mapping("job-noph", {"{TC_No_1}": "10000000146"})
        text = "Bu metinde hiç placeholder yok."
        result = await store.unmask_text("job-noph", text)
        assert result == text

    async def test_empty_mapping_returns_text_unchanged(self, store):
        text = "Metin: {TC_No_1}"
        result = await store.unmask_text("nonexistent-job", text)
        assert result == text

    async def test_multiple_occurrences_all_replaced(self, store):
        # LLM might repeat a placeholder multiple times in its response
        await store.save_mapping("job-multi", {"{Person_1}": "Ahmet Yılmaz"})
        masked = "{Person_1} bugün geldi. {Person_1} dün de gelmişti."
        result = await store.unmask_text("job-multi", masked)
        assert result == "Ahmet Yılmaz bugün geldi. Ahmet Yılmaz dün de gelmişti."

    async def test_adjacent_placeholders(self, store):
        await store.save_mapping("job-adj", {
            "{TC_No_1}": "10000000146",
            "{Phone_No_1}": "05321234567",
        })
        masked = "{TC_No_1}{Phone_No_1}"
        result = await store.unmask_text("job-adj", masked)
        # Either order is fine depending on regex alternation; both values present
        assert "10000000146" in result
        assert "05321234567" in result

    async def test_longer_placeholder_takes_priority(self, store):
        # If {Person_1} and {Person_10} both exist, {Person_10} must not be
        # partially replaced as {Person_1} + "0".
        await store.save_mapping("job-len", {
            "{Person_1}": "Ali",
            "{Person_10}": "Mehmet",
        })
        masked = "İsimler: {Person_10} ve {Person_1}"
        result = await store.unmask_text("job-len", masked)
        assert result == "İsimler: Mehmet ve Ali"


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------

class TestDelete:
    async def test_delete_removes_key(self, store):
        await store.save_mapping("job-del", {"{TC_No_1}": "10000000146"})
        await store.delete("job-del")
        result = await store.get_all("job-del")
        assert result == {}

    async def test_delete_nonexistent_is_safe(self, store):
        await store.delete("does-not-exist")   # should not raise


# ---------------------------------------------------------------------------
# ttl()
# ---------------------------------------------------------------------------

class TestTTL:
    async def test_ttl_positive_after_save(self, store):
        await store.save_mapping("job-ttl", {"{TC_No_1}": "10000000146"})
        remaining = await store.ttl("job-ttl")
        assert remaining > 0

    async def test_ttl_negative_for_missing_key(self, store):
        remaining = await store.ttl("no-such-job")
        assert remaining == -2   # Redis: key does not exist


# ---------------------------------------------------------------------------
# End-to-end: regex engine → _apply_masks → MappingStore → unmask
# ---------------------------------------------------------------------------

class TestEndToEndRoundTrip:
    """
    Verifies the complete on-prem pipeline without any network calls.
    Regex detects PII → masks text → stores mapping → recovers original.
    """

    DOCUMENT = (
        "Sayın Müşterimiz,\n"
        "TC Kimlik No 10000000146 sahibi müşterimizin\n"
        "TR330006100519786457841326 numaralı IBAN hesabına\n"
        "15 Haziran 2024 tarihinde ₺12.500,00 transfer yapılmıştır.\n"
        "İletişim: ali.veli@banka.example.com veya 0532 123 45 67\n"
    )

    async def test_masked_text_contains_no_original_pii(self, store):
        spans = regex_engine.detect(self.DOCUMENT)
        masked, mapping = _apply_masks(self.DOCUMENT, spans)

        # None of the detected original values should appear in masked text
        for original in mapping.values():
            assert original not in masked, f"PII leaked into masked text: {original}"

    async def test_masked_text_contains_placeholders(self, store):
        spans = regex_engine.detect(self.DOCUMENT)
        masked, _ = _apply_masks(self.DOCUMENT, spans)
        assert "{" in masked   # at least one placeholder present

    async def test_full_unmask_recovers_original(self, store):
        spans = regex_engine.detect(self.DOCUMENT)
        masked, mapping = _apply_masks(self.DOCUMENT, spans)

        await store.save_mapping("job-e2e", mapping)
        recovered = await store.unmask_text("job-e2e", masked)

        assert recovered == self.DOCUMENT

    async def test_mapping_keys_match_placeholders_in_text(self, store):
        spans = regex_engine.detect(self.DOCUMENT)
        masked, mapping = _apply_masks(self.DOCUMENT, spans)

        # Every placeholder in the mapping should appear in the masked text
        for placeholder in mapping:
            assert placeholder in masked, f"Placeholder not found in masked text: {placeholder}"

    async def test_simulated_llm_response_unmasks_correctly(self, store):
        """LLM receives masked text, replies using placeholders, we de-mask."""
        spans = regex_engine.detect(self.DOCUMENT)
        masked, mapping = _apply_masks(self.DOCUMENT, spans)
        await store.save_mapping("job-llm", mapping)

        # Simulate LLM echoing one of the placeholders back
        tc_placeholder = next(k for k in mapping if k.startswith("{TC_No_"))
        llm_response = f"Verilen TC Kimlik No {tc_placeholder} için işlem tamamlandı."

        unmasked = await store.unmask_text("job-llm", llm_response)
        assert "10000000146" in unmasked
        assert tc_placeholder not in unmasked
