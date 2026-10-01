"""
Unit tests for the keyword-anchored document-number patterns:
Customer_No, SGK_No, Policy_No, Contract_No, Invoice_No, Reference_No,
plus the Anne/Baba Adı person-field pattern.

These tags have no inherent structure (no checksum) — detection relies on the
Turkish label preceding the number. The keyword itself must stay unmasked
(capture_group=1): "Müşteri No:" remains readable, only the digits get a span.
"""

import pytest

from app.masking.regex_engine import engine


def spans_of(text: str, label: str):
    return [s for s in engine.detect(text) if s.label == label]


# ---------------------------------------------------------------------------
# Customer_No — the must-have
# ---------------------------------------------------------------------------

class TestCustomerNo:

    def test_basic(self):
        text = "Müşteri No: 12345678 adına kayıtlıdır."
        spans = spans_of(text, "Customer_No")
        assert len(spans) == 1
        assert spans[0].text == "12345678"

    def test_numarasi_variant(self):
        text = "Müşteri Numarası 0012345678 ile giriş yapınız."
        spans = spans_of(text, "Customer_No")
        assert len(spans) == 1
        assert spans[0].text == "0012345678"

    def test_all_caps_dotted_i(self):
        # "MÜŞTERİ NUMARASI" — Turkish dotted İ in uppercase
        text = "MÜŞTERİ NUMARASI: 99887766"
        spans = spans_of(text, "Customer_No")
        assert len(spans) == 1
        assert spans[0].text == "99887766"

    def test_ocr_degraded_ascii(self):
        # OCR often strips diacritics: "Musteri No"
        text = "Musteri No: 445566778"
        spans = spans_of(text, "Customer_No")
        assert len(spans) == 1

    def test_keyword_not_masked(self):
        text = "Müşteri No: 12345678"
        span = spans_of(text, "Customer_No")[0]
        assert text[span.start: span.end] == "12345678"
        assert "Müşteri" not in span.text

    def test_no_match_without_keyword(self):
        assert spans_of("Sipariş kodu 12345678 hazırlandı.", "Customer_No") == []

    def test_no_match_keyword_without_no(self):
        assert spans_of("Müşteri memnuniyeti %95 olarak ölçüldü.", "Customer_No") == []


# ---------------------------------------------------------------------------
# Other document numbers
# ---------------------------------------------------------------------------

class TestSgkNo:

    def test_sgk_sicil(self):
        spans = spans_of("SGK Sicil No: 1234567890123", "SGK_No")
        assert len(spans) == 1
        assert spans[0].text == "1234567890123"

    def test_ssk_variant(self):
        spans = spans_of("SSK No: 12345678", "SGK_No")
        assert len(spans) == 1


class TestPolicyNo:

    def test_alphanumeric_policy(self):
        spans = spans_of("Poliçe No: POL-2024-123456 yürürlüktedir.", "Policy_No")
        assert len(spans) == 1
        assert spans[0].text == "POL-2024-123456"

    def test_numeric_policy(self):
        spans = spans_of("Poliçe Numarası: 887766554", "Policy_No")
        assert len(spans) == 1


class TestContractNo:

    def test_alphanumeric_contract(self):
        spans = spans_of("Sözleşme No: SZL-2024/00123 ekte sunulmuştur.", "Contract_No")
        assert len(spans) == 1
        assert spans[0].text == "SZL-2024/00123"

    def test_contract_beats_case_no_on_year_slash_format(self):
        # "2024/12345" alone matches the bare Case_No pattern; with the
        # Sözleşme keyword the higher-confidence Contract_No must win.
        text = "Sözleşme No: 2024/12345"
        all_spans = engine.detect(text)
        labels = [s.label for s in all_spans if s.text == "2024/12345"]
        assert labels == ["Contract_No"]


class TestInvoiceNo:

    def test_efatura_format(self):
        spans = spans_of("Fatura No: GIB2024000001234", "Invoice_No")
        assert len(spans) == 1
        assert spans[0].text == "GIB2024000001234"


class TestReferenceNo:

    def test_dekont(self):
        spans = spans_of("Dekont No: 99887766", "Reference_No")
        assert len(spans) == 1

    def test_islem_dotted_i(self):
        spans = spans_of("İşlem No: TX-2024-001", "Reference_No")
        assert len(spans) == 1
        assert spans[0].text == "TX-2024-001"

    def test_referans(self):
        spans = spans_of("Referans Numarası: REF-998877", "Reference_No")
        assert len(spans) == 1


# ---------------------------------------------------------------------------
# Anne/Baba Adı person fields (KYC forms)
# ---------------------------------------------------------------------------

class TestParentNameFields:

    def test_anne_adi(self):
        spans = spans_of("Anne Adı: Fatma  Baba Adı: Hasan", "Person")
        names = sorted(s.text for s in spans)
        assert names == ["Fatma", "Hasan"]

    def test_two_word_name(self):
        spans = spans_of("Baba Adı: Hasan Demir", "Person")
        assert len(spans) == 1
        assert spans[0].text == "Hasan Demir"

    def test_lowercase_not_matched(self):
        # Value must be Titlecase — lowercase words after the keyword are
        # not names (e.g. instructions in a form template).
        assert spans_of("Anne adı: belirtilmemiş", "Person") == []
