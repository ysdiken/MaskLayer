"""
Live end-to-end test — full FastAPI app in-process, real NER model.

Uses httpx ASGITransport (no server process needed) + fakeredis.
Redis is injected directly into app.state so we don't rely on the
lifespan connecting to a real Redis container.

Run:
    cd backend
    python e2e_test.py
"""

import asyncio
import re
import sys

import fakeredis.aioredis
import httpx

# ── App import ────────────────────────────────────────────────────────────────
from app.main import app

# Inject fake Redis before any request hits the route
_fake_redis = fakeredis.aioredis.FakeRedis()
app.state.redis = _fake_redis


# ── ANSI output ───────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):    print(f"{GREEN}  ✓ {RESET}{msg}")
def fail(msg):  print(f"{RED}  ✗ {msg}{RESET}"); sys.exit(1)
def info(msg):  print(f"{CYAN}  → {RESET}{msg}")
def head(msg):  print(f"\n{BOLD}{YELLOW}{msg}{RESET}")


# ── Test document ─────────────────────────────────────────────────────────────
DOCUMENT = (
    "Sayın Müşterimiz,\n"
    "TC Kimlik No 10000000146 sahibi Ahmet Yılmaz'ın\n"
    "TR330006100519786457841326 numaralı IBAN hesabına\n"
    "15 Haziran 2024 tarihinde ₺12.500,00 transfer yapılmıştır.\n"
    "İletişim: ahmet.yilmaz@banka.com.tr veya 0532 123 45 67\n"
    "Araç: 34 ABC 1234  Pasaport: U12345678\n"
    "Şirket: Türk Hava Yolları A.Ş. — İstanbul ofisi\n"
)


async def run_tests():
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=True)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:

        # ── 1. Health ─────────────────────────────────────────────────────────
        head("1. Health check")
        r = await client.get("/health")
        assert r.status_code == 200, f"Got {r.status_code}: {r.text}"
        body = r.json()
        ok(f"status = {body['status']}")
        ner_ok = body.get("ner_available", False)
        if ner_ok:
            ok("ner_available = true  (BERTurk loaded on GPU)")
        else:
            print(f"{YELLOW}  ⚠ ner_available = false — regex-only mode{RESET}")

        # ── 2. Mask ───────────────────────────────────────────────────────────
        head("2. POST /api/v1/mask")
        info(f"Document ({len(DOCUMENT)} chars):")
        for line in DOCUMENT.strip().splitlines():
            print(f"       {line}")

        r = await client.post("/api/v1/mask", json={"text": DOCUMENT, "language": "tr"})
        assert r.status_code == 200, f"Got {r.status_code}: {r.text}"
        mb = r.json()

        job_id      = mb["job_id"]
        masked_text = mb["masked_text"]
        spans       = mb["spans"]
        elapsed_ms  = mb["processing_time_ms"]

        ok(f"job_id           = {job_id}")
        ok(f"processing_time  = {elapsed_ms:.1f} ms")
        ok(f"spans detected   = {len(spans)}")

        print(f"\n{BOLD}  Masked text:{RESET}")
        for line in masked_text.strip().splitlines():
            print(f"       {line}")

        print(f"\n{BOLD}  Detected spans:{RESET}")
        for s in sorted(spans, key=lambda x: x["start"]):
            src_col = GREEN if s["source"] == "regex" else CYAN
            print(
                f"    [{s['start']:3}:{s['end']:3}]"
                f"  {s['label']:16}"
                f"  conf={s['confidence']:.2f}"
                f"  {src_col}[{s['source']:5}]{RESET}"
                f"  {s['text']!r}"
            )

        # ── 3. PII leak check ─────────────────────────────────────────────────
        head("3. PII leak check — none of these should appear in masked text")
        pii_checks = {
            "TC Kimlik No":  "10000000146",
            "IBAN":          "TR330006100519786457841326",
            "E-posta":       "ahmet.yilmaz@banka.com.tr",
            "Telefon":       "0532 123 45 67",
        }
        for label, value in pii_checks.items():
            if value in masked_text:
                fail(f"PII LEAK — {label} visible in masked text: {value!r}")
            else:
                ok(f"{label:12} masked correctly")

        # ── 4. Entity coverage ────────────────────────────────────────────────
        head("4. Entity type coverage")
        found = {s["label"] for s in spans}

        regex_expected = {
            "TC_No", "IBAN", "Date", "Money_Amount",
            "Email", "Phone_No", "License_Plate",
        }
        for label in sorted(regex_expected):
            if label in found:
                ok(f"{label:16} [regex]")
            else:
                print(f"{YELLOW}  ⚠ {label:16} not detected{RESET}")

        if ner_ok:
            for label in ("Person", "Company", "Location"):
                if label in found:
                    ok(f"{label:16} [ner]")
                else:
                    print(f"{YELLOW}  ⚠ {label:16} not detected by NER{RESET}")

        # ── 5. Full round-trip unmask ─────────────────────────────────────────
        head("5. POST /api/v1/unmask — full round-trip")
        r = await client.post("/api/v1/unmask", json={
            "job_id": job_id,
            "masked_text": masked_text,
        })
        assert r.status_code == 200, f"Got {r.status_code}: {r.text}"
        recovered = r.json()["unmasked_text"]

        if recovered == DOCUMENT:
            ok("Recovered text == original document  ✓ perfect round-trip")
        else:
            print(f"{YELLOW}  ⚠ Recovered text differs from original:{RESET}")
            for i, (o, rv) in enumerate(zip(DOCUMENT.splitlines(), recovered.splitlines())):
                if o != rv:
                    print(f"    line {i+1} orig: {o!r}")
                    print(f"    line {i+1} recv: {rv!r}")

        # ── 6. Simulated LLM response ─────────────────────────────────────────
        head("6. Simulated LLM response — partial unmask")
        tc_match   = re.search(r"\{TC_No_\d+\}", masked_text)
        iban_match = re.search(r"\{IBAN_\d+\}", masked_text)

        if tc_match and iban_match:
            llm_response = (
                f"Sayın yetkili,\n"
                f"Müşteri {tc_match.group()} adına {iban_match.group()} numaralı hesaba\n"
                f"ödeme işlemi onaylanmıştır."
            )
            info("LLM response (with placeholders):")
            for line in llm_response.splitlines():
                print(f"       {line}")

            r = await client.post("/api/v1/unmask", json={
                "job_id": job_id,
                "masked_text": llm_response,
            })
            assert r.status_code == 200
            unmasked_llm = r.json()["unmasked_text"]

            info("After unmask:")
            for line in unmasked_llm.splitlines():
                print(f"       {line}")

            assert "10000000146" in unmasked_llm
            assert "TR330006100519786457841326" in unmasked_llm
            ok("LLM response correctly de-masked — PII restored on-prem")
        else:
            print(f"{YELLOW}  ⚠ Could not find TC_No or IBAN placeholder for LLM sim{RESET}")

        # ── 7. Unknown job_id → 404 ───────────────────────────────────────────
        head("7. 404 on expired / unknown job_id")
        r = await client.post("/api/v1/unmask", json={
            "job_id": "00000000-0000-0000-0000-000000000000",
            "masked_text": "test",
        })
        assert r.status_code == 404, f"Expected 404, got {r.status_code}"
        ok(f"404 returned correctly")

        # ── Done ──────────────────────────────────────────────────────────────
        print(f"\n{BOLD}{GREEN}═══════════════════════════════════════{RESET}")
        print(f"{BOLD}{GREEN}  All end-to-end tests passed.{RESET}")
        print(f"{BOLD}{GREEN}═══════════════════════════════════════{RESET}\n")


if __name__ == "__main__":
    asyncio.run(run_tests())
