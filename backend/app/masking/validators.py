"""
Checksum and format validators for Turkish PII types.

Each function takes a raw matched string (may contain spaces/dashes) and
returns True if the value is structurally valid. These are called by the
regex engine to filter out digit sequences that happen to match a pattern
but fail the mathematical check — the primary mechanism for keeping
false-positive rates low without NER.

Thesis note: Measuring precision before vs. after checksum validation is a
clean ablation experiment. Expect ~30-50% FP reduction on TC/IBAN/VKN for
free-text documents.
"""


def validate_tc_kimlik(raw: str) -> bool:
    """
    Validate Turkish National ID (TC Kimlik No) using the official checksum.

    Algorithm (published by Nüfus ve Vatandaşlık İşleri Genel Müdürlüğü):
      - 11 digits, first digit ≠ 0
      - d[9]  = (7 * sum_of_odd_positions[0,2,4,6,8] - sum_of_even_positions[1,3,5,7]) mod 10
      - d[10] = (sum of first 10 digits) mod 10
    """
    digits = raw.replace(" ", "").replace("-", "")
    if len(digits) != 11 or not digits.isdigit() or digits[0] == "0":
        return False
    d = [int(c) for c in digits]

    odd_sum = d[0] + d[2] + d[4] + d[6] + d[8]
    even_sum = d[1] + d[3] + d[5] + d[7]
    check10 = (7 * odd_sum - even_sum) % 10
    if check10 != d[9]:
        return False

    check11 = sum(d[:10]) % 10
    return check11 == d[10]


def validate_iban(raw: str) -> bool:
    """
    Validate a Turkish IBAN using the ISO 13616 MOD-97 check.

    Turkish IBANs are always 26 characters: TR + 2 check digits + 22-digit BBAN.
    Spaces and dashes are stripped before validation.
    """
    clean = raw.replace(" ", "").replace("-", "").upper()
    if not clean.startswith("TR") or len(clean) != 26:
        return False
    if not clean[2:].isdigit():
        return False

    # Rearrange: move first 4 chars to end, then convert letters to digits
    rearranged = clean[4:] + clean[:4]
    numeric = ""
    for ch in rearranged:
        if ch.isalpha():
            numeric += str(ord(ch) - ord("A") + 10)
        else:
            numeric += ch

    return int(numeric) % 97 == 1


def validate_vkn(raw: str) -> bool:
    """
    Validate Turkish Tax ID (Vergi Kimlik No / VKN) checksum.

    Algorithm (Gelir İdaresi Başkanlığı):
      For i in 0..8:
        tmp[i] = (digit[i] + (9 - i)) mod 10
        if tmp[i] == 0: v[i] = 0
        else: v[i] = (tmp[i] * 2^(9-i)) mod 9; if v[i] == 0 → v[i] = 9
      check = (10 - (sum(v) mod 10)) mod 10
      check must equal digit[9]

    Note: VKN is 10 digits. TC Kimlik is 11 digits — they cannot overlap if
    word-boundary anchored, so running both patterns is safe.
    """
    digits = raw.replace(" ", "").replace("-", "")
    if len(digits) != 10 or not digits.isdigit():
        return False
    d = [int(c) for c in digits]

    v = []
    for i in range(9):
        tmp = (d[i] + (9 - i)) % 10
        if tmp == 0:
            v.append(0)
        else:
            val = (tmp * (2 ** (9 - i))) % 9
            v.append(val if val != 0 else 9)

    check = (10 - (sum(v) % 10)) % 10
    return check == d[9]


def luhn_check(raw: str) -> bool:
    """
    Luhn algorithm for credit/debit card number validation.

    Strips all non-digit characters before checking. Valid for 13-19 digit
    card numbers (Visa 16, Mastercard 16, Amex 15, etc.).
    """
    digits = [int(c) for c in raw if c.isdigit()]
    if len(digits) < 13 or len(digits) > 19:
        return False

    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:  # every second digit from the right
            d *= 2
            if d > 9:
                d -= 9
        total += d

    return total % 10 == 0


# ---------------------------------------------------------------------------
# Province code lookup for Turkish license plates (01 Adana → 81 Düzce)
# ---------------------------------------------------------------------------

VALID_PROVINCE_CODES: frozenset[str] = frozenset(
    str(i).zfill(2) for i in range(1, 82)
)


def validate_license_plate(raw: str) -> bool:
    """Check that the plate starts with a valid Turkish province code (01–81)."""
    code = raw.replace(" ", "")[:2]
    return code in VALID_PROVINCE_CODES
