"""
Synthetic training-data generator for NER domain-adaptation.

Produces BIO-labelled Turkish contract/form sentences (7-label scheme matching
the baked model: B/I-PER, B/I-LOC, B/I-ORG, O). Two goals:

  1. POSITIVE signal — real Person/Org/Location entities in contract contexts,
     so PER/LOC/ORG recall is preserved.
  2. NEGATIVE signal (the point) — capitalised Turkish common/domain nouns
     (Kredi, Platin, Sözleşme, Vergi, …) shown as O, including at clause starts
     and mid-sentence exactly as contracts capitalise them. This teaches the
     model that capitalised ≠ organisation, fixing the real-doc over-tagging.

DISJOINT from the eval sets: the entity/term pools and templates here are not
the gold synthetic corpus (eval/gold) nor the real contracts (eval/real).

Usage:
    python -m finetune.data_gen            # writes finetune/train.jsonl
"""

from __future__ import annotations

import json
import random
from pathlib import Path

OUT = Path(__file__).resolve().parent / "train.jsonl"

# ---------------------------------------------------------------------------
# Entity pools (POSITIVE — labelled PER / ORG / LOC)
# ---------------------------------------------------------------------------

FIRST_NAMES = [
    "Ahmet", "Mehmet", "Mustafa", "Ali", "Hüseyin", "Hasan", "İbrahim", "Murat",
    "Osman", "Kemal", "Orhan", "Selim", "Levent", "Fatih", "Kenan", "Hakan",
    "Gökhan", "Serkan", "Onur", "Cem", "Volkan", "Uğur", "Sinan", "Kerem",
    "Emre", "Burak", "Kadir", "Tarık", "Selçuk", "Cenk", "Erdem", "Barış",
    "Ayşe", "Fatma", "Emine", "Hatice", "Zeynep", "Elif", "Meryem", "Sultan",
    "Burcu", "Sema", "Selin", "Defne", "Ece", "Pınar", "Yasemin", "Esra",
    "Merve", "Büşra", "Derya", "Sibel", "Gamze", "Ebru", "Aslı", "Dilara",
    "Özge", "Gizem", "Ceren", "Damla", "Bahar", "Nur", "Hande", "Şeyma",
]
SURNAMES = [
    "Yılmaz", "Demir", "Şahin", "Çelik", "Yıldız", "Yıldırım", "Öztürk", "Aydın",
    "Özdemir", "Arslan", "Doğan", "Kılıç", "Aslan", "Çetin", "Kaya", "Koç",
    "Kurt", "Özkan", "Şimşek", "Polat", "Korkmaz", "Çakır", "Güneş", "Aksoy",
    "Bulut", "Keskin", "Tekin", "Acar", "Karaca", "Bozkurt", "Aktaş", "Şen",
    "Köse", "Turan", "Yalçın", "Coşkun", "Eren", "Güler", "Sarı", "Uçar",
    "Kara", "Köksal", "Duran", "Ünal", "Kaplan", "Çiftçi", "Erdoğan", "Avcı",
]
# Company name building blocks → ORG (multi-word, with legal form)
CO_NAME = ["Mavi", "Yıldız", "Anadolu", "Doğuş", "Ada", "Güneş", "Deniz", "Akın",
           "Ege", "Toros", "Marmara", "Bereket", "Öz", "Birlik", "Zirve", "Nova",
           "Teknova", "Pamir", "Çınar", "Altın", "Kuzey", "Batı", "Star", "Pi"]
CO_SECTOR = ["Teknoloji", "Tekstil", "İnşaat", "Otomotiv", "Bilişim", "Gıda",
             "Enerji", "Lojistik", "Yazılım", "Sigorta", "Turizm", "Makine",
             "Sanayi", "Ticaret", "Tarım", "Mobilya", "Kimya", "Elektronik"]
CO_LEGAL = ["A.Ş.", "Ltd. Şti.", "A.Ş.", "Holding A.Ş.", "San. ve Tic. A.Ş.", "A.Ş."]

PLACES = [
    "İstanbul", "Ankara", "İzmir", "Bursa", "Antalya", "Adana", "Konya", "Mersin",
    "Kocaeli", "Gaziantep", "Kayseri", "Eskişehir", "Samsun", "Denizli", "Trabzon",
    "Sakarya", "Manisa", "Muğla", "Aydın", "Balıkesir", "Tekirdağ", "Hatay",
    "Kadıköy", "Beşiktaş", "Şişli", "Çankaya", "Bornova", "Nilüfer", "Muratpaşa",
]

# ---------------------------------------------------------------------------
# Common / domain nouns (NEGATIVE — labelled O even though capitalised)
# The core of the fine-tune: these are what the WikiANN model wrongly tags ORG.
# ---------------------------------------------------------------------------

COMMON_NOUNS = [
    "Kredi", "Sözleşme", "Hesap", "Banka", "Müşteri", "Taşıt", "Araç", "Platin",
    "Altın", "Gümüş", "Mevduat", "Vergi", "Faiz", "Taksit", "Madde", "Taraf",
    "Ücret", "Teminat", "Rehin", "Temerrüt", "Sigorta", "Anapara", "Bedel",
    "Tutar", "Ödeme", "Tahsilat", "Fatura", "Dekont", "Form", "Belge", "Talimat",
    "Yönetmelik", "Tüzük", "Genelge", "Borç", "Alacak", "Gelir", "Gider",
    "Masraf", "Komisyon", "Prim", "Poliçe", "Kasko", "Acente", "Şube", "Talep",
    "Onay", "İmza", "Tebligat", "Bildirim", "Beyan", "Taahhüt", "Yükümlülük",
    "Sorumluluk", "Koşul", "Şart", "Hüküm", "Esas", "Karar", "Dosya", "Mahsup",
    "Muaccel", "Bileşik", "Akdi", "Nakit", "Döviz", "Bakiye", "Limit", "Vade",
    "Plan", "Bilgilendirme", "Başvuru", "Hizmet", "Ürün", "Tedarik", "Teslimat",
    "Garanti", "Cayma", "Fesih", "İcra", "Haciz", "Kefalet", "İpotek", "Virman",
    "Havale", "Transfer", "Lehtar", "Vekil", "Vekâlet", "Kurum", "Personel",
    "Çalışan", "İşveren", "Kiracı", "Alıcı", "Satıcı", "Maliyet", "Oran", "Fon",
]
# Multi-word domain phrases that whole-phrase over-tag as ORG (all O)
COMMON_PHRASES = [
    ["Taşıt", "Kredisi"], ["Tüketici", "Kredisi"], ["Bağlı", "Kredi"],
    ["Konut", "Kredisi"], ["Kıymetli", "Maden", "Hesabı"], ["Vadeli", "Mevduat"],
    ["Rehin", "Tesis", "Ücreti"], ["Kredi", "Tahsis", "Ücreti"],
    ["Akdi", "Faiz", "Oranı"], ["Temerrüt", "Faiz", "Oranı"], ["Ödeme", "Planı"],
    ["Bilgilendirme", "Formu"], ["Kredi", "Sözleşmesi"], ["Hesap", "Özeti"],
]

# ---------------------------------------------------------------------------
# Templates — segments are: ("lit", text) | ("PER",) | ("ORG",) | ("LOC",) |
#                           ("TERM",) | ("TERMS",)  (TERMS = a few O nouns)
# ---------------------------------------------------------------------------

TEMPLATES = [
    [("TERM",), ("lit", "kapsamında"), ("PER",), ("lit", "ile"), ("ORG",), ("lit", "arasında imzalanmıştır.")],
    [("PER",), ("lit", ","), ("LOC",), ("lit", "şubesinde"), ("TERM",), ("lit", "başvurusu yaptı.")],
    [("lit", "Bu"), ("TERM",), ("lit", ","), ("ORG",), ("lit", "tarafından düzenlenmiştir.")],
    [("TERM",), ("lit", "tutarı"), ("ORG",), ("lit", "hesabına yatırılacaktır.")],
    [("lit", "Sayın"), ("PER",), ("lit", ","), ("TERM",), ("lit", "talebiniz onaylanmıştır.")],
    [("ORG",), ("lit", "adına"), ("PER",), ("lit", "tarafından"), ("TERM",), ("lit", "imzalanmıştır.")],
    [("TERM",), ("lit", "ve"), ("TERM",), ("lit", "hükümleri"), ("TERM",), ("lit", "uyarınca uygulanır.")],
    [("lit", "Davacı"), ("PER",), ("lit", "ile davalı"), ("ORG",), ("lit", "arasındaki dava görülmüştür.")],
    [("PER",), ("lit", ","), ("TERM",), ("lit", "bedelini"), ("LOC",), ("lit", "ofisinde ödedi.")],
    [("TERMS",), ("lit", "bu sözleşmenin ayrılmaz parçasıdır.")],
    [("lit", "İşbu"), ("TERM",), ("lit", ","), ("ORG",), ("lit", "ile"), ("PER",), ("lit", "arasında akdedilmiştir.")],
    [("PER",), ("lit", "adlı müşterimizin"), ("TERM",), ("lit", "hesabı"), ("LOC",), ("lit", "şubesinde açılmıştır.")],
    [("TERM",), ("lit", "Tutarı,"), ("TERM",), ("lit", "Maliyeti ve"), ("TERM",), ("lit", "ödenecektir.")],
    [("ORG",), ("lit", ","), ("TERM",), ("lit", "kapsamında"), ("LOC",), ("lit", "bölgesinde faaliyet gösterir.")],
    [("lit", "Taşınmaz"), ("PER",), ("lit", "adına kayıtlı olup"), ("TERM",), ("lit", "altına alınmıştır.")],
    [("TERM",), ("lit", "ödeme planınız"), ("ORG",), ("lit", "tarafından hazırlanmıştır.")],
    [("lit", "Yetkili"), ("PER",), ("lit", ","), ("ORG",), ("lit", "adına beyanda bulunmuştur.")],
    [("lit", "Müşteri"), ("PER",), ("lit", ","), ("LOC",), ("lit", "ilinde ikamet etmektedir.")],
    [("TERM",), ("lit", "ödemenizin gecikmesi halinde"), ("TERM",), ("lit", "faizi işletilir.")],
    [("PER",), ("lit", "ve"), ("PER",), ("lit", "şirketi"), ("ORG",), ("lit", "kurmuştur.")],
]


def _entity_tokens(label: str) -> tuple[list[str], list[str]]:
    if label == "PER":
        val = f"{random.choice(FIRST_NAMES)} {random.choice(SURNAMES)}"
    elif label == "ORG":
        val = f"{random.choice(CO_NAME)} {random.choice(CO_SECTOR)} {random.choice(CO_LEGAL)}"
    else:  # LOC
        val = random.choice(PLACES)
    words = val.split()
    tags = [f"B-{label}"] + [f"I-{label}"] * (len(words) - 1)
    return words, tags


def _fill(segments) -> tuple[list[str], list[str]]:
    toks: list[str] = []
    tags: list[str] = []
    for seg in segments:
        kind = seg[0]
        if kind == "lit":
            for w in seg[1].split():
                toks.append(w); tags.append("O")
        elif kind in ("PER", "ORG", "LOC"):
            w, t = _entity_tokens(kind)
            toks.extend(w); tags.extend(t)
        elif kind == "TERM":
            if random.random() < 0.35:
                phrase = random.choice(COMMON_PHRASES)
                for w in phrase:
                    toks.append(w); tags.append("O")
            else:
                toks.append(random.choice(COMMON_NOUNS)); tags.append("O")
        elif kind == "TERMS":
            for _ in range(random.randint(2, 4)):
                toks.append(random.choice(COMMON_NOUNS)); tags.append("O")
    return toks, tags


def generate(n: int = 6000, seed: int = 13) -> list[dict]:
    random.seed(seed)
    rows = []
    for _ in range(n):
        toks, tags = _fill(random.choice(TEMPLATES))
        rows.append({"tokens": toks, "tags": tags})
    return rows


def main() -> None:
    rows = generate()
    with OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # Quick stats
    from collections import Counter
    tagc = Counter(t for r in rows for t in r["tags"])
    print(f"Wrote {len(rows)} sentences → {OUT}")
    print("tag counts:", dict(tagc))
    print("example:", rows[0]["tokens"][:12], rows[0]["tags"][:12])


if __name__ == "__main__":
    main()
