# MaskLayer

**Context-preserving PII masking for Turkish documents.** MaskLayer replaces
personal and financial data with consistent placeholders, so sensitive
documents can be processed by LLMs and other external services without the
original values ever leaving your infrastructure. The mapping back to the
original values is stored on-prem and can be reversed when needed.

```
IN : Müşteri Ahmet Yılmaz (TC Kimlik No: 10000000146), Mavi Ada Teknoloji A.Ş. adına
     İstanbul şubesinden TR330006100519786457841326 IBAN'lı hesaba 125.000,00 TL gönderdi.
     İletişim: ahmet.yilmaz@example.com, 0532 123 45 67. Ahmet Yılmaz talimatı onayladı.

OUT: Müşteri {Person_1} (TC Kimlik No: {TC_No_1}), {Company_1} adına
     {Location_1} şubesinden {IBAN_1} IBAN'lı hesaba {Money_Amount_1} gönderdi.
     İletişim: {Email_1}, {Phone_No_1}. {Person_1} talimatı onayladı.
```

The same entity always gets the same placeholder (`{Person_1}` appears twice
above), so the masked text stays readable and useful to a downstream model.

## Background

MaskLayer started as my master's thesis project at İstanbul Technical
University (İTÜ). It is a personal project and is not affiliated with or
endorsed by İTÜ. The motivation: Turkish organisations bound by KVKK (Turkey's
personal data protection law) and BDDK (banking regulator) rules want to use
LLMs on contracts, forms and statements, but cannot send customer PII to them.

## How it works

1. **Text extraction**: pdfplumber for digital PDFs, Tesseract 5 (Turkish)
   for scanned pages and images, python-docx for DOCX.
2. **Detection**, three complementary engines:
   - **Regex + validators**: TC Kimlik No, IBAN, tax number and card numbers
     (with checksum validation), phone, email, dates, money amounts, licence
     plates, case numbers, addresses, and keyword-anchored IDs (customer,
     contract, policy, invoice, SGK and reference numbers).
   - **Transformer NER**: a BERTurk-based Turkish NER model for persons,
     companies, locations, titles and facilities, with long-document chunking
     and precision filters for legal citations, defined contract terms and
     public institutions.
   - **Gazetteer**: company legal-form detection (`A.Ş.`, `Ltd. Şti.`, …) and
     Turkish name lists.
3. **Span merging**: overlap resolution (deterministic regex types win;
   otherwise the longest span) and coreference grouping, so every mention of an
   entity shares one placeholder.
4. **Masking policy**: an admin-configurable filter that chooses which labels
   are rendered as placeholders, without affecting detection.
5. **Mapping store**: placeholder ↔ original value maps in Redis with a TTL,
   used by `/api/v1/unmask` to restore the original text on-prem.
6. **Audit and review**: PostgreSQL job and audit records, plus an analyst
   correction loop whose output can be exported as training data.

The React frontend supports drag-and-drop upload, a side-by-side
original/masked view with span review, the masking-policy admin panel, and an
optional chat sidebar.

> **Note on the chat sidebar:** it is the only feature that calls an external
> service (OpenAI), and it only ever sends the **masked** text. It only works
> when `OPENAI_API_KEY` is set. Everything else runs fully offline.

## Results

Evaluated with the included harness ([`backend/eval`](backend/eval/README.md)):

| test set | metric | result |
|----------|--------|--------|
| Synthetic gold corpus (22 docs, 158 spans) | strict micro F1, regex + NER + gazetteer | **0.997** |
| | PII character leak rate | **0.0%** with the fine-tuned model (0.9% baseline) |
| Real bank contract (customer PII) | F1, baseline NER → fine-tuned NER | 0.53 → **0.71** |

The synthetic score is an upper bound. On real contracts the out-of-domain NER
model over-tagged capitalised legal and banking terms as companies; domain
fine-tuning fixed most of this. See
[`backend/eval/real/README.md`](backend/eval/real/README.md) and
[`backend/finetune/README.md`](backend/finetune/README.md) for the full
analysis and its limitations (n=5 real documents).

## Getting started

**Prerequisites:** Python 3.11+, Node.js 18+, Docker, and
[Tesseract 5](https://github.com/tesseract-ocr/tesseract) with Turkish language
data (`tur`).

```bash
# 1. Configuration: copy the example and set your own passwords
cp .env.example .env

# 2. Infrastructure (Redis + PostgreSQL)
docker compose up -d

# 3. Backend
cd backend
pip install -r requirements.txt
python scripts/bake_model.py        # one-time download of the base NER model
uvicorn app.main:app --port 8000    # or ./start.ps1 on Windows (loads .env)

# 4. Frontend (in a second terminal)
cd frontend
npm install
npm run dev
```

After step 3 the backend runs fully offline. `docker-compose.yml` also starts
MinIO, which is reserved for document storage and not used by the backend yet.

### Fine-tuned model (optional, recommended)

The fine-tuned NER weights are not distributed in this repository, but you can
reproduce them. Training data is generated synthetically and a GPU is
recommended:

```bash
cd backend
python -m finetune.data_gen   # writes finetune/train.jsonl
python -m finetune.train      # writes models/bert-turkish-ner-ft/
```

The engine uses the fine-tuned model automatically when it is present,
otherwise the base model. Override with `MASKLAYER_NER_MODEL=<path>`.

## API

| method | endpoint | purpose |
|--------|----------|---------|
| `GET` | `/health` | liveness check |
| `POST` | `/api/v1/mask` | mask raw text |
| `POST` | `/api/v1/upload` | extract and mask a PDF, image or DOCX |
| `POST` | `/api/v1/unmask` | put original values back into text (e.g. an LLM response) using a job's mapping |
| `POST` | `/api/v1/ablation` | compare regex-only / NER-only / full detection |
| `POST` | `/api/v1/corrections` | log an analyst correction |
| `GET` | `/api/v1/corrections/stats` | correction statistics |
| `GET` | `/api/v1/corrections/training-data` | export corrections as training data |
| `GET` / `PUT` | `/api/v1/admin/mask-policy` | read or update the masking policy |
| `POST` | `/api/v1/chat` | optional chat over the masked text (OpenAI) |

Interactive docs are available at `http://localhost:8000/docs` while the
backend is running.

## Development

```bash
cd backend
pytest                                         # unit tests
python -m eval.run_eval --modes regex,ner,full # detection evaluation / ablation
```

```
backend/
  app/          FastAPI app: api/, masking/ (pipeline, engines), ocr/, db/
  config/       masking policy
  eval/         gold corpus, scorer, results history
  finetune/     synthetic data generator and NER fine-tuning
  scripts/      model download
  tests/
frontend/       React + Vite + TailwindCSS
infra/          PostgreSQL init script
```

## Limitations

- Detection quality on real documents is validated on a small sample (n=5
  bank contracts). Treat the system as a strong first pass with human review,
  not as a guarantee of complete anonymisation.
- Detection is built and tuned for Turkish; other languages are not supported.
- Masking reduces, but does not eliminate, re-identification risk from
  context. Compliance with KVKK or any other regulation remains the
  responsibility of the deploying organisation.

## Security

MaskLayer is a research prototype and ships **without authentication**:

- Every API endpoint is open, including `/api/v1/unmask` (which returns the
  original values for a job) and `PUT /api/v1/admin/mask-policy`.
- CORS allows requests from any origin.
- `docker-compose.yml` binds Redis, PostgreSQL and MinIO to `127.0.0.1` only.
  Keep it that way: Redis holds the placeholder ↔ original value maps.

Run it only on a trusted machine. Before exposing it to any network, add
authentication (for example an API gateway or OAuth2), restrict CORS to your
frontend's origin, and set strong passwords in `.env`.

To report a vulnerability, please use GitHub's private
[security advisory](https://github.com/ysdiken/MaskLayer/security/advisories/new)
form rather than a public issue.

## License

Licensed under the [Apache License 2.0](LICENSE). You are free to use, modify
and distribute this project, including commercially, provided you keep the
copyright and [NOTICE](NOTICE) attribution. If you use MaskLayer in research,
please cite it (see [`CITATION.cff`](CITATION.cff) or GitHub's
"Cite this repository" button).
