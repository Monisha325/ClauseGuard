# ClauseGuard

ClauseGuard is an AI-powered contract review assistant. A user uploads a
PDF or DOCX contract; an asynchronous pipeline extracts and chunks it into
individual clauses, classifies each clause into a legal category, indexes
it for semantic search, and an LLM agent flags risky clauses (severity +
explanation + citation) and proposes grounded negotiation points — all
surfaced through a React dashboard with email/OTP-verified authentication.

## Highlights

- **Fine-tuned DistilBERT clause classifier**: 97.9% test accuracy on a
  CUAD-derived contract subset vs. 41.3% for the live heuristic
  baseline's stage-1 classifier alone. See
  [`experiments/clause_classifier_finetune/`](./experiments/clause_classifier_finetune/).
- **LoRA/PEFT distillation experiment (in progress)**: distilling the
  live 70B risk-flagging teacher (Groq `llama-3.3-70b-versatile`) into an
  open-weight Qwen2-0.5B student, with a dedicated data pipeline and an
  honestly-reported status — see
  [`experiments/llm_distillation/`](./experiments/llm_distillation/) for
  exactly what's done vs. still in progress.
- **Retrieval-aware clause search**: BM25 + dense embeddings fused via
  Reciprocal Rank Fusion, reranked by a cross-encoder, served through a
  full FastAPI + PostgreSQL + React pipeline.
- **75% inference latency cut (1.6s → 403ms)**: root-caused a CPU
  self-attention cost blowup on long clause text and fixed it via
  sequence-length truncation — see `backend/retrieval/reranker.py` and
  [`docs/M1_Validation_Results.md`](./docs/M1_Validation_Results.md) for
  the full before/after measurement.

## Features

- **Email/OTP authentication** — signup with bcrypt-hashed passwords, a
  6-digit OTP verification code (bcrypt-hashed, 10-minute expiry, rate
  limited) sent via Brevo, JWT-based sessions.
- **PDF/DOCX ingestion** — real magic-byte file-type verification, a hard
  25MB size cap, OCR fallback (Tesseract) for scanned/image-only pages
  with a confidence-calibrated garbage filter.
- **Clause-aware chunking** — heading detection (numbered sections,
  articles, ALL-CAPS titles) with token-budgeted splitting/merging
  (`tiktoken`), never breaking mid-sentence.
- **Two-stage clause classification** — heading-lexicon pattern matching,
  falling back to an embedding-centroid classifier for anything the
  lexicon misses.
- **Hybrid semantic search** — vector search (Chroma) fused with BM25
  keyword search via Reciprocal Rank Fusion, reranked by a cross-encoder,
  gated by per-category confidence thresholds.
- **LLM risk analysis** — every clause is flagged (low/medium/high
  severity + explanation + citation) via forced tool-calling against
  Groq (`llama-3.3-70b-versatile`); high/medium clauses additionally get a
  grounded negotiation-point suggestion, with citations checked against
  the actual retrieved context.
- **Async processing** — upload returns immediately; a Celery worker runs
  the full pipeline in the background while the frontend polls live
  per-stage progress.
- **Contract history** — a paginated "My Contracts" view with per-contract
  risk summaries, and full deletion across Postgres, Chroma, and disk.
- **Per-user daily upload cap** and **provider rate-limit handling** (Voyage
  embeddings, Groq LLM calls) with real retry/backoff logic.
- **Audit trail** — every validation failure, retry, and flag decision is
  recorded for later inspection.

## Architecture

```
React SPA (Vite)  <-- JWT bearer -->  FastAPI (app container)
                                          |
                                          |-- Postgres (users, contracts,
                                          |    clauses, flagged_clauses,
                                          |    audit_log, upload_events)
                                          |-- Redis (Celery broker/backend)
                                          |-- enqueues -> Celery worker
                                                            |
                                                            |-- extract -> chunk
                                                            |-- classify
                                                            |-- embed/index -> ChromaDB
                                                            |-- flag clauses -> Groq
                                                            |-- suggest negotiation -> Groq
```

`app` (FastAPI/uvicorn) and `worker` (Celery) are the **same Docker image**
running different commands, sharing two named volumes
(`contract_storage`, `chroma_data`) since the worker processes files the
API container received.

## Tech Stack

| Layer | Technology |
|---|---|
| Backend framework | FastAPI, Uvicorn |
| Database | PostgreSQL 16 (SQLAlchemy 2.0 + psycopg2) |
| Async tasks | Celery 5.6, Redis 7 (broker + result backend) |
| Auth | PyJWT (HS256), bcrypt |
| Document parsing | PyMuPDF (PDF), python-docx, Tesseract OCR (pytesseract) |
| Tokenization | tiktoken (`cl100k_base`) |
| Embeddings | Voyage AI (`voyage-3-lite`) |
| Vector store | ChromaDB (persistent client) |
| Keyword search | rank-bm25 (BM25Okapi) |
| Reranking | sentence-transformers `CrossEncoder` (`ms-marco-MiniLM-L6-v2`) |
| LLM | Groq (`llama-3.3-70b-versatile`), forced tool-calling |
| Transactional email | Brevo REST API |
| Frontend | React 18, Vite, framer-motion |
| Infra | Docker Compose |

## Prerequisites

- Docker and Docker Compose
- API keys for:
  - [Voyage AI](https://www.voyageai.com/) (embeddings)
  - [Groq](https://console.groq.com/) (LLM tool-calling)
  - [Brevo](https://www.brevo.com/) (transactional email — free tier, 300/day)

## Setup

1. Clone the repo and copy the environment template:

   ```bash
   cp .env.example .env
   ```

2. Fill in `.env`:

   | Variable | Description |
   |---|---|
   | `POSTGRES_USER` / `POSTGRES_PASSWORD` / `POSTGRES_DB` | Postgres credentials (defaults work for local dev) |
   | `JWT_SECRET_KEY` | Random secret, **at least 16 characters** |
   | `VOYAGE_API_KEY` | Voyage AI API key |
   | `GROQ_API_KEY` | Groq API key |
   | `DAILY_UPLOAD_LIMIT_PER_USER` | Optional, default `10` |
   | `BREVO_API_KEY` | Brevo API key (Settings → SMTP & API → API Keys) |
   | `BREVO_SENDER_EMAIL` | Must be a **verified sender** on your Brevo account |

3. Start the backend stack:

   ```bash
   docker compose up --build
   ```

   This starts `db` (Postgres), `redis`, `app` (FastAPI on host port
   **8010**), and `worker` (Celery). The API is available at
   `http://localhost:8010`, with `GET /health` for a readiness check.

4. Start the frontend:

   ```bash
   cd frontend
   npm install
   npm run dev
   ```

   The dev server runs on `http://localhost:5174` (pinned in
   `vite.config.js`; the backend's CORS policy only allows this exact
   origin).

## Project Structure

```
backend/
  main.py                 FastAPI app entrypoint
  config.py                Environment-backed settings
  db.py                    SQLAlchemy engine/session setup
  routes/                  auth, contracts, retrieval endpoints
  models/                  SQLAlchemy models (User, Contract, Clause, ...)
  ingestion/                PDF/DOCX extraction, OCR, chunking, upload validation
  classification/           Heading-match + embedding-centroid clause classifiers
  embeddings/               Voyage AI client, Chroma indexing, shared rate limiting
  retrieval/                BM25, RRF fusion, cross-encoder reranker, confidence thresholds
  agent/                    Groq tool-calling: flag_clause, suggest_negotiation
  pipeline/                 End-to-end per-contract orchestration
  worker/                   Celery app + task wrapper
  auth/                     JWT issuing/verification
  services/                 OTP logic, Brevo email client
  middleware/               Per-user daily upload rate limiting
  observability/            Structured timing instrumentation, audit log
frontend/
  src/                      React components (Landing, AuthPage, Dashboard, ContractHistory)
eval/                       Retrieval/classification/generation evaluation scripts
docs/                       Design notes and validation results
```

## API Overview

| Endpoint | Description |
|---|---|
| `POST /signup` | Create an account (email + password) |
| `POST /verify-otp` | Verify signup with the emailed 6-digit code |
| `POST /resend-otp` | Request a new code (rate limited) |
| `POST /login` | Returns a bearer JWT |
| `GET /me` | Returns the authenticated user's ID |
| `POST /contracts/upload` | Upload a PDF/DOCX; returns immediately, processing runs async |
| `GET /contracts` | Paginated list of the user's past uploads with risk summaries |
| `GET /contracts/{id}/status` | Poll processing progress (`current_stage`, counts) |
| `GET /contracts/{id}/flagged-clauses` | Full per-clause risk assessment results |
| `POST /contracts/{id}/search` | Semantic search within a contract (optional reranking) |
| `DELETE /contracts/{id}` | Delete a contract across Postgres, Chroma, and disk |

All contract-scoped routes are JWT-protected and enforce per-user
ownership (a non-owned or nonexistent `contract_id` returns `404`, never
`403`, to avoid confirming existence to a non-owner).

## Evaluation

`eval/` contains standalone scripts (run inside the `app`/`worker`
container, since they depend on the live Postgres/Chroma config) for
Hit@K retrieval quality, reranker A/B comparison, threshold sanity
checks, and generation quality. See `docs/M1_Validation_Results.md` for
the current, honestly-reported state of these evaluations — including
which results are still inconclusive due to a small labeled dataset.

## Experiments

Both experiments below are **additive** — nothing in `experiments/`
replaces or modifies the live system in `backend/`.

- [`experiments/clause_classifier_finetune/`](./experiments/clause_classifier_finetune/) —
  a fine-tuned DistilBERT clause classifier, evaluated honestly against
  the live system's real stage-1 classifier. See that folder's
  `README.md`/`REPORT.md` for full methodology, results, and disclosed
  limitations (short version: 97.9% test accuracy on a 3-class subset,
  vs. 41.3%/43% coverage for the live system's stage-1 classifier alone
  — stage 2 was not measurable in the environment this was run in).
- [`experiments/llm_distillation/`](./experiments/llm_distillation/) —
  **in progress**: distilling the live 70B risk-flagging teacher into an
  open-weight Qwen2-0.5B student via PEFT/LoRA. The data pipeline (subset
  selection, real teacher labeling, contract-grouped train/val/test
  split) is done and the training pipeline's mechanics are smoke-tested;
  the actual fine-tuning run and the teacher-vs-student evaluation
  harness are not yet in this snapshot. See that folder's `README.md`
  for the exact phase-by-phase status.

## Known Limitations

- Retrieval/classification quality evals currently run against a small
  (N=5) labeled fixture — see `docs/M1_Validation_Results.md`.
- Single Uvicorn worker / single Celery worker process; a couple of
  in-memory caches (classification similarity, centroid vectors) are
  process-local and would need a shared cache to scale horizontally.
- No virus/malware scanning on uploads (only structural file-type
  validation).
- Voyage AI's free-tier rate limit (3 requests/minute) is the effective
  throughput ceiling for embedding-heavy operations.
