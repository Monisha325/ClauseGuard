"""M10: embed every persisted clause for a contract and index it into
ChromaDB, respecting Voyage AI's confirmed free-tier limits.

Scope: ONE contract per call, single-process, synchronous. No retrieval/
query logic (M11), no reranking, no FastAPI route, no Celery wiring (M15).
Not a production-grade distributed rate limiter — there's no shared state
across concurrent processes/users, just a simple, correct pacing
mechanism sufficient for indexing one contract at a time.

Rate limits: M9's independent review confirmed via real API calls that
this Voyage account is on the free tier — 3 requests/minute, 10,000
tokens/minute. Both are respected by rate_limit.pace() below (see that
module's own docstring). On a genuine RateLimitError (e.g. from a
concurrent test run stacking on top of this process's own pacing),
rate_limit.embed_with_retry() waits ~65s and retries up to twice before
giving up — a transient hit doesn't fail the whole contract.

POST-INCIDENT REFACTOR: the pacing/retry logic that used to live here
directly (_pace(), _embed_with_retry(), _estimate_tokens(), and the 5
rate-limit constants) has been extracted, unchanged in algorithm or
numbers, into embeddings/rate_limit.py -- a real production incident
(classification/centroid_fallback.py's M23 stage-2 fallback making bare,
unpaced Voyage calls of its own, hitting this SAME account's SAME limit,
and crashing a real pipeline run) needed this exact same protection at a
second call site, and reimplementing it a second time was rejected in
favor of both call sites sharing one already-verified implementation.
This module's own behavior is unchanged: same constants, same algorithm,
same log content at this call site (see the loop below) -- only the
code's location moved.

Chroma: a single persistent collection ("clauses") holds every contract's
clauses. Retrieval-time isolation (M11) will filter by the contract_id/
user_id metadata stored on every entry here — this milestone's job is
making sure that metadata is always present and correct, not the
filtering itself.
"""

import logging
import time
import uuid
from collections import deque

import chromadb

from embeddings.rate_limit import (
    MIN_SECONDS_BETWEEN_REQUESTS,
    REQUESTS_PER_MINUTE,
    TOKENS_PER_MINUTE,
    embed_with_retry,
    estimate_tokens,
    pace,
)
from embeddings.voyage_client import MODEL
from models.clause import Clause
from models.contract import Contract

logger = logging.getLogger("clauseguard.embeddings.index")

CHROMA_PERSIST_DIR = "/app/chroma_data"
COLLECTION_NAME = "clauses"


def _get_chroma_collection():
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    return client.get_or_create_collection(COLLECTION_NAME)


def delete_contract_vectors(contract_id: uuid.UUID) -> int:
    """M37: delete every Chroma vector belonging to contract_id, matched
    by the exact same "contract_id" metadata field index_contract_clauses()
    above writes on every upsert -- the deletion counterpart to that
    function's own insert path, scoped the identical way M11/M24's own
    retrieval-time queries already scope by contract_id.

    Returns the real count of vectors that existed for this contract_id
    immediately BEFORE the delete -- queried first via collection.get(),
    not assumed or inferred from the delete call's own (non-informative)
    return value -- so the caller has real, verifiable evidence of how
    many vectors were actually removed. Callers needing to CONFIRM zero
    vectors remain afterward should query collection.get(where=...) again
    themselves post-delete (this function does not re-verify its own
    work -- that is the caller's/the test's job, kept independent on
    purpose).

    Idempotent: calling this again for a contract_id with zero remaining
    vectors (e.g. a retried deletion after an earlier partial failure --
    see routes/contracts.py's DELETE endpoint for the full ordering
    reasoning) is a safe no-op that returns 0 -- Chroma's delete(where=...)
    simply matches nothing, the same way deleting an already-deleted
    Postgres row or an already-removed file is harmless to repeat.

    NOTE -- distinct from M34's own tracked stale-vector gap: this
    function is invoked ONLY for an ACTIVE, explicit contract deletion
    (DELETE /contracts/{id}). It does not touch, and is not a fix for,
    index_contract_clauses() itself never purging OLD vectors before a
    pipeline RE-RUN persists fresh Clause rows with fresh clause_id UUIDs
    (agent/suggest_negotiation.py's own KNOWN GAP docstring) -- that is a
    separate, still-open, still-tracked issue this milestone does not
    touch.
    """
    collection = _get_chroma_collection()
    existing = collection.get(where={"contract_id": str(contract_id)}, include=[])
    existing_count = len(existing["ids"])
    if existing_count == 0:
        return 0
    collection.delete(where={"contract_id": str(contract_id)})
    return existing_count


def index_contract_clauses(contract_id: uuid.UUID, db_session) -> int:
    """Embed every persisted Clause for contract_id and upsert it into
    the Chroma "clauses" collection, keyed by clause_id (so re-running
    this for the same contract updates existing entries instead of
    duplicating them).

    Returns the number of clauses indexed.
    """
    contract = db_session.query(Contract).filter(Contract.id == contract_id).one_or_none()
    if contract is None:
        raise ValueError(f"No contract found for contract_id={contract_id}")

    clauses = (
        db_session.query(Clause)
        .filter(Clause.contract_id == contract_id)
        .order_by(Clause.page_number)
        .all()
    )
    if not clauses:
        logger.info("Contract %s has no clauses to index.", contract_id)
        return 0

    total_tokens = sum(estimate_tokens(c.text) for c in clauses)
    estimated_seconds = max(0, len(clauses) - 1) * MIN_SECONDS_BETWEEN_REQUESTS
    logger.info(
        "Indexing contract %s: %d clauses, ~%d estimated tokens. At "
        "Voyage's free-tier pacing (%d req/min), this will take "
        "approximately %ds (~%.1f min).",
        contract_id, len(clauses), total_tokens,
        REQUESTS_PER_MINUTE, estimated_seconds, estimated_seconds / 60,
    )
    if total_tokens > TOKENS_PER_MINUTE:
        logger.warning(
            "Contract %s's total clause text (~%d tokens) exceeds the "
            "%d tokens/min free-tier cap on its own; the token-pacing "
            "check in rate_limit.pace() will add waits beyond the "
            "request-count floor as needed.",
            contract_id, total_tokens, TOKENS_PER_MINUTE,
        )

    collection = _get_chroma_collection()
    history: deque[tuple[float, int]] = deque()
    indexed_count = 0

    for i, clause in enumerate(clauses):
        clause_tokens = estimate_tokens(clause.text)
        pace(history, clause_tokens, context=f"embedding clause {i + 1}")

        vector = embed_with_retry(clause.text, context=f"embedding clause {i + 1}/{len(clauses)}")
        history.append((time.monotonic(), clause_tokens))

        collection.upsert(
            ids=[str(clause.id)],
            embeddings=[vector],
            metadatas=[{
                "clause_id": str(clause.id),
                "contract_id": str(contract_id),
                "user_id": str(contract.user_id),
                "embedding_model_version": MODEL,
                "heading_path": clause.heading_path,
                "category": clause.category or "",
            }],
            documents=[clause.text],
        )
        indexed_count += 1
        logger.info("embedded %d/%d clauses...", indexed_count, len(clauses))

    return indexed_count
