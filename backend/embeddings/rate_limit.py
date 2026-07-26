"""Shared Voyage AI free-tier pacing/retry logic.

ORIGIN: this was M10's own private implementation, living entirely
inside embeddings/index.py (_pace(), _embed_with_retry(),
_estimate_tokens(), and the 5 constants below). Extracted here, verbatim
in algorithm, so a SECOND real call site --
classification/centroid_fallback.py's M23 stage-2 fallback -- can reuse
the exact same, already-verified protection instead of a second,
independent reimplementation of the same real constraint on the same
Voyage account.

WHY THIS EXTRACTION WAS NEEDED (a real, reproduced bug, not a
hypothetical): classify_by_centroid() (M23) calls embed_text() for every
clause whose heading doesn't match M8's lexicon, during
ingestion/chunker.py's persist_chunks() -- BEFORE embeddings/index.py's
own index_contract_clauses() stage ever runs. That call was previously
bare (no pacing, no retry), even though it hits the IDENTICAL Voyage
account and the IDENTICAL 3 RPM/10K TPM free-tier limit
index_contract_clauses() has always correctly respected. A contract with
several unclassifiable headings could fire multiple rapid, unpaced
Voyage calls during persistence alone, trip the real rate limit, and
crash the ENTIRE pipeline with an unhandled VoyageEmbeddingError --
reproduced live: calls 1-3 succeeded, call 4 raised
voyageai.error.RateLimitError uncaught, exactly matching a real
production failure (contract CN_Sheet_Final.pdf).

REUSE, NOT DUPLICATION: embeddings/index.py's own index_contract_clauses()
now imports pace()/embed_with_retry()/estimate_tokens() from here instead
of defining its own copies -- its own behavior (constants, algorithm,
retry count, log content) is UNCHANGED; only the source of the code
moved. See that module's own comments at its (now much shorter)
pacing-related call site.

GENERALIZED SIGNATURE, ONE DELIBERATE DIFFERENCE FROM THE ORIGINAL: the
original _pace()/_embed_with_retry() took a `clause_number`/`total` pair
purely to phrase their own log messages ("...before embedding clause
%d.", "...clause %d/%d..."). That framing doesn't fit centroid_fallback.py's
own calling pattern (many independent calls across a contract's
clauses -- and across different contracts within the same long-lived
worker process -- with no natural "clause N of M" to report). Replaced
with a single free-text `context` string each caller supplies, so both
call sites still get an accurate, specific log line, without this shared
module needing to know anything about "clauses" as a concept at all.
"""

import logging
import time
from collections import deque

import tiktoken
import voyageai.error

from embeddings.voyage_client import VoyageEmbeddingError, embed_text

logger = logging.getLogger("clauseguard.embeddings.rate_limit")

# Voyage's real, confirmed free-tier limits for this account (voyage-3-lite,
# no payment method on file) -- see embeddings/index.py's own module
# docstring (M9/M10) for the original confirmation via real API calls.
REQUESTS_PER_MINUTE = 3
TOKENS_PER_MINUTE = 10_000
MIN_SECONDS_BETWEEN_REQUESTS = 21  # 60/3 = 20s exactly; +1s safety margin
RATE_LIMIT_RETRY_WAIT_SECONDS = 65  # just over the 60s window
MAX_RATE_LIMIT_RETRIES = 2

_encoding = tiktoken.get_encoding("cl100k_base")


def estimate_tokens(text: str) -> int:
    return len(_encoding.encode(text))


def pace(history: deque, next_tokens: int, context: str) -> None:
    """Block, if needed, before the next embedding request so we stay
    within BOTH the 3 req/min and 10K tokens/min free-tier limits.
    Identical algorithm to the original _pace() -- see embeddings/index.py's
    git history / this module's own docstring for that origin.

    `history`: a deque of (timestamp, token_count) tuples the CALLER owns
    and passes in on every call -- deliberately not global state inside
    this shared module, so two independent callers (index_contract_clauses()'s
    own per-call-scoped history, and centroid_fallback.py's own
    persistent-across-calls history) never interfere with each other's
    pacing windows despite sharing this same function.

    Request-count pacing: a fixed floor between consecutive requests
    (MIN_SECONDS_BETWEEN_REQUESTS, ~21s). Any two consecutive calls at
    least ~20s apart means any 4 consecutive calls span > 60s, so this
    alone is sufficient to guarantee <= 3 requests per rolling 60s
    window without needing to explicitly count a sliding window.

    Token pacing: tracks (timestamp, token_count) for calls still inside
    the trailing 60s window and, if adding this request's tokens would
    exceed TOKENS_PER_MINUTE, waits for the window's oldest entry to age
    out.

    `context`: a short, human-readable phrase describing what's about to
    be embedded (e.g. "embedding clause 3", "stage-2 centroid
    classification") -- used only for the log line below.
    """
    now = time.monotonic()
    while history and now - history[0][0] >= 60:
        history.popleft()

    floor_wait = 0.0
    if history:
        floor_wait = MIN_SECONDS_BETWEEN_REQUESTS - (now - history[-1][0])

    token_wait = 0.0
    if history:
        window_tokens = sum(tokens for _, tokens in history)
        if window_tokens + next_tokens > TOKENS_PER_MINUTE:
            token_wait = 60 - (now - history[0][0])

    wait = max(floor_wait, token_wait, 0.0)
    if wait > 0:
        reason = "token-budget" if token_wait > floor_wait else "request-pacing"
        logger.info(
            "Pacing (%s): waiting %.1fs before %s.",
            reason, wait, context,
        )
        time.sleep(wait)


def embed_with_retry(text: str, context: str) -> list[float]:
    """Call embed_text(), retrying up to MAX_RATE_LIMIT_RETRIES times if
    the failure is specifically a Voyage RateLimitError (detected via the
    original exception preserved on VoyageEmbeddingError.__cause__, not
    string-matching). Any other error (auth, malformed input, network)
    propagates immediately -- only a rate limit is worth waiting out.
    Identical algorithm to the original _embed_with_retry().

    `context`: see pace()'s own docstring -- same purpose, used only for
    the log line on a retry.
    """
    attempt = 0
    while True:
        try:
            return embed_text(text)
        except VoyageEmbeddingError as exc:
            is_rate_limit = isinstance(exc.__cause__, voyageai.error.RateLimitError)
            if is_rate_limit and attempt < MAX_RATE_LIMIT_RETRIES:
                attempt += 1
                logger.warning(
                    "Rate limit hit on %s (retry %d/%d): %s "
                    "Waiting %ds before retrying.",
                    context, attempt, MAX_RATE_LIMIT_RETRIES,
                    exc, RATE_LIMIT_RETRY_WAIT_SECONDS,
                )
                time.sleep(RATE_LIMIT_RETRY_WAIT_SECONDS)
                continue
            raise
