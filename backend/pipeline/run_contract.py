"""M13: the full per-contract pipeline, synchronous and single-process
(Celery/async is M15). extract -> chunk -> persist -> classify ->
embed/index -> flag every persisted clause via Groq -> persist
FlaggedClause rows.

M28 ADDITION: each major stage below (extraction, chunking, persistence,
embedding/indexing, flagging) plus the overall end-to-end pipeline
duration is now wrapped in observability/timing.py's timed_stage(), a
real, persistent, always-on per-call latency measurement -- not a
one-off benchmark, an accumulating record from real runs, comparable
against the architecture doc's ~2.5s/5s end-to-end target. To get
per-stage granularity, this module now calls ingestion.chunker's four
already-independently-callable sub-steps (extract_text, chunk_document,
persist_chunks) plus embeddings.index's index_contract_clauses directly,
INSTEAD OF going through chunker.process_contract()'s single
convenience-wrapper call -- chunker.py's own docstring explicitly
sanctions this ("each step remains independently callable and
independently testable on its own"), so this is not a restructuring of
process_contract() itself (still there, still unmodified, still usable
by anything that wants the one-call convenience form) -- it's simply
this caller choosing to compose the same real steps directly, which is
the only way to get real per-stage timing without modifying chunker.py
at all.

Two independent rate limits are paced separately, since they're
different providers with different confirmed numbers:
  - Voyage embeddings: paced inside index_contract_clauses() itself (M10),
    confirmed 3 RPM.
  - Flagging LLM: paced in THIS module (see below).

PROVIDER MIGRATION (Gemini -> Groq, 2026-07-09): this module's flagging
call was originally against Gemini (gemini-2.5-flash-lite, itself a
switch from an even more restrictive gemini-3.5-flash -- see git
history / agent/flag_clause.py's own docstring). Gemini's confirmed
free-tier daily cap (GenerateRequestsPerDayPerProjectPerModel-FreeTier,
20 requests/day/model) repeatedly caused real, multi-hour lockouts --
most recently reproduced live on 2026-07-09 (a single 4-clause contract
exhausting the entire daily budget on its own). Migrated to Groq
(llama-3.3-70b-versatile) -- confirmed via Groq's own live docs
(console.groq.com/docs/rate-limits) at 30 RPM / 1,000 RPD / 12K TPM /
100K TPD on the free tier: dramatically more headroom than Gemini's 20
RPD, though explicitly NOT an uncapped quota (see _retry_reason()'s own
docstring below for how a persistent, non-transient rate limit is still
handled without assuming Groq can never lock out a run).

These two stages run one fully after the other (not interleaved
per-clause) -- extraction/chunking/persistence/embedding-indexing finish
completely before this module's flagging loop starts on a single clause,
since embedding does not depend on flagging or vice versa, and
interleaving them would only complicate the pacing logic for no benefit.

FIXED (post-Phase-2 hardening, tracked in docs/M1_Validation_Results.md):
M27's independent review found _is_rate_limit_error() (the old name)
only matched 429 rate-limit errors, not 5xx server errors -- a real
Gemini 503 was observed live, unretried, multiple times during M27/M28's
own testing. _retry_reason() below retries on BOTH a genuine rate limit
and a genuine 503 service-unavailable, using the exact same backoff
constants for either -- see that function's own docstring for why this
is deliberately narrow (503 specifically, not "any 5xx"), a principle
preserved unchanged across the Gemini -> Groq migration (only the
concrete exception TYPES checked changed -- see _retry_reason()'s own
docstring for Groq's real, confirmed exception hierarchy:
groq.RateLimitError always carries status_code 429;
groq.InternalServerError is Groq's SDK's generic catch-all for any
status_code >= 500, so distinguishing "503 specifically" now requires
checking exc.status_code directly rather than a dedicated exception
class, confirmed via groq._client.Groq._make_status_error's own real
source).
"""

import logging
import time
import uuid

import groq

from agent.flag_clause import FlagClauseError, flag_clause
from agent.suggest_negotiation import (
    NegotiationNeedsManualReviewError,
    SuggestNegotiationError,
    get_related_clauses_for_grounding,
    suggest_negotiation_point,
)
from agent.tools import FlagClauseOutput
from agent.validate import NeedsManualReviewError
from embeddings.index import index_contract_clauses
from embeddings.voyage_client import VoyageEmbeddingError
from ingestion.chunker import chunk_document, persist_chunks
from ingestion.extract import extract_text_with_timeout
from models.clause import Clause
from models.contract import Contract
from models.flagged_clause import FlaggedClause
from observability.audit import (
    EVENT_API_ERROR_FAILED,
    EVENT_API_ERROR_RETRY,
    EVENT_FINAL_FLAG_DECISION,
    EVENT_GROUNDING_ERROR,
    EVENT_GROUNDING_FAILURE,
    write_audit_entry,
)
from observability.timing import timed_stage

logger = logging.getLogger("clauseguard.pipeline.run_contract")


def _set_current_stage(db_session, contract: Contract, stage: str) -> None:
    """M40: stamps Contract.current_stage and commits IMMEDIATELY, as its
    own small, standalone UPDATE -- deliberately separate from whatever
    that stage's own real work/commit does later, so a polling client
    (GET /contracts/{id}/status) genuinely observes this change the
    MOMENT a stage begins, not only after it (or some later stage)
    finishes. One column, one already-loaded row: negligible next to the
    real stage work that follows (extraction, a live Groq/Voyage call,
    etc.) -- confirmed via this milestone's own testing (see that
    testing step's timing comparison).
    """
    contract.current_stage = stage
    db_session.commit()

# Groq free-tier limits for llama-3.3-70b-versatile -- CONFIRMED via
# Groq's own live docs (console.groq.com/docs/rate-limits, checked
# 2026-07-09): 30 RPM / 1,000 RPD / 12K TPM / 100K TPD. Dramatically more
# generous than the Gemini models this pipeline previously ran on
# (gemini-3.5-flash: 5 RPM / 20 RPD; gemini-2.5-flash-lite: 20 RPD --
# both repeatedly exhausted by ordinary testing/use), but NOT literally
# uncapped -- pacing below stays conservative rather than assuming a
# provider number can never bind again.
GROQ_REQUESTS_PER_MINUTE = 30  # llama-3.3-70b-versatile, confirmed via Groq's own docs
MIN_SECONDS_BETWEEN_FLAG_CALLS = 3  # deliberately more conservative than the bare 60/30=2s
# Shared by BOTH retryable conditions (429 rate-limit AND 503 service-
# unavailable, see _retry_reason() below) -- the task here (wait, then
# try again) is the same for either, and this project has no separate
# real-observed retry-delay figure for 503 the way it does for 429, so
# reusing the same, already-confirmed-safe wait/retry-count pair is the
# honest choice over inventing a second set of numbers with no evidence
# behind them.
RATE_LIMIT_RETRY_WAIT_SECONDS = 20  # comfortably clears Groq's 60s per-minute window
MAX_RATE_LIMIT_RETRIES = 2

# Groq's RateLimitError (429) covers BOTH a per-minute limit (RPM/TPM,
# clears within ~60s -- worth retrying) AND a per-day limit (RPD/TPD,
# cannot possibly clear within this same pipeline run -- retrying wastes
# MAX_RATE_LIMIT_RETRIES * RATE_LIMIT_RETRY_WAIT_SECONDS per clause on a
# call guaranteed to fail again). Unlike Gemini, Groq's 429 response body
# does not carry a distinguishable quotaId-style field for which kind was
# hit (confirmed via Groq's own rate-limits docs, which describe the
# response headers but not a body-level per-day-vs-per-minute marker) --
# so this project falls back to the one real, provider-supplied signal
# that DOES distinguish them: the `retry-after` header's magnitude. A
# genuine per-minute hit reports a retry-after well under 60s; a
# per-day/token-budget hit reports something on the order of hours.
# GROQ_LONG_RETRY_AFTER_THRESHOLD_SECONDS is set well above any possible
# per-minute retry-after (which cannot exceed ~60s by definition) and
# well below a same-day reset, so a genuine per-minute hit is never
# misclassified as long-lived. This heuristic is NOT empirically
# reproduced against a real exhausted Groq daily quota (doing so would
# mean deliberately burning a full day's 1,000-request budget) -- it is
# a reasoned extrapolation from Groq's own documented header semantics,
# not a guess made up from nothing, but should be treated as less
# rock-solid than the rest of this module's other confirmed numbers.
GROQ_LONG_RETRY_AFTER_THRESHOLD_SECONDS = 90

# Sentinel severity for a clause whose flag_clause() call failed even
# after retries. A visible, queryable placeholder row -- chosen over
# silently omitting the clause from flagged_clauses entirely, since a
# clearly-marked failure is safer than a clause that just looks like it
# was reviewed and found unremarkable.
FLAGGING_FAILED_SEVERITY = "flagging_failed"

# M33: a SEPARATE sentinel, deliberately distinct from FLAGGING_FAILED_SEVERITY
# above -- see agent/validate.py's NeedsManualReviewError docstring for the
# full reasoning. flagging_failed means an actual API/infrastructure
# failure; needs_manual_review means Groq responded successfully but its
# own tool-call output failed schema validation twice in a row (the
# original attempt and its one retry) -- a model-output-quality problem,
# not an outage, and worth distinguishing in the persisted row so a human
# reviewing flagged_clauses can tell the two apart.
NEEDS_MANUAL_REVIEW_SEVERITY = "needs_manual_review"
# A sentinel object (not a plain string or None) so _flag_with_retry()'s
# return value can distinguish FOUR outcomes -- success (a real
# FlagClauseOutput), a hard API/infra failure (None, unchanged from
# M13), this needs_manual_review case, and this new long-rate-limit
# case -- without any of the four being confusable with each other.
_NEEDS_MANUAL_REVIEW = object()

# Distinct from _NEEDS_MANUAL_REVIEW and from a bare None failure: this
# sentinel additionally tells run_contract_pipeline's own flagging loop to
# stop calling Groq entirely for the REST of this run (see that loop's
# own handling below) -- a bare None failure (e.g. a genuine non-retryable
# 4xx, or a 429/503 that exhausted its retries) carries no such run-wide
# implication and must not be confused with this one.
_LONG_RATE_LIMIT_EXHAUSTED = object()


def _retry_after_seconds(exc: groq.RateLimitError) -> float | None:
    """Extract the real `retry-after` header (seconds) from a Groq
    RateLimitError's response, if present -- confirmed via Groq's own
    rate-limits docs (console.groq.com/docs/rate-limits, checked
    2026-07-09): "retry-after is only set if you hit the rate limit and
    status code 429 is returned." Used by _retry_reason() below to tell
    a genuine per-minute rate limit (retry-after well under 60s) apart
    from a per-day/token-budget exhaustion (retry-after on the order of
    hours) -- see GROQ_LONG_RETRY_AFTER_THRESHOLD_SECONDS's own comment
    for why this is the chosen signal and its confirmed-vs-reasoned
    caveat.

    Returns None if the header is absent or not a valid number -- callers
    must treat that as "can't tell", not "safe to assume short."
    """
    response = getattr(exc, "response", None)
    header_value = response.headers.get("retry-after") if response is not None else None
    if header_value is None:
        return None
    try:
        return float(header_value)
    except ValueError:
        return None


def _retry_reason(exc: FlagClauseError) -> str | None:
    """Returns a short, specific reason string if exc's real underlying
    cause (exc.__cause__, set via `raise FlagClauseError(...) from exc`
    in agent/flag_clause.py) is a transient, worth-retrying Groq API
    error, else None.

    Two, and only two, conditions retry here, deliberately narrow. NOTE:
    by the time this function is called (from _flag_with_retry()'s own
    except block), a groq.RateLimitError with a LONG retry-after (a
    likely per-day/RPD/TPD exhaustion) has ALREADY been intercepted and
    returned early, one level up -- see _flag_with_retry()'s own
    docstring for why that case needs to signal more than "retry or
    don't" (it needs to stop the rest of this run's clauses too, not just
    skip retrying this one). So any groq.RateLimitError THIS function
    sees is, by construction, one _flag_with_retry() has already decided
    is short-lived enough to be worth retrying:
      - groq.RateLimitError (always status_code 429) -- the original,
        always-retried condition (with the long-retry-after case already
        filtered out before reaching here).
      - groq.InternalServerError (Groq's generic catch-all for any
        status_code >= 500 -- confirmed via groq._client.Groq's own
        _make_status_error source, which has no separate 503-specific
        exception class) whose exc.status_code == 503 specifically --
        the same post-Phase-2 M27/M28 principle preserved across the
        Gemini -> Groq migration: a real 503 was observed live,
        unretried, during that milestone's own testing, and this project
        still does not retry "any 5xx", only a confirmed 503.

    Deliberately NOT "any RateLimitError regardless of retry-after" or
    "any InternalServerError regardless of status_code": a genuine,
    long-lived exhaustion or a 500/502/504 this project has never
    actually observed must not be silently retried as if it were
    transient -- retrying a real, non-transient failure wastes the
    MIN_SECONDS_BETWEEN_FLAG_CALLS-paced retry budget and delays surfacing
    a real problem, which is exactly the "retry everything" failure mode
    this function's narrowness exists to avoid. If a different real 5xx
    code is ever observed in practice, it should be added here the same
    way 503 was -- confirmed first, not guessed.

    Returns a reason string (not just True/False) so the caller can log
    an accurate message -- calling a 503 a "rate limit" would itself be a
    real inaccuracy, not just a style nitpick, given this project's own
    standing discipline against comments/logs that say something false.
    """
    cause = exc.__cause__
    if isinstance(cause, groq.RateLimitError):
        return "rate limit (429)"
    if isinstance(cause, groq.InternalServerError) and getattr(cause, "status_code", None) == 503:
        return "service unavailable (503)"
    return None


def _flag_with_retry(
    clause_text: str,
    clause_number: int,
    total: int,
    contract_id: uuid.UUID,
    clause_id: uuid.UUID,
) -> FlagClauseOutput | None | object:
    """Call flag_clause(), retrying only on a genuine transient hit (see
    _retry_reason() above -- a 429 rate limit or a 503 service-
    unavailable, nothing else). This is the ONLY retry loop this
    function runs itself; M33's separate validation-retry (exactly one
    attempt, entirely inside flag_clause() -> agent/validate.py) has
    already happened by the time flag_clause() returns or raises here --
    the two never share state or double-count each other's attempts
    (see agent/validate.py's own docstring, COMPOSITION section).

    Returns (never raises, for any failure -- a single clause failing
    must never take down the rest of the pipeline):
      - a real FlagClauseOutput on success
      - None for a hard API/infrastructure failure (a transient error
        that exhausted its retries, or any other FlagClauseError) --
        unchanged from M13
      - the _NEEDS_MANUAL_REVIEW sentinel (M33) if Groq's tool call
        output failed schema validation on both attempts inside
        flag_clause() -- a categorically different failure mode from
        the two above, so a distinct return value rather than also
        being folded into None.
      - the _LONG_RATE_LIMIT_EXHAUSTED sentinel if the 429's retry-after
        is at/above GROQ_LONG_RETRY_AFTER_THRESHOLD_SECONDS (see
        _retry_after_seconds()'s own docstring) -- a likely per-day/
        token-budget exhaustion, not a per-minute hit. Returned
        immediately, with ZERO retries: that kind of limit cannot clear
        within this same run, so retrying it can only reproduce the
        identical failure MAX_RATE_LIMIT_RETRIES more times for nothing.
        See run_contract_pipeline's own handling of this sentinel for how
        it also short-circuits the REST of this run's remaining clauses.

    The caller is responsible for recording a visible marker for either
    failure return value.

    M35: contract_id/clause_id are forwarded to flag_clause() (which
    itself only forwards them again, to agent/validate.py -- see that
    module's own docstring) so validation events can be audited with
    real IDs. This function writes its OWN audit entries for the two
    events squarely in its own scope -- an api_error_retry firing, and
    the terminal api_error_failed state -- since M35's spec assigns
    "retries" for THIS specific 429/503 axis to run_contract.py, not
    agent/validate.py (which handles the separate schema-validation
    retry axis).
    """
    attempt = 0
    while True:
        try:
            return flag_clause(clause_text, contract_id=contract_id, clause_id=clause_id)
        except NeedsManualReviewError as exc:
            logger.error(
                "flag_clause needs manual review for clause %d/%d -- "
                "Groq's tool call output failed schema validation on "
                "both the original attempt and its one retry: %s",
                clause_number, total, exc,
            )
            return _NEEDS_MANUAL_REVIEW
        except FlagClauseError as exc:
            cause = exc.__cause__
            if isinstance(cause, groq.RateLimitError):
                retry_after = _retry_after_seconds(cause)
                if retry_after is not None and retry_after >= GROQ_LONG_RETRY_AFTER_THRESHOLD_SECONDS:
                    # A long retry-after strongly suggests a per-day/
                    # token-budget exhaustion, not the ordinary per-minute
                    # limit -- see _retry_after_seconds()'s and
                    # GROQ_LONG_RETRY_AFTER_THRESHOLD_SECONDS's own
                    # docstrings for why this signal (not a guess) is used
                    # and its confirmed-vs-reasoned caveat.
                    logger.error(
                        "Groq rate limit with a long retry-after (%.0fs, "
                        ">= %ds threshold) flagging clause %d/%d -- NOT "
                        "retrying (likely a per-day/token-budget "
                        "exhaustion that cannot clear within this "
                        "pipeline run): %s",
                        retry_after, GROQ_LONG_RETRY_AFTER_THRESHOLD_SECONDS,
                        clause_number, total, exc,
                    )
                    write_audit_entry(
                        contract_id=contract_id,
                        clause_id=clause_id,
                        event_type=EVENT_API_ERROR_FAILED,
                        details=(
                            f"clause {clause_number}/{total} flagging failed "
                            f"(Groq rate limit with long retry-after "
                            f"{retry_after:.0f}s, not retried): {exc}"
                        ),
                    )
                    return _LONG_RATE_LIMIT_EXHAUSTED
            reason = _retry_reason(exc)
            if reason is not None and attempt < MAX_RATE_LIMIT_RETRIES:
                attempt += 1
                logger.warning(
                    "Groq %s hit flagging clause %d/%d (retry %d/%d): %s "
                    "Waiting %ds before retrying.",
                    reason, clause_number, total, attempt, MAX_RATE_LIMIT_RETRIES,
                    exc, RATE_LIMIT_RETRY_WAIT_SECONDS,
                )
                write_audit_entry(
                    contract_id=contract_id,
                    clause_id=clause_id,
                    event_type=EVENT_API_ERROR_RETRY,
                    details=(
                        f"clause {clause_number}/{total}: {reason}, retry "
                        f"{attempt}/{MAX_RATE_LIMIT_RETRIES}: {exc}"
                    ),
                )
                time.sleep(RATE_LIMIT_RETRY_WAIT_SECONDS)
                continue
            logger.error(
                "flag_clause failed for clause %d/%d (not retried -- %s): %s",
                clause_number, total,
                "retries exhausted" if reason is not None else "non-retryable error",
                exc,
            )
            write_audit_entry(
                contract_id=contract_id,
                clause_id=clause_id,
                event_type=EVENT_API_ERROR_FAILED,
                details=(
                    f"clause {clause_number}/{total} flagging failed "
                    f"({'retries exhausted' if reason is not None else 'non-retryable error'}): {exc}"
                ),
            )
            return None


def run_contract_pipeline(contract_id: uuid.UUID, db_session) -> dict:
    """Run the full pipeline for one contract and return a summary dict:
    {"clauses_persisted", "clauses_indexed", "clauses_flagged", "clauses_failed"}.

    Re-run safety: re-running this for the same contract_id replaces its
    flagged_clauses rows AND its clauses rows, both without accumulating
    duplicates.

    FIXED (post-Phase-2 hardening, tracked in docs/M1_Validation_Results.md):
    this used to delete+insert flagged_clauses only at the very end of a
    full run -- too late to protect persist_chunks()'s own delete of the
    OLD clauses rows earlier in the same run, since Postgres won't let you
    delete a clauses row a flagged_clauses row still references
    (flagged_clauses_clause_id_fkey). Re-running this pipeline on a
    contract that already had flagged_clauses rows from a prior run
    reliably crashed persist_chunks() with a real
    psycopg2.errors.ForeignKeyViolation -- reproduced live multiple times
    (M28's own testing, and again during M28's independent review).

    Fix: the old flagged_clauses rows for this contract_id are now deleted
    BEFORE persist_chunks() runs, on this SAME db_session, with NO commit
    in between -- so that delete rides inside persist_chunks()'s own
    delete-then-insert-then-commit transaction rather than being a
    separate, earlier commit of its own. This preserves the exact same
    all-or-nothing guarantee persist_chunks() already had: if anything
    fails before ITS commit (a bad chunk, a dropped connection), the
    rollback undoes the flagged_clauses delete too, leaving the contract
    with whatever clauses AND flagged_clauses rows it had before this call
    -- never a partial mix. The new flagged_clauses rows are still only
    ever built AFTER the flagging loop below (they need the freshly
    persisted clauses' own fresh clause_id values, which don't exist until
    persist_chunks() has already run) -- only the OLD rows' cleanup moved
    earlier, not the new rows' insert. Confirmed nothing between the old
    cleanup point and this new one ever reads flagged_clauses for this
    contract_id (extraction/chunking/persistence/embedding-indexing/the
    flagging loop itself all only read/write clauses, never
    flagged_clauses, until the final insert below).
    """
    contract = db_session.query(Contract).filter(Contract.id == contract_id).one_or_none()
    if contract is None:
        raise ValueError(f"No contract found for contract_id={contract_id}")

    logger.info("Pipeline starting for contract %s (%s).", contract_id, contract.filename)
    contract_id_str = str(contract_id)

    with timed_stage("end_to_end", contract_id=contract_id_str):
        _set_current_stage(db_session, contract, "extracting")
        with timed_stage("extraction", contract_id=contract_id_str):
            # M32: extract_text_with_timeout() (not the bare extract_text())
            # -- a real, hard, process-level timeout around parsing, since
            # M32's upload validation happens before this point and cannot
            # by itself catch a file that passes size/MIME checks but is
            # pathologically structured to hang or exhaust memory during
            # actual parsing. Raises ExtractionTimeoutError on a real
            # timeout, which propagates up uncaught (same as any other
            # extraction failure) to worker/tasks.py's own existing
            # exception handling, which already marks the contract
            # "failed" -- no change needed there.
            pages = extract_text_with_timeout(contract.storage_path)

        _set_current_stage(db_session, contract, "chunking")
        with timed_stage("chunking", contract_id=contract_id_str):
            chunks = chunk_document(pages)

        _set_current_stage(db_session, contract, "persisting")
        with timed_stage("persistence", contract_id=contract_id_str):
            # Old flagged_clauses rows deleted here, BEFORE persist_chunks()
            # -- same session, NOT committed yet, so this delete becomes
            # part of persist_chunks()'s own atomic transaction (see this
            # function's own docstring above for why this ordering, not a
            # separate earlier commit, is what actually fixes the FK bug
            # without weakening the original atomicity guarantee).
            db_session.query(FlaggedClause).filter(FlaggedClause.contract_id == contract_id).delete()
            persist_chunks(contract_id, chunks, db_session)

        _set_current_stage(db_session, contract, "embedding")
        with timed_stage("embedding_indexing", contract_id=contract_id_str):
            indexed_count = index_contract_clauses(contract_id, db_session)

        logger.info(
            "Extraction, chunking, persistence, and embedding/indexing complete "
            "for contract %s: %d clauses indexed.",
            contract_id, indexed_count,
        )

        clauses = (
            db_session.query(Clause)
            .filter(Clause.contract_id == contract_id)
            .order_by(Clause.page_number)
            .all()
        )
        total = len(clauses)
        logger.info(
            "Flagging %d clauses for contract %s via Groq (pacing ~%ds/request, "
            "~%d RPM free-tier budget for %s).",
            total, contract_id, MIN_SECONDS_BETWEEN_FLAG_CALLS, GROQ_REQUESTS_PER_MINUTE,
            "llama-3.3-70b-versatile",
        )

        flagged_rows: list[FlaggedClause] = []
        failed_count = 0
        last_call_time: float | None = None
        # Set True the first time _flag_with_retry() returns
        # _LONG_RATE_LIMIT_EXHAUSTED (a rate limit with a long retry-after,
        # see that sentinel's own comment) -- a likely per-day/token-budget
        # exhaustion cannot reset before this same run finishes, so every
        # remaining clause is skipped below without spending a single
        # further Groq call on a request that is guaranteed to fail
        # identically.
        long_rate_limit_exhausted = False

        _set_current_stage(db_session, contract, "flagging")
        with timed_stage("flagging", contract_id=contract_id_str, n_clauses=total):
            for i, clause in enumerate(clauses, start=1):
                if long_rate_limit_exhausted:
                    failed_count += 1
                    flagged_rows.append(
                        FlaggedClause(
                            clause_id=clause.id,
                            contract_id=contract_id,
                            severity=FLAGGING_FAILED_SEVERITY,
                            explanation=(
                                "Automated flagging skipped for this clause: "
                                "Groq's rate limit was already confirmed "
                                "exhausted (long retry-after) earlier in "
                                "this same pipeline run; no risk assessment "
                                "was produced. See server logs for the "
                                "originating clause's real retry-after value."
                            ),
                            citation=clause.text,
                        )
                    )
                    write_audit_entry(
                        contract_id=contract_id,
                        clause_id=clause.id,
                        event_type=EVENT_FINAL_FLAG_DECISION,
                        details=(
                            f"severity={FLAGGING_FAILED_SEVERITY}: skipped, "
                            "Groq rate limit (long retry-after) already "
                            "confirmed exhausted earlier this run"
                        ),
                    )
                    logger.info(
                        "flagged %d/%d clauses (skipped -- long rate limit "
                        "already confirmed exhausted this run)...",
                        i, total,
                    )
                    continue

                if last_call_time is not None:
                    wait = MIN_SECONDS_BETWEEN_FLAG_CALLS - (time.monotonic() - last_call_time)
                    if wait > 0:
                        time.sleep(wait)

                result = _flag_with_retry(clause.text, i, total, contract_id, clause.id)
                last_call_time = time.monotonic()

                if result is _LONG_RATE_LIMIT_EXHAUSTED:
                    long_rate_limit_exhausted = True
                    failed_count += 1
                    flagged_rows.append(
                        FlaggedClause(
                            clause_id=clause.id,
                            contract_id=contract_id,
                            severity=FLAGGING_FAILED_SEVERITY,
                            explanation=(
                                "Automated flagging failed: Groq's rate "
                                "limit is exhausted with a long retry-after "
                                "(likely a per-day/token-budget "
                                "exhaustion); no risk assessment was "
                                "produced. This clause and any remaining "
                                "clauses in this run will not be retried "
                                "until the limit clears."
                            ),
                            citation=clause.text,
                        )
                    )
                    write_audit_entry(
                        contract_id=contract_id,
                        clause_id=clause.id,
                        event_type=EVENT_FINAL_FLAG_DECISION,
                        details=(
                            f"severity={FLAGGING_FAILED_SEVERITY}: Groq "
                            "rate limit (long retry-after) exhausted, not "
                            "retried (remaining clauses in this run will "
                            "be skipped)"
                        ),
                    )
                elif result is None:
                    failed_count += 1
                    flagged_rows.append(
                        FlaggedClause(
                            clause_id=clause.id,
                            contract_id=contract_id,
                            severity=FLAGGING_FAILED_SEVERITY,
                            explanation=(
                                "Automated flagging failed for this clause after "
                                "retries; no risk assessment was produced. See "
                                "server logs for the underlying error."
                            ),
                            citation=clause.text,
                        )
                    )
                    write_audit_entry(
                        contract_id=contract_id,
                        clause_id=clause.id,
                        event_type=EVENT_FINAL_FLAG_DECISION,
                        details=f"severity={FLAGGING_FAILED_SEVERITY}: automated flagging failed after retries",
                    )
                elif result is _NEEDS_MANUAL_REVIEW:
                    # M33: distinct from the flagging_failed branch above --
                    # see NEEDS_MANUAL_REVIEW_SEVERITY's own comment for why.
                    failed_count += 1
                    flagged_rows.append(
                        FlaggedClause(
                            clause_id=clause.id,
                            contract_id=contract_id,
                            severity=NEEDS_MANUAL_REVIEW_SEVERITY,
                            explanation=(
                                "Groq's flag_clause tool call output failed "
                                "schema validation on both the original attempt "
                                "and its one retry; no risk assessment was "
                                "produced. This clause needs manual review. See "
                                "server logs for both attempts' raw output and "
                                "validation errors."
                            ),
                            citation=clause.text,
                        )
                    )
                    write_audit_entry(
                        contract_id=contract_id,
                        clause_id=clause.id,
                        event_type=EVENT_FINAL_FLAG_DECISION,
                        details=f"severity={NEEDS_MANUAL_REVIEW_SEVERITY}: schema validation failed on both attempts",
                    )
                else:
                    flagged_rows.append(
                        FlaggedClause(
                            clause_id=clause.id,
                            contract_id=contract_id,
                            severity=result.severity,
                            explanation=result.explanation,
                            citation=result.citation,
                        )
                    )
                    write_audit_entry(
                        contract_id=contract_id,
                        clause_id=clause.id,
                        event_type=EVENT_FINAL_FLAG_DECISION,
                        details=f"severity={result.severity}: {result.explanation}",
                    )
                    # M34: suggest_negotiation_point() -- ONLY for
                    # high/medium severity, deliberately NOT every flagged
                    # clause. This is a SECOND real Groq call per
                    # qualifying clause, on top of flag_clause()'s call
                    # above, against the same shared free-tier quota; it's
                    # arbitrary and wasteful to propose a negotiation point
                    # for a "low" severity clause Groq itself found
                    # unremarkable -- there's nothing worth negotiating
                    # there. Paced against the SAME last_call_time this
                    # loop already tracks (both calls hit the same
                    # model/project/quota, not a separate budget) --
                    # updated again below after this call completes, so
                    # the NEXT clause's own pacing wait correctly accounts
                    # for whichever call happened last.
                    if result.severity in ("high", "medium"):
                        wait = MIN_SECONDS_BETWEEN_FLAG_CALLS - (time.monotonic() - last_call_time)
                        if wait > 0:
                            time.sleep(wait)
                        try:
                            related = get_related_clauses_for_grounding(
                                contract_id, clause.text, db_session
                            )
                            if not related:
                                # get_related_clauses_for_grounding() already
                                # filters out stale Chroma candidates (see its
                                # own KNOWN GAP docstring) -- if EVERYTHING
                                # retrieved for this clause turned out stale,
                                # or genuinely nothing was retrieved, there is
                                # no real context to ground a suggestion in.
                                # Skip the Groq call entirely rather than
                                # sending a prompt with an empty "Related
                                # clauses:" section (which would just waste a
                                # call and likely fail grounding anyway).
                                logger.warning(
                                    "suggest_negotiation_point skipped for "
                                    "clause %d/%d: no related clauses were "
                                    "retrieved (see any stale-Chroma-vector "
                                    "warning above).",
                                    i, total,
                                )
                            else:
                                negotiation = suggest_negotiation_point(clause.text, related)
                                logger.info(
                                    "suggest_negotiation_point succeeded for clause %d/%d "
                                    "(severity=%s): suggestion=%r cited_clause_ids=%s",
                                    i, total, result.severity,
                                    negotiation.suggestion, negotiation.cited_clause_ids,
                                )
                        except NegotiationNeedsManualReviewError as exc:
                            # No persistence layer for negotiation suggestions
                            # exists yet (out of scope for this milestone,
                            # same as any UI display) -- logged clearly so
                            # this is real, inspectable, distinct-from-a-
                            # crash evidence rather than a silently accepted
                            # ungrounded suggestion.
                            logger.error(
                                "suggest_negotiation_point needs manual review "
                                "for clause %d/%d: %s",
                                i, total, exc,
                            )
                            # M35: this IS the "a citation fails grounding"
                            # event this milestone's spec asks for --
                            # suggest_negotiation.py itself is NOT modified
                            # (not in the M35 file list; it already retries
                            # internally, see its own module docstring), so
                            # this is recorded from here, the caller, on the
                            # terminal outcome (grounding failed on BOTH the
                            # original attempt and its one internal retry).
                            write_audit_entry(
                                contract_id=contract_id,
                                clause_id=clause.id,
                                event_type=EVENT_GROUNDING_FAILURE,
                                details=(
                                    f"clause {i}/{total}: negotiation suggestion "
                                    f"cited ungrounded clause_id(s) on all "
                                    f"attempts: {exc}"
                                ),
                            )
                        except SuggestNegotiationError as exc:
                            logger.error(
                                "suggest_negotiation_point failed for clause %d/%d: %s",
                                i, total, exc,
                            )
                            write_audit_entry(
                                contract_id=contract_id,
                                clause_id=clause.id,
                                event_type=EVENT_GROUNDING_ERROR,
                                details=f"clause {i}/{total}: suggest_negotiation_point failed: {exc}",
                            )
                        except VoyageEmbeddingError as exc:
                            # Found via a real production crash
                            # (2026-07-09, contracts 752d37cb and
                            # 364ce12f): get_related_clauses_for_
                            # grounding()'s own embed_with_retry() call
                            # (see that function's own docstring) raises
                            # this on a PERSISTENT, retries-exhausted
                            # Voyage failure -- pacing/retry already
                            # handled everything transient; this is what
                            # survives that. A single clause's grounding
                            # data being unavailable must never crash the
                            # whole contract's run, the same principle
                            # _flag_with_retry() already applies to
                            # flag_clause()'s own failures -- skip ONLY
                            # this clause's negotiation suggestion,
                            # clearly logged, and let the rest of the
                            # pipeline (remaining clauses' flagging AND
                            # their own grounding/suggestions) continue
                            # normally.
                            logger.error(
                                "suggest_negotiation_point skipped for clause "
                                "%d/%d: grounding retrieval's Voyage embed "
                                "call failed persistently (retries "
                                "exhausted): %s",
                                i, total, exc,
                            )
                            write_audit_entry(
                                contract_id=contract_id,
                                clause_id=clause.id,
                                event_type=EVENT_GROUNDING_ERROR,
                                details=(
                                    f"clause {i}/{total}: suggest_negotiation_point "
                                    f"skipped, grounding retrieval's Voyage embed "
                                    f"call failed persistently: {exc}"
                                ),
                            )
                        except KeyError as exc:
                            # Defense in depth alongside
                            # get_related_clauses_for_grounding()'s own
                            # internal filter (see its KNOWN GAP docstring):
                            # if a stale-Chroma-vector KeyError were ever
                            # raised here despite that filter (e.g. a future
                            # change to this call chain reintroducing a
                            # similar unguarded lookup), this clause's
                            # negotiation suggestion is skipped, clearly
                            # logged, and the REST of the pipeline continues
                            # -- a single clause's grounding data being stale
                            # must never crash the whole contract's run, the
                            # same principle _flag_with_retry() already
                            # applies to flag_clause()'s own failures.
                            logger.error(
                                "suggest_negotiation_point skipped for clause "
                                "%d/%d due to an unexpected KeyError (likely "
                                "a stale Chroma vector not caught by "
                                "get_related_clauses_for_grounding()'s own "
                                "filter): %s",
                                i, total, exc,
                            )
                            write_audit_entry(
                                contract_id=contract_id,
                                clause_id=clause.id,
                                event_type=EVENT_GROUNDING_ERROR,
                                details=(
                                    f"clause {i}/{total}: suggest_negotiation_point "
                                    f"skipped due to an unexpected KeyError "
                                    f"(likely a stale Chroma vector): {exc}"
                                ),
                            )
                        last_call_time = time.monotonic()

                logger.info("flagged %d/%d clauses...", i, total)

        try:
            # No delete needed here anymore -- the old flagged_clauses rows
            # for this contract_id were already deleted earlier, as part of
            # persist_chunks()'s own transaction (see this function's own
            # docstring). This is purely the NEW rows' insert now.
            db_session.add_all(flagged_rows)
            # M40: stamped in the SAME commit as the flagged_rows insert
            # above (not a separate _set_current_stage() call) -- this is
            # the one point where "current_stage=complete" and "the real
            # work that makes it true" become durable together, so a
            # poller can never observe current_stage=complete before the
            # flagged_clauses rows it implies actually exist.
            contract.current_stage = "complete"
            db_session.commit()
        except Exception:
            db_session.rollback()
            raise

    logger.info(
        "Pipeline complete for contract %s: %d clauses flagged (%d failed).",
        contract_id, len(flagged_rows), failed_count,
    )

    return {
        "clauses_persisted": total,
        "clauses_indexed": indexed_count,
        "clauses_flagged": len(flagged_rows) - failed_count,
        "clauses_failed": failed_count,
    }
