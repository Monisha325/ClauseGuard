"""M34: suggest_negotiation_point -- the SECOND tool in this project
(after M12's flag_clause). For a flagged clause, retrieves related
clauses via the EXISTING fusion+rerank retrieval pipeline (M20-M24,
routes/retrieval.py), then asks the LLM to propose a negotiation point
that cites only clause_ids from that specific retrieved set.

PROVIDER MIGRATION (Gemini -> Groq, 2026-07-09): see
agent/flag_clause.py's own module docstring for the full reasoning
(Gemini's confirmed 20-requests/day free-tier cap repeatedly causing
real lockouts) and the real evidence verifying Groq's tool-calling
guarantee. This module follows the exact same migration, model
(llama-3.3-70b-versatile), and forced-tool-choice pattern.

TWO-TOOL CONFIGURATION: see agent/tools.py's own "M34" section header
for the full reasoning (verified against M12's actual tool-forcing setup
before writing this module) -- this is a completely SEPARATE API call
from flag_clause(), with its own tool_choice restricted to just this one
tool, not a shared call where the model picks between two tools.

GROUNDING -- THE CORE, EXPLICITLY-FLAGGED RISK OF THIS MILESTONE: a
citation is only trustworthy if it names a clause_id that was ACTUALLY
in the specific retrieved set passed to THIS call -- not any clause_id
that merely exists somewhere in the contract, and not some other call's
retrieved set. suggest_negotiation_point() below builds `retrieved_ids`
directly from the SAME `related_clauses` argument the prompt itself was
built from (never refetched, never widened) and checks every cited
clause_id against exactly that set, every attempt.

RETRY PHILOSOPHY: reuses M33's agent/validate.py DESIGN (exactly one
bounded retry, the specific error fed back into the prompt so the model
can self-correct, a distinct terminal failure state on repeat failure)
for a DIFFERENT failure axis -- a grounding violation, not a Pydantic
schema-validation failure. Not the same code: validate.py's
validate_with_retry() is typed specifically around FlagClauseOutput and
pydantic.ValidationError, and reusing it here would mean either
weakening that typing or bolting an unrelated failure mode onto a
module that already has one clear job. The retry loop below is
structurally the same shape (a bounded `range()`, not a `while True`)
for the same reason M33's is: a real, explicitly accepted ceiling, not
an oversight.

SCOPE BOUNDARY -- API-error retries: unlike pipeline/run_contract.py's
_flag_with_retry() (which retries flag_clause() itself on a 429/503),
this module does NOT add a parallel API-error retry loop around its own
LLM call. That would be new scope beyond what this milestone asked
for (only the grounding-retry mechanism was requested) and would spend
even MORE of the shared Groq quota per clause on top of an already-
second call. An actual API/infrastructure failure here (network error,
no tool call returned, rate limit, etc.) is raised once as
SuggestNegotiationError and left to the caller (pipeline/run_contract.py)
to log and move on -- consistent with this project's "a single clause
failing must never take down the rest of the pipeline" principle.

QUOTA: this is a SECOND real Groq call per qualifying clause, on top
of flag_clause()'s existing one, against the SAME shared free-tier
RPM/RPD budget -- see pipeline/run_contract.py's own call site for how
this is paced against the same MIN_SECONDS_BETWEEN_FLAG_CALLS budget and
scoped to high/medium severity clauses only, not run for every clause.
"""

import hashlib
import json
import logging
import os
import time
import uuid
from collections import deque

from groq import Groq
from pydantic import ValidationError

from agent.tools import (
    SUGGEST_NEGOTIATION_TOOL_NAME,
    SuggestNegotiationOutput,
    build_suggest_negotiation_tool,
)
from embeddings.index import _get_chroma_collection
from embeddings.rate_limit import embed_with_retry, estimate_tokens, pace
from embeddings.voyage_client import VoyageEmbeddingError
from models.clause import Clause
from retrieval.bm25_index import fetch_contract_clauses, search_bm25
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.reranker import rerank
from routes.retrieval import RERANK_CANDIDATES, TOP_K

logger = logging.getLogger("clauseguard.agent.suggest_negotiation")

MODEL = "llama-3.3-70b-versatile"
MAX_OUTPUT_TOKENS = 2048

MAX_GROUNDING_RETRIES = 1  # exactly one retry -- see module docstring

# THIRD real call site for Voyage's embed_text(), found via a real
# production crash (2026-07-09): get_related_clauses_for_grounding()
# below called embed_text() bare, no pacing/retry, same class of bug as
# classification/centroid_fallback.py's stage-2 fallback (fixed earlier
# the same day) -- confirmed exposed by the Groq migration's much faster
# flagging pace triggering multiple rapid grounding calls within
# Voyage's real 3 RPM window, crashing TWO real contracts in a row with
# an uncaught VoyageEmbeddingError (voyageai.error.RateLimitError as the
# real cause) propagating all the way out of run_contract_pipeline().
# REUSE, NOT DUPLICATION: shares the exact same embeddings/rate_limit.py
# pace()/embed_with_retry() embeddings/index.py (M10) and
# classification/centroid_fallback.py (M23) already use. Persistent
# module-level history (not per-call-scoped like embeddings/index.py's
# own) for the same reason centroid_fallback.py's own history is
# persistent -- this function is called repeatedly, once per
# high/medium-severity clause, across a single long-lived worker
# process, and every call must see every OTHER call's recent pacing
# history to actually stay under the real rolling-60s limit.
_grounding_pacing_history: deque[tuple[float, int]] = deque()


class SuggestNegotiationError(Exception):
    """Raised for any failure suggesting a negotiation point via Groq --
    missing API key, a rejected request, a response with no tool call at
    all, the wrong tool called, or the tool call's own arguments failing
    Pydantic validation. Mirrors agent/flag_clause.py's FlagClauseError:
    a real API/infrastructure (or malformed-response) failure, distinct
    from a grounding failure (see NegotiationNeedsManualReviewError
    below), which is this module's own new, separate risk axis.
    """


class NegotiationNeedsManualReviewError(Exception):
    """Raised when Groq's suggest_negotiation_point tool call cites a
    clause_id NOT in the retrieved context set, on BOTH the original
    attempt and its one retry -- an ungrounded (fabricated, or drawn
    from elsewhere in the contract) citation. Same philosophy as M33's
    agent/validate.py NeedsManualReviewError: Groq responded
    successfully, but its own output failed a real correctness check
    twice in a row, so this is a distinct, visible failure state --
    never silently persisted or surfaced as a trustworthy suggestion.
    """


def _clause_ref(text: str) -> str:
    """Same short, stable log-correlation identifier pattern as
    agent/flag_clause.py's _clause_ref()."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def get_related_clauses_for_grounding(
    contract_id: uuid.UUID, query_text: str, db_session
) -> list[dict]:
    """Runs query_text through the EXACT SAME fusion+rerank retrieval
    pipeline routes/retrieval.py's search_contract() uses for
    use_reranker=true (M20-M24: vector search + BM25, RRF-fused,
    reranked by the cross-encoder) -- composing the SAME already-
    independently-tested functions that route calls (embed_with_retry,
    _get_chroma_collection, fetch_contract_clauses, search_bm25,
    reciprocal_rank_fusion, rerank), NOT reimplementing their logic.

    This calls those functions directly rather than the HTTP route
    itself: that route is bound to an authenticated request/response
    cycle (Depends, HTTPException, Pydantic response models) that
    doesn't fit a call from inside the already-authorized Celery
    pipeline. Mirrors pipeline/run_contract.py's own established
    precedent (see that module's docstring) of calling a target
    module's already-independently-callable sub-steps directly instead
    of duplicating their logic or awkwardly routing through an
    HTTP-shaped entry point that wasn't built for this caller.

    Deliberately SKIPS retrieval/thresholds.py's gate() (the
    confident/low_confidence split): that gate's specific numbers were
    calibrated for the CATEGORY-classification-confidence question
    (M20), a different question from "is this a good retrieval
    candidate to ground a negotiation suggestion in". The top TOP_K
    reranked results are returned as-is.

    The flagged clause itself is NOT excluded from its own query results
    on purpose -- a suggestion citing the very clause being negotiated
    is legitimate grounding, not a self-reference bug; kept simple
    rather than adding an exclusion rule the milestone never asked for.

    KNOWN GAP -- STALE CHROMA VECTORS (found during this milestone's own
    independent review; containment fix here, root cause NOT fixed by
    this function): index_contract_clauses() never purges a contract's
    OLD Chroma vectors before a pipeline re-run persists fresh Clause
    rows with fresh clause_id UUIDs -- inherited from M20-M24's
    routes/retrieval.py, which has the identical unguarded
    metadata_by_id[clause_id] pattern this function originally copied.
    On a contract that's been re-processed, Chroma's vector search can
    therefore return a clause_id with no matching row in `clause_rows`
    below (queried fresh, this call, from Postgres -- the real source of
    truth), which would otherwise raise an uncaught KeyError when
    building rerank()'s input. That crash is UNACCEPTABLE here
    specifically (unlike in the original HTTP route, where it would only
    fail one request): this function is called from inside
    pipeline/run_contract.py's synchronous per-clause loop, so an
    uncaught KeyError here would take down an ENTIRE contract's
    processing, not just skip one clause's negotiation suggestion. The
    filter below discards any such stale candidate before it can reach
    rerank() -- containment, not a fix for index_contract_clauses()
    itself never purging old vectors (a separate, still-open, tracked
    gap for future hardening work, alongside M32 review's own separately-
    tracked concurrency-locking gap -- neither is this function's job to
    resolve).

    Returns up to TOP_K dicts: {"clause_id", "heading_path", "category",
    "text"}. clause_id is a str (Clause.id stringified), matching
    routes/retrieval.py's own SearchResult/GatedSearchResult convention.

    Raises VoyageEmbeddingError (unchanged propagation, now via the
    shared pace()/embed_with_retry() below) if a persistent, retries-
    exhausted Voyage failure occurs -- this function does NOT catch it
    itself, matching classification/centroid_fallback.py's own
    centroid_similarities() precedent (raise, let the caller decide) --
    see pipeline/run_contract.py's own grounding call site for the
    graceful per-clause degradation this now enables.
    """
    tokens = estimate_tokens(query_text)
    pace(_grounding_pacing_history, tokens, context="grounding retrieval embed")
    query_vector = embed_with_retry(query_text, context="grounding retrieval embed")
    _grounding_pacing_history.append((time.monotonic(), tokens))
    collection = _get_chroma_collection()
    result = collection.query(
        query_embeddings=[query_vector],
        n_results=RERANK_CANDIDATES,
        where={"contract_id": str(contract_id)},
        include=["metadatas", "distances"],
    )
    candidates = [
        {"clause_id": metadata["clause_id"], "vector_score": 1.0 / (1.0 + distance)}
        for metadata, distance in zip(result["metadatas"][0], result["distances"][0])
    ]

    bm25_source_candidates = fetch_contract_clauses(contract_id, db_session)
    clause_rows = db_session.query(Clause).filter(Clause.contract_id == contract_id).all()
    metadata_by_id = {
        str(c.id): {"heading_path": c.heading_path, "category": c.category, "text": c.text}
        for c in clause_rows
    }

    vector_ranking = [(c["clause_id"], c["vector_score"]) for c in candidates]
    bm25_ranking = search_bm25(query_text, bm25_source_candidates, n_results=RERANK_CANDIDATES)
    fused = reciprocal_rank_fusion(vector_ranking, bm25_ranking)[:RERANK_CANDIDATES]

    # Containment fix (see KNOWN GAP above): drop any fused candidate
    # whose clause_id has no matching row in metadata_by_id -- a stale
    # Chroma vector left over from an earlier pipeline run on this same
    # contract_id -- BEFORE it can reach rerank() and raise a KeyError.
    # Logged (not silent) so a real, recurring pattern of stale vectors
    # is visible in aggregate, the same "surface it, don't hide it"
    # philosophy this project already applies to low-confidence results
    # elsewhere (e.g. M20's gate()).
    stale_ids = [clause_id for clause_id, _rrf_score in fused if clause_id not in metadata_by_id]
    if stale_ids:
        logger.warning(
            "get_related_clauses_for_grounding: dropping %d stale Chroma "
            "clause_id(s) for contract_id=%s with no matching Clause row "
            "(likely orphaned by an earlier pipeline re-run that never "
            "purged old vectors): %s",
            len(stale_ids), contract_id, stale_ids,
        )
    fused = [(clause_id, score) for clause_id, score in fused if clause_id in metadata_by_id]

    reranked = rerank(
        query_text,
        [(clause_id, metadata_by_id[clause_id]["text"]) for clause_id, _rrf_score in fused],
    )

    return [
        {
            "clause_id": clause_id,
            "heading_path": metadata_by_id[clause_id]["heading_path"],
            "category": metadata_by_id[clause_id]["category"],
            "text": metadata_by_id[clause_id]["text"],
        }
        for clause_id, _sigmoid_score in reranked[:TOP_K]
    ]


def _build_prompt(
    clause_text: str,
    related_clauses: list[dict],
    ungrounded_ids: list[str] | None,
) -> str:
    """The base prompt lists every related clause with its real clause_id
    inline, so Groq has no reason to invent one. When ungrounded_ids is
    not None, this is a RETRY prompt -- the specific clause_id(s) that
    failed grounding on the previous attempt are named explicitly, so the
    model has a concrete, actionable reason to correct itself (the exact
    same "feed the specific error back" principle as M33's
    agent/flag_clause.py._build_prompt()).
    """
    context_block = "\n\n".join(
        f"[clause_id: {c['clause_id']}] {c['heading_path']}\n{c['text']}"
        for c in related_clauses
    )
    base = (
        "You are a contract negotiation advisor. The following clause has "
        "been flagged as risky. Using ONLY the related clauses listed "
        "below as supporting context, propose ONE concrete negotiation "
        "point for the flagged clause, and call the "
        f"{SUGGEST_NEGOTIATION_TOOL_NAME} tool with your suggestion. Cite "
        "clause_id(s) in cited_clause_ids ONLY if they appear in the "
        "related clauses list below, exactly as shown -- never a "
        "clause_id from anywhere else.\n\n"
        f"Flagged clause:\n{clause_text}\n\n"
        f"Related clauses:\n{context_block}"
    )
    if ungrounded_ids is None:
        return base
    return (
        base
        + "\n\nYour previous suggestion cited clause_id(s) "
        + ", ".join(ungrounded_ids)
        + " which do NOT appear in the related clauses list above. Call "
        "the tool again, citing ONLY clause_id(s) that appear in that "
        "list -- omit cited_clause_ids entirely (an empty list) if none "
        "of them genuinely support your suggestion."
    )


def _call_groq_once(
    client: Groq,
    tool: dict,
    clause_ref: str,
    prompt: str,
) -> SuggestNegotiationOutput:
    """Exactly ONE real Groq request plus ONE schema-validation check --
    no grounding check and no retry logic of its own (see
    suggest_negotiation_point() below for both). Mirrors
    agent/flag_clause.py's _call_groq_once() closely, including its own
    tool_choice restricted to just this one tool name (see this module's
    own docstring, TWO-TOOL CONFIGURATION).

    Raises SuggestNegotiationError for any API/infrastructure failure,
    OR if the tool call's own arguments fail Pydantic validation --
    unlike flag_clause.py, this milestone does not add a second,
    independent validation-retry mechanism on top of the grounding-retry
    one (see this module's own docstring, SCOPE BOUNDARY); a malformed
    response here is treated the same as any other hard failure.
    """
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            tools=[tool],
            tool_choice={"type": "function", "function": {"name": SUGGEST_NEGOTIATION_TOOL_NAME}},
            max_completion_tokens=MAX_OUTPUT_TOKENS,
        )
    except Exception as exc:
        raise SuggestNegotiationError(
            f"Groq request failed (model={MODEL!r}): {exc}"
        ) from exc

    choice = response.choices[0]
    logger.info(
        "suggest_negotiation_point call clause_ref=%s model=%s finish_reason=%s usage=%s",
        clause_ref, MODEL, choice.finish_reason, response.usage,
    )

    tool_calls = choice.message.tool_calls
    if not tool_calls:
        raise SuggestNegotiationError(
            "Groq did not return a tool call for suggest_negotiation_point "
            f"-- got a plain-text response instead: {choice.message.content!r}"
        )

    call = tool_calls[0]
    if call.function.name != SUGGEST_NEGOTIATION_TOOL_NAME:
        raise SuggestNegotiationError(
            f"Groq called unexpected tool {call.function.name!r}, expected "
            f"{SUGGEST_NEGOTIATION_TOOL_NAME!r}"
        )

    try:
        args = json.loads(call.function.arguments)
    except json.JSONDecodeError as exc:
        raise SuggestNegotiationError(
            f"suggest_negotiation_point tool call arguments were not valid "
            f"JSON: {exc} (raw: {call.function.arguments!r})"
        ) from exc

    try:
        return SuggestNegotiationOutput.model_validate(args)
    except ValidationError as exc:
        raise SuggestNegotiationError(
            f"suggest_negotiation_point tool call arguments failed validation: {exc}"
        ) from exc


def _grounding_violations(cited_ids: list[str], retrieved_ids: set[str]) -> list[str]:
    """Returns the subset of cited_ids that are NOT members of
    retrieved_ids -- empty means fully grounded. `retrieved_ids` must be
    exactly the clause_id set retrieved for THIS SPECIFIC call (see
    suggest_negotiation_point()'s own docstring below) -- passing the
    wrong set here (e.g. all contract clauses, or a different call's
    retrieved set) would silently defeat the entire point of this check.
    """
    return [cid for cid in cited_ids if cid not in retrieved_ids]


def suggest_negotiation_point(
    clause_text: str,
    related_clauses: list[dict],
) -> SuggestNegotiationOutput:
    """Suggest a negotiation point for clause_text, grounded ONLY in
    related_clauses (a list of {"clause_id", "heading_path", "category",
    "text"} dicts -- typically get_related_clauses_for_grounding()'s own
    return value for this exact clause, but any caller-supplied list of
    the same shape works).

    GROUNDING CHECK -- this milestone's own explicit, central risk:
    `retrieved_ids` (below) is built directly from `related_clauses`,
    the SAME argument this function's own prompt is constructed from --
    never refetched from the database, never widened to "all clauses in
    the contract", never carried over from a different call. Every
    attempt's cited_clause_ids is checked against exactly that set.

    On a grounding violation, retries EXACTLY ONCE (MAX_GROUNDING_RETRIES
    = 1, enforced by the bounded `range()` below, not a `while True`)
    with the specific ungrounded clause_id(s) fed back into the retry
    prompt. If the retry ALSO fails grounding, raises
    NegotiationNeedsManualReviewError -- never returns or silently
    accepts an ungrounded suggestion.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise SuggestNegotiationError(
            "GROQ_API_KEY is not set in the environment. Set it in "
            ".env (see .env.example) before calling suggest_negotiation_point()."
        )

    # THE grounding set for this call, and only this call -- built once,
    # from the exact `related_clauses` argument passed in, reused for
    # every attempt below (never rebuilt from a broader source).
    retrieved_ids = {c["clause_id"] for c in related_clauses}

    client = Groq(api_key=api_key)
    tool = build_suggest_negotiation_tool()
    clause_ref = _clause_ref(clause_text)

    ungrounded_ids: list[str] | None = None
    total_attempts = MAX_GROUNDING_RETRIES + 1

    for attempt_number in range(total_attempts):
        prompt = _build_prompt(clause_text, related_clauses, ungrounded_ids)
        result = _call_groq_once(client, tool, clause_ref, prompt)

        violations = _grounding_violations(result.cited_clause_ids, retrieved_ids)
        if not violations:
            if attempt_number > 0:
                logger.info(
                    "suggest_negotiation_point grounding retry SUCCEEDED for "
                    "clause_ref=%s (attempt %d/%d).",
                    clause_ref, attempt_number + 1, total_attempts,
                )
            return result

        attempts_remaining = total_attempts - (attempt_number + 1)
        logger.warning(
            "suggest_negotiation_point grounding FAILED for clause_ref=%s "
            "(attempt %d/%d) -- cited=%s retrieved_ids=%s ungrounded=%s -- %s.",
            clause_ref, attempt_number + 1, total_attempts,
            result.cited_clause_ids, sorted(retrieved_ids), violations,
            f"retrying with the ungrounded clause_id(s) fed back into the prompt ({attempts_remaining} retry(ies) left)"
            if attempts_remaining > 0
            else "no retries left, marking needs_manual_review",
        )
        ungrounded_ids = violations

    raise NegotiationNeedsManualReviewError(
        f"suggest_negotiation_point output cited ungrounded clause_id(s) "
        f"{ungrounded_ids} for clause_ref={clause_ref} on all "
        f"{total_attempts} attempt(s) (1 original + {MAX_GROUNDING_RETRIES} "
        f"retry). Retrieved set for this call was {sorted(retrieved_ids)}."
    )
