import logging
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile, status
from pydantic import BaseModel
from sqlalchemy import func

from auth.dependencies import get_current_user_id
from classification.centroid_fallback import SIMILARITY_THRESHOLD, centroid_similarities
from classification.heading_match import classify_heading
from db import SessionLocal
from embeddings.index import delete_contract_vectors
from ingestion.upload_validation import (
    FileTooLargeError,
    InvalidFileTypeError,
    validate_and_stream_to_disk,
    validate_content_length_header,
)
from middleware.rate_limit import DailyUploadLimitExceededError, check_daily_upload_limit
from models.clause import Clause
from models.contract import Contract
from models.flagged_clause import FlaggedClause
from models.upload_event import UploadEvent
from worker.tasks import process_contract_task

router = APIRouter()
logger = logging.getLogger("clauseguard.routes.contracts")

STORAGE_ROOT = Path("storage")

# M29: how close to M23's SIMILARITY_THRESHOLD (0.5) a stage-2 centroid
# match is allowed to sit before its category assignment is flagged as
# LOW confidence rather than HIGH. NOT reusing M20's threshold VALUES
# (0.62/0.71/0.65/0.55/0.68) -- those are calibrated against a totally
# different score (M21's reranker sigmoid, query-vs-clause) and reusing
# them here against a centroid cosine-similarity score would be exactly
# the "threshold applied to the wrong score" bug M20 exists to prevent.
# 0.10 is a fresh, independent judgment call for THIS score: M23's own
# independent review found real stage-2 cases only ~0.085 away from the
# 0.5 boundary on either side -- a margin has to be wider than that to
# actually catch those, so 0.10 is deliberately a bit more conservative
# than the narrowest real case observed, not a value borrowed from
# elsewhere. A stage-2 match with similarity in (0.5, 0.60] is LOW
# confidence (close enough to the boundary that it could plausibly have
# gone either way); similarity > 0.60 is HIGH confidence. Stage-2 can
# never assign a category at similarity <= 0.5 in the first place (see
# classify_by_centroid's own threshold check), so 0.5 is never itself a
# reachable "assigned" case here.
LOW_CONFIDENCE_MARGIN = 0.10
CENTROID_LOW_CONFIDENCE_CEILING = SIMILARITY_THRESHOLD + LOW_CONFIDENCE_MARGIN

# M29: process-local memoization of centroid_similarities() results, keyed
# by clause_id. centroid_similarities() makes a REAL, LIVE Voyage embed
# call every time it's invoked (see centroid_fallback.py's own docstring)
# -- this endpoint is polled repeatedly by the frontend (M14/M16), so
# recomputing this on every single poll would re-pay a live embedding
# call for the exact same, unchanging clause text over and over, quickly
# burning Voyage's free-tier 3 RPM budget for no new information. Mirrors
# the exact same in-process-memoization pattern centroid_fallback.py
# already uses for its own centroids (_centroids / _get_centroids) --
# not a new caching paradigm, the same one already established in this
# codebase. Correctly scoped: this app runs a single uvicorn worker (see
# backend/Dockerfile's CMD, no --workers flag), so a module-level dict is
# visible to every request in this process.
#
# Known, accepted limitation (documented, not silently ignored): this
# cache is process-local and in-memory only -- it is lost on a container
# restart (paying one more live Voyage call per affected clause on first
# access after restart, same as a cold cache anywhere else in this
# codebase) and would NOT be shared correctly across multiple worker
# processes if this app is ever scaled to --workers > 1. A real fix for
# either of those would be a persisted column on Clause, which is out of
# scope here (see module docstring below for why).
_similarity_cache: dict[uuid.UUID, dict[str, float]] = {}


def _get_similarities_cached(clause_id: uuid.UUID, text: str) -> dict[str, float] | None:
    """Real cosine similarities against all 5 category centroids, computed
    at most once per clause_id per process lifetime (see cache docstring
    above). Returns None (not a raised exception) if the underlying
    Voyage call fails -- a transient embedding-provider error on ONE
    clause's diagnostic confidence lookup must never take down the whole
    flagged-clauses response for every OTHER clause in the contract; the
    caller treats None the same as "could not determine confidence" (LOW,
    the conservative/safe default -- never silently reported as HIGH when
    it's actually unknown).
    """
    if clause_id in _similarity_cache:
        return _similarity_cache[clause_id]
    try:
        similarities = centroid_similarities(text)
    except Exception:
        logger.warning(
            "centroid_similarities() failed for clause %s -- reporting classification "
            "confidence as LOW (unknown) rather than crashing the whole response.",
            clause_id,
            exc_info=True,
        )
        return None
    _similarity_cache[clause_id] = similarities
    return similarities


class ContractUploadResponse(BaseModel):
    id: uuid.UUID
    filename: str
    status: str


class FlaggedClauseResponse(BaseModel):
    clause_id: uuid.UUID
    heading_path: str
    category: str | None
    severity: str
    explanation: str
    citation: str
    # M29: which classification stage produced `category` above (or that
    # neither stage did) -- "heading_match" | "centroid_fallback" |
    # "unclassified". Inferred by re-running stage 1's classify_heading()
    # (cheap, pure pattern matching, no API call) and, only when that
    # returns None, stage 2's real centroid_similarities() (see the
    # module-level cache above for why this isn't recomputed from
    # scratch on every poll). This is not itself a stored field on
    # Clause (out of scope -- see below); it's re-derived from the same
    # classification config used at persist time. If that config
    # (heading_match.py's lexicon, or centroid_fallback.py's
    # CATEGORY_EXAMPLES/centroids) is ever changed after a clause was
    # persisted, this recomputation would reflect the CURRENT config, not
    # necessarily what actually ran at persist time -- a known,
    # documented limitation of inferring provenance after the fact rather
    # than storing it, accepted here because storing it would require a
    # new Clause column, which is out of scope for this milestone.
    #
    # STABILITY, INDEPENDENTLY VERIFIED (not just assumed): this live
    # recomputation is stable and reproducible in real production use,
    # because it always embeds the exact same text (cl.text, heading+body)
    # that classify_clause() embedded at original persist time. Confirmed
    # by: (a) embed_text() called twice for identical text produced
    # bit-identical vectors (max abs diff 0.0) -- Voyage's embeddings are
    # NOT a source of jitter here; (b) the same real stage-2 clause
    # returned the identical similarity value across 12 consecutive calls
    # in one process, across 3 independent fresh processes, and after a
    # real container restart. A discrepancy WAS seen once during this
    # milestone's own development testing (a clause recomputed at 0.6018
    # against a persist-time decision made below 0.5), but it was traced
    # to a test-tooling mismatch, not a live-system risk: an earlier,
    # now-removed offline calibration script embedded body-text-only,
    # while this endpoint (like persist time) always embeds heading+body
    # -- two different inputs being compared, not the same input
    # producing different outputs. That script is gone; it has no bearing
    # on real endpoint behavior. The one genuine, still-accepted
    # limitation is the config-drift case described above (lexicon/
    # centroids changed AFTER a clause was persisted) -- not embedding
    # non-determinism.
    classification_stage: str
    # "high" | "low" -- see LOW_CONFIDENCE_MARGIN above and this
    # milestone's own reasoning for exactly how this is derived per
    # stage.
    classification_confidence: str
    # The real cosine similarity behind a "centroid_fallback" or
    # "unclassified" classification_confidence decision (None for
    # "heading_match", which has no similarity score to report -- stage 1
    # is a categorical pattern match, not a scored one). Exposed raw, not
    # just used internally, so a real user can see exactly how close a
    # borderline classification actually was.
    classification_similarity: float | None


class FlaggedClausesListResponse(BaseModel):
    contract_id: uuid.UUID
    status: str
    flagged_clauses: list[FlaggedClauseResponse]


class ContractStatusResponse(BaseModel):
    id: uuid.UUID
    status: str
    # M40: the REAL current pipeline stage ("extracting" | "chunking" |
    # "persisting" | "embedding" | "flagging" | "complete" | "failed" |
    # None), read directly off Contract.current_stage -- see that
    # column's own docstring (models/contract.py) for exactly when/how
    # it's updated and what None vs "failed" each genuinely mean.
    current_stage: str | None
    clauses_persisted: int
    clauses_flagged: int


# M38 ("My Contracts" history): a per-contract summary for the LIST view
# below -- deliberately NOT the same shape as FlaggedClauseResponse's own
# per-clause classification_confidence recomputation (get_flagged_clauses,
# below). That recomputation makes a real, live Voyage call (memoized,
# but still a real per-clause cost) for every stage-2-classified clause
# -- fine for opening ONE contract, but multiplied across every clause of
# every contract on a LIST page, it would turn "show me my history" into
# a burst of live embedding calls against Voyage's real 3 RPM budget for
# information a list view doesn't need. This summary intentionally uses
# ONLY the already-persisted, already-final FlaggedClause.severity column
# -- zero live API calls, zero recomputation, safe to compute for a whole
# page of contracts in one query (see the grouped query in list_contracts
# below). Three buckets, coarser than the full 4-bucket possible_risk/
# no_flag split the single-contract detail view shows (that split needs
# the live confidence score this summary deliberately skips):
#   clauses_flagged        -- severity in (high, medium): real risk
#   clauses_needs_review   -- severity in (needs_manual_review,
#                              flagging_failed): processing didn't
#                              produce a trustworthy assessment
#   clauses_no_flag        -- severity == low: no risk found
class ContractSummary(BaseModel):
    id: uuid.UUID
    filename: str
    status: str
    created_at: datetime
    clauses_flagged: int
    clauses_needs_review: int
    clauses_no_flag: int


class ContractListResponse(BaseModel):
    contracts: list[ContractSummary]
    limit: int
    offset: int
    total: int


@router.post(
    "/contracts/upload",
    response_model=ContractUploadResponse,
    status_code=status.HTTP_201_CREATED,
)
def upload_contract(
    request: Request,
    file: UploadFile = File(...),
    user_id: uuid.UUID = Depends(get_current_user_id),
):
    """M32: validated BEFORE the file is saved to storage or a Contract
    row is created -- reject bad uploads "at the door", never let them
    into storage or the pipeline at all. See ingestion/upload_validation.py
    for the real size/MIME checks themselves; this route only wires them
    in at the right point and turns a rejection into a clear, specific
    4xx response (never a generic 500 or a raw stack trace).

    M36 ADDITION: the per-user daily upload cap (middleware/rate_limit.py)
    is checked FIRST, before even the cheap Content-Length fast-path
    check below -- a user already over their daily limit should trigger
    essentially zero real work: no file write, no Contract row, no
    Celery task. See that module's own docstring for the full "daily"/
    "per-user"/"what counts" reasoning behind this check.

    M36-FIX (post-M37 holistic review): an UploadEvent row (models/
    upload_event.py) is now written in the SAME transaction as the
    Contract row below, at the exact same point -- i.e. still only for
    genuinely-accepted, post-M32-validation uploads, never for a
    rejected/invalid one. This is deliberately NOT written any earlier
    (e.g. right where check_daily_upload_limit() itself runs, before
    validation) -- doing so would start charging a user's daily quota
    for uploads that get rejected for being too large or the wrong file
    type, which never actually reach the pipeline or cost anything real,
    silently changing what this limit counts. See middleware/
    rate_limit.py's own docstring (M36-FIX section) for why this table,
    rather than the Contract row itself, is now the rate limiter's
    source of truth: a Contract row (and everything M37 cascades with
    it) can be permanently deleted by its own owner, which used to mean
    deleting a contract silently freed up that day's quota slot again --
    a real, confirmed exploit this new table exists specifically to
    close, by never being touched by anything else in this codebase once
    written.

    Ordering, deliberately: (0) the daily upload limit (M36); (1) the
    cheap, spoofable Content-Length fast-path check, before touching the
    body at all; (2) the real, unspoofable byte-count + magic-byte check
    while streaming the file to disk; (3) the Contract row and Celery
    dispatch below only ever happen after ALL THREE have passed. A
    rejected upload never creates a Contract row and never reaches the
    pipeline.
    """
    db = SessionLocal()
    try:
        try:
            check_daily_upload_limit(user_id, db)
        except DailyUploadLimitExceededError as exc:
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(exc)) from exc

        declared_size_header = request.headers.get("content-length")
        declared_size = None
        if declared_size_header is not None:
            try:
                declared_size = int(declared_size_header)
            except ValueError:
                # A malformed (non-numeric) header is treated the same as a
                # missing one -- no fast-path pre-rejection, but this changes
                # nothing about the REAL enforcement below, which never
                # trusted this header for anything but an early, optional exit.
                logger.warning("Ignoring malformed Content-Length header: %r", declared_size_header)
        try:
            validate_content_length_header(declared_size)
        except FileTooLargeError as exc:
            raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(exc)) from exc

        contract_id = uuid.uuid4()
        contract_dir = STORAGE_ROOT / str(user_id) / str(contract_id)
        contract_dir.mkdir(parents=True, exist_ok=True)
        storage_path = contract_dir / file.filename

        file.file.seek(0)
        try:
            validate_and_stream_to_disk(file.file, storage_path, file.filename)
        except FileTooLargeError as exc:
            raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(exc)) from exc
        except InvalidFileTypeError as exc:
            raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=str(exc)) from exc

        contract = Contract(
            id=contract_id,
            user_id=user_id,
            filename=file.filename,
            storage_path=str(storage_path),
            status="processing",
        )
        db.add(contract)
        # M36-FIX: written in the SAME transaction/commit as the Contract
        # row above -- see this function's own docstring for why here,
        # not any earlier, and middleware/rate_limit.py's own docstring
        # for why this table (not the Contract row) is now what
        # check_daily_upload_limit() counts against.
        db.add(UploadEvent(user_id=user_id))
        db.commit()
        db.refresh(contract)

        # M15: the full pipeline (extract, chunk, persist, classify,
        # embed/index, flag every clause via Groq) now runs
        # asynchronously in a separate Celery worker container -- this
        # request returns immediately rather than blocking for the
        # several real minutes M13's synchronous version took. The
        # worker (worker/tasks.py) calls the EXACT SAME
        # run_contract_pipeline() this route used to call directly, and
        # is responsible for moving status to "complete"/"failed" when
        # it finishes. The frontend polls GET /contracts/{id}/status
        # (M16, below) until a terminal state, then fetches
        # GET /contracts/{id}/flagged-clauses (M14) for the actual results.
        #
        # Race-condition note (M16): the frontend could in principle poll
        # /status before the row it's asking about exists. In practice
        # this can't happen here -- db.commit() above happens-before
        # .delay() and before this response is even returned, and a
        # Postgres commit is durable/visible to any other connection the
        # moment it completes. By the time the browser has this
        # response body to start polling from, the row is already
        # guaranteed visible. The frontend still tolerates a 404 on its
        # first couple of polls anyway (see App.jsx), as cheap defense in
        # depth against this reasoning ever becoming wrong (e.g. a future
        # read-replica setup), not because it's expected to fire today.
        process_contract_task.delay(str(contract_id))

        return ContractUploadResponse(
            id=contract.id,
            filename=contract.filename,
            status=contract.status,
        )
    finally:
        db.close()


# M38: page size is a real, enforced ceiling, not just a suggestion --
# see list_contracts()'s own docstring for why a user's history is never
# returned unbounded.
DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100


@router.get(
    "/contracts",
    response_model=ContractListResponse,
)
def list_contracts(
    user_id: uuid.UUID = Depends(get_current_user_id),
    limit: int = Query(DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    offset: int = Query(0, ge=0),
):
    """M38 ("My Contracts" history): every contract the authenticated user
    has ever uploaded, most recent first -- the real capability every
    earlier frontend milestone (M14/M16/M29/M33) explicitly scoped out.
    The data was always correctly persisted; this is the first route
    that lets a user see more than their single most-recent upload.

    OWNERSHIP -- filtered at the QUERY level (WHERE user_id = :user_id),
    not the fetch-then-compare-then-404 pattern every other per-contract
    route in this file uses (get_contract_status, get_flagged_clauses,
    delete_contract): those routes take a client-supplied contract_id and
    must check whether THAT SPECIFIC row belongs to the caller. This
    route takes no contract_id at all -- there is no input a caller could
    supply to ask for a different user's rows, because user_id itself
    comes only from the verified JWT (get_current_user_id), never from
    any request parameter. This is the same "never trust a client-
    supplied identity, only the token's" principle those routes already
    apply, expressed as a WHERE clause instead of a post-fetch check
    because a list has no single row to compare against.

    PAGINATION: limit defaults to 20, capped at 100 (FastAPI's Query(...,
    le=...) enforces this server-side regardless of what a caller passes
    -- not just a documented convention a client could ignore). offset
    defaults to 0. `total` (the real, unpaginated count of this user's
    contracts) is returned alongside the page so the frontend can tell
    whether more pages exist without a separate request.

    SUMMARY COUNTS: see ContractSummary's own docstring for exactly what
    clauses_flagged/clauses_needs_review/clauses_no_flag mean and why
    they deliberately do NOT require any live Voyage call -- computed
    here via ONE grouped query across every contract on this page
    (severity counts, grouped by contract_id), not one query per
    contract, so a full page of history costs one extra query total, not
    N.

    A contract still status="processing" naturally has zero FlaggedClause
    rows yet (the pipeline hasn't reached that stage), so all three
    counts are correctly 0 for it -- the frontend is expected to route a
    processing contract to the existing live-polling view instead of a
    static summary, the same way a fresh upload already does.

    A deleted (M37) contract simply no longer matches this query at all
    -- no special-casing needed here; the same real WHERE user_id=...
    filter that scopes this list also means a cascaded-away Contract row
    can never appear in it again.
    """
    db = SessionLocal()
    try:
        total = db.query(Contract).filter(Contract.user_id == user_id).count()

        contracts = (
            db.query(Contract)
            .filter(Contract.user_id == user_id)
            .order_by(Contract.created_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )

        contract_ids = [c.id for c in contracts]
        severity_rows = (
            db.query(FlaggedClause.contract_id, FlaggedClause.severity, func.count(FlaggedClause.id))
            .filter(FlaggedClause.contract_id.in_(contract_ids))
            .group_by(FlaggedClause.contract_id, FlaggedClause.severity)
            .all()
            if contract_ids
            else []
        )
        counts_by_contract: dict[uuid.UUID, dict[str, int]] = {}
        for cid, severity, count in severity_rows:
            counts_by_contract.setdefault(cid, {})[severity] = count

        summaries = []
        for contract in contracts:
            sev = counts_by_contract.get(contract.id, {})
            summaries.append(
                ContractSummary(
                    id=contract.id,
                    filename=contract.filename,
                    status=contract.status,
                    created_at=contract.created_at,
                    clauses_flagged=sev.get("high", 0) + sev.get("medium", 0),
                    clauses_needs_review=sev.get("needs_manual_review", 0) + sev.get("flagging_failed", 0),
                    clauses_no_flag=sev.get("low", 0),
                )
            )

        return ContractListResponse(contracts=summaries, limit=limit, offset=offset, total=total)
    finally:
        db.close()


@router.get(
    "/contracts/{contract_id}/status",
    response_model=ContractStatusResponse,
)
def get_contract_status(
    contract_id: uuid.UUID,
    user_id: uuid.UUID = Depends(get_current_user_id),
):
    """M16: lightweight polling endpoint -- the frontend calls this every
    few seconds after upload until status is "complete" or "failed", then
    switches to GET /contracts/{id}/flagged-clauses (M14) for the actual
    results. Also returns clause counts so far, cheap to compute from the
    same rows and useful for showing real progress during a multi-minute
    run, but the core field is just `status`.

    Same ownership-check pattern as every other per-contract route in
    this codebase (M11's search route, M14's flagged-clauses route): a
    404 (not 403) on a non-owned or nonexistent contract_id -- this is
    still confirming information about a specific user's contract, so it
    gets the same authorization rigor as routes that return full clause
    data, not a lighter check just because a bare status string seems
    "less sensitive."
    """
    db = SessionLocal()
    try:
        contract = db.query(Contract).filter(Contract.id == contract_id).one_or_none()
        if contract is None or contract.user_id != user_id:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Contract not found")

        clauses_persisted = db.query(Clause).filter(Clause.contract_id == contract_id).count()
        clauses_flagged = db.query(FlaggedClause).filter(FlaggedClause.contract_id == contract_id).count()

        return ContractStatusResponse(
            id=contract.id,
            status=contract.status,
            current_stage=contract.current_stage,
            clauses_persisted=clauses_persisted,
            clauses_flagged=clauses_flagged,
        )
    finally:
        db.close()


@router.get(
    "/contracts/{contract_id}/flagged-clauses",
    response_model=FlaggedClausesListResponse,
)
def get_flagged_clauses(
    contract_id: uuid.UUID,
    user_id: uuid.UUID = Depends(get_current_user_id),
):
    """M14: read-only fetch of a contract's flagged clauses, added to
    unblock the frontend -- the upload response (above) only ever
    returned counts, never the actual list with severity/explanation/
    citation.

    Same ownership-check pattern as M11's search route: a Postgres-level
    check (does the authenticated user own this contract?) runs before
    anything else, and a non-owned or nonexistent contract_id returns
    404 (not 403), so the response can't be used to distinguish "doesn't
    exist" from "exists but isn't yours".
    """
    db = SessionLocal()
    try:
        contract = db.query(Contract).filter(Contract.id == contract_id).one_or_none()
        if contract is None or contract.user_id != user_id:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Contract not found")

        rows = (
            db.query(FlaggedClause, Clause)
            .join(Clause, Clause.id == FlaggedClause.clause_id)
            .filter(FlaggedClause.contract_id == contract_id)
            .order_by(Clause.page_number, Clause.id)
            .all()
        )
        flagged_clauses = []
        for fc, cl in rows:
            # M29: re-derive which stage produced cl.category (see
            # FlaggedClauseResponse's own docstring above for why this is
            # recomputed rather than stored). Stage 1 first, cheap and
            # pure -- no API call, safe to run on every single request.
            stage1_category = classify_heading(cl.heading_path)
            if stage1_category is not None:
                classification_stage = "heading_match"
                classification_confidence = "high"
                classification_similarity = None
            else:
                # Stage 1 didn't match, so cl.category (if set at all)
                # came from stage 2 -- get the real similarity, memoized
                # per clause_id (see _get_similarities_cached above) so
                # this doesn't re-pay a live Voyage call on every poll.
                similarities = _get_similarities_cached(cl.id, cl.text)
                if similarities is None:
                    # Live lookup failed (see _get_similarities_cached) --
                    # report LOW confidence (unknown, not silently HIGH)
                    # rather than crash this clause's entire entry.
                    classification_stage = "centroid_fallback" if cl.category else "unclassified"
                    classification_confidence = "low"
                    classification_similarity = None
                else:
                    best_category = max(similarities, key=similarities.get)
                    best_score = similarities[best_category]
                    classification_similarity = round(best_score, 4)
                    if cl.category is not None:
                        classification_stage = "centroid_fallback"
                        classification_confidence = (
                            "low" if best_score <= CENTROID_LOW_CONFIDENCE_CEILING else "high"
                        )
                    else:
                        # Neither stage assigned anything -- always LOW
                        # confidence per this milestone's own spec, no
                        # margin/judgment call needed here (unlike the
                        # stage-2-assigned case above).
                        classification_stage = "unclassified"
                        classification_confidence = "low"

            flagged_clauses.append(
                FlaggedClauseResponse(
                    clause_id=fc.clause_id,
                    heading_path=cl.heading_path,
                    category=cl.category,
                    severity=fc.severity,
                    explanation=fc.explanation,
                    citation=fc.citation,
                    classification_stage=classification_stage,
                    classification_confidence=classification_confidence,
                    classification_similarity=classification_similarity,
                )
            )
        return FlaggedClausesListResponse(
            contract_id=contract_id,
            status=contract.status,
            flagged_clauses=flagged_clauses,
        )
    finally:
        db.close()


@router.delete(
    "/contracts/{contract_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_contract(
    contract_id: uuid.UUID,
    user_id: uuid.UUID = Depends(get_current_user_id),
):
    """M37: real deletion across every store a contract's data actually
    lives in -- Postgres (Contract + cascaded Clause/FlaggedClause/
    AuditLog rows, see models/clause.py, models/flagged_clause.py, and
    models/audit_log.py for the full per-table ondelete audit this
    endpoint's own correctness depends on), Chroma (real embedded
    vectors, embeddings/index.py's delete_contract_vectors()), and the
    real uploaded file on disk (contract.storage_path, M4's convention).

    Same ownership-check pattern as every other per-contract route in
    this codebase (M11's search route, M14's flagged-clauses route,
    M16's status route): a 404 (not 403) on a non-owned OR nonexistent
    contract_id, byte-identical either way -- this project's established
    anti-enumeration discipline, not a lighter check just because this
    route deletes rather than reads.

    ORDERING -- CHROMA, THEN FILE, THEN POSTGRES, deliberately, not an
    arbitrary choice (M37's own explicitly-flagged partial-failure risk):
    Postgres + Chroma + the filesystem are three separate systems with no
    shared transaction across them, so *some* ordering has to be picked
    for what happens if a later step fails after an earlier one already
    succeeded. Every one of these three deletes is naturally IDEMPOTENT
    on its own (Chroma's delete(where=...) matching zero vectors is a
    no-op; Path.unlink(missing_ok=True) tolerates an already-gone file;
    deleting an already-deleted Postgres row is impossible in the first
    place since the row simply won't be found). Given that, the ordering
    that matters is: do the two RETRIABLE, easily-redone steps FIRST,
    and do the step that removes this contract's own IDENTIFYING ANCHOR
    (the Postgres Contract row itself, and with it the user's own
    ability to even ask for this contract_id again) LAST.

    Concretely: if Chroma's delete or the file unlink fails partway
    through (a real I/O error, a transient Chroma error), the Postgres
    Contract row is STILL THERE, completely unchanged -- the user (or an
    operator) can simply call this same DELETE endpoint again, and
    whichever step already succeeded is a safe no-op the second time,
    while whichever step failed gets a fresh real attempt. If Postgres
    were deleted FIRST instead, and Chroma/file deletion failed
    afterward, there would be no way to ask "please finish deleting
    contract X's leftover data" through this API ever again -- the very
    identifier needed to retry is gone the moment the Postgres row is
    gone. Doing the Postgres delete LAST (and inside its own single
    transaction, so Contract + every cascaded child row commits or rolls
    back together, never partially) is what makes this endpoint safely
    retriable to a fully-deleted end state regardless of which specific
    step failed on a previous attempt.
    """
    db = SessionLocal()
    try:
        contract = db.query(Contract).filter(Contract.id == contract_id).one_or_none()
        if contract is None or contract.user_id != user_id:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Contract not found")

        storage_path = Path(contract.storage_path)

        deleted_vector_count = delete_contract_vectors(contract_id)
        logger.info(
            "Deleted %d Chroma vector(s) for contract %s.", deleted_vector_count, contract_id,
        )

        storage_path.unlink(missing_ok=True)
        logger.info("Deleted storage file %s for contract %s (or it was already gone).", storage_path, contract_id)

        # Single transaction: Contract's own row, plus every cascaded
        # Clause/FlaggedClause/AuditLog row (see this endpoint's own
        # docstring and the three models' own ondelete= audit) commit or
        # roll back together -- never a partial mix of Postgres tables.
        db.delete(contract)
        db.commit()
        logger.info("Deleted contract %s (Postgres row + cascaded rows) for user %s.", contract_id, user_id)
    finally:
        db.close()
