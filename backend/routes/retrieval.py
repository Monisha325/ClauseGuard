"""M11: POST /contracts/{contract_id}/search — the first retrieval
endpoint. Pure vector search only: embed the query with the same
embed_text() used at index time, search Chroma scoped to one contract,
return the top-5 clauses with real similarity scores.

M21 ADDITION: an optional cross-encoder reranking stage (see
retrieval/reranker.py), toggled per-request via `use_reranker` on the
request body. When true: retrieval widens to the top-20 vector-search
candidates (RERANK_CANDIDATES), the cross-encoder reranks all 20 against
the real query text, and the returned `score` field becomes the
reranker's sigmoid-transformed relevance score instead of the raw
vector-similarity score -- still 5 results returned, same response
shape, just a different ranking signal. When false (the default): this
endpoint's behavior is byte-for-byte identical to pre-M21 M11 -- same
top-5-by-vector-similarity path, same score formula, unchanged.

DEFAULT for use_reranker is False. M19's own independent review already
flagged ms-marco-MiniLM's domain mismatch (web-search training data, not
legal text) as a known, unresolved risk, and M22's A/B eval against
labeled data -- the thing that would actually tell us whether reranking
helps on real clause data -- hasn't run yet. Defaulting to the already-
verified pure-vector behavior is the safer choice until that evidence
exists; use_reranker=true is available for anyone (namely M22) who wants
to opt in and measure it.

M20 ADDITION: when use_reranker=true, the reranked top-5 are further
split into `confident_results` and `low_confidence_results` by
retrieval/thresholds.py's per-category threshold table, applied to the
reranker's real post-sigmoid score (see the explicit trace at the call
site below -- this is the exact bug the v4 architecture doc's threshold
revision fixed: gating on raw vector similarity instead of the
calibrated post-sigmoid score would produce meaningless results).
Low-confidence results are never silently dropped -- same "surface
uncertainty, don't hide it" philosophy as M13's flagging_failed sentinel.

GATING AND use_reranker=false: gating is intentionally NOT applied when
use_reranker=false. Raw vector similarity (1/(1+L2 distance)) has no
defined relationship to thresholds calibrated against the reranker's
sigmoid output -- applying them anyway would silently reintroduce
exactly the "gate on the wrong score" bug this milestone exists to
prevent, just one level up (gating on a score that was never sigmoid-
calibrated at all, category-specific threshold or not). Rather than
invent a second, meaningless threshold table for raw similarity, the
use_reranker=false path is left completely untouched from M21/M11 (same
flat `results` response, no confident/low_confidence split, no
thresholds module even imported for that path) -- this also means M21's
own independently-verified "use_reranker=false is byte-for-byte
identical to pre-M21 M11" property is mechanically preserved, not just
assumed to still hold.

M24 ADDITION: when use_reranker=true, the candidate pool fed to the
reranker is no longer vector-search's top-20 alone. It's now the RRF
fusion (retrieval/fusion.py, k=60) of TWO independently-retrieved top-20
lists -- vector search (Chroma, unchanged) and BM25 keyword search
(retrieval/bm25_index.py, both scoped to this same contract_id) --
truncated back down to RERANK_CANDIDATES=20 fused candidates before
reranking, preserving the reranker's existing ~20-candidate latency
profile (M21's own measured p95 target was tuned against that count,
not a larger fused union). This is a REAL, always-on part of the
use_reranker=true path, not hidden behind a second flag -- use_reranker
is already the established toggle for "run the expensive multi-stage
pipeline"; fusion is now simply part of what that pipeline does, the
same way M20's gating became part of it without its own separate flag.
No new toggle was added for this reason; M25's eval (should it need a
fusion-off reranked baseline for comparison) can call
retrieval/reranker.py's rerank() directly against a vector-only
candidate list itself, the same way M22's own eval already bypassed
this HTTP endpoint entirely to call rerank() directly.

use_reranker=false is UNCHANGED by this addition -- still the exact
pre-M21 top-5-by-vector-similarity path, no BM25 call, no fusion, no
Postgres clause fetch beyond the original ownership check. Fusion only
ever runs on the use_reranker=true path, for the same reason gating
only ever runs there (see above): there is no meaningful, calibrated way
to apply RRF-fused-then-reranked-shaped logic to a path that was never
going to touch the reranker in the first place.

Explicitly out of scope here (per M11's original spec, still true for
gating/BM25 individually, now delivered together): the confidence
threshold SANITY/calibration analysis (whether these five numbers are
actually good) is M26's job, not this milestone's -- this file only
implements the gating mechanism using the numbers as given, and the
fusion mechanism using k=60 as given.

Authorization: a Postgres-level ownership check (does the authenticated
user's user_id match this Contract's user_id?) runs BEFORE any Chroma
query — that check is the authoritative source of truth, not the Chroma
metadata filter. The Chroma query is ALSO scoped by contract_id in its
`where` filter as defense in depth, not as a substitute for the Postgres
check. A non-owned (or nonexistent) contract_id returns 404, not 403 —
deliberately chosen so the response can't be used to confirm whether a
given contract_id exists at all for a user who doesn't own it.
"""

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from auth.dependencies import get_current_user_id
from db import SessionLocal
from embeddings.index import _get_chroma_collection
from embeddings.voyage_client import VoyageEmbeddingError, embed_text
from models.clause import Clause
from models.contract import Contract
from retrieval.bm25_index import fetch_contract_clauses, search_bm25
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.reranker import rerank
from retrieval.thresholds import gate

router = APIRouter()

TOP_K = 5
RERANK_CANDIDATES = 20  # per spec: "reranking of top-20 candidates" -- also each of vector/BM25's own top-N before RRF fusion (M24), and the size the fused list is truncated back down to before reranking


class SearchRequest(BaseModel):
    query: str
    use_reranker: bool = False


class SearchResult(BaseModel):
    clause_id: str
    heading_path: str
    category: str | None
    text: str
    score: float


class GatedSearchResult(BaseModel):
    clause_id: str
    heading_path: str
    category: str | None
    text: str
    score: float
    threshold: float | None  # None means "no category-specific threshold applied" (Unclassified/unrecognized category) -- see thresholds.gate()'s own docstring, not a fabricated number. NOTE: response_model_exclude_none on the route (see below) means a None threshold/category is OMITTED from the JSON entirely (no key), not rendered as a literal null -- confirmed via this milestone's own testing; same "clearly absent, not fabricated" signal either way, just worth knowing before parsing the raw response.


class SearchResponse(BaseModel):
    # Exactly one of these two shapes is populated per response, never
    # both: `results` for use_reranker=false (unchanged pre-M21 M11
    # shape), `confident_results`/`low_confidence_results` for
    # use_reranker=true (M20's gated shape). response_model_exclude_none
    # on the route keeps whichever fields are unused out of the actual
    # JSON, so callers only ever see the fields relevant to what they asked for.
    results: list[SearchResult] | None = None
    confident_results: list[GatedSearchResult] | None = None
    low_confidence_results: list[GatedSearchResult] | None = None


@router.post("/contracts/{contract_id}/search", response_model=SearchResponse, response_model_exclude_none=True)
def search_contract(
    contract_id: uuid.UUID,
    payload: SearchRequest,
    user_id: uuid.UUID = Depends(get_current_user_id),
):
    db = SessionLocal()
    try:
        contract = db.query(Contract).filter(Contract.id == contract_id).one_or_none()
        if contract is None or contract.user_id != user_id:
            # 404, not 403: see module docstring — doesn't confirm to a
            # non-owner that this contract_id exists at all.
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Contract not found")
    finally:
        db.close()

    try:
        query_vector = embed_text(payload.query)
    except VoyageEmbeddingError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))

    collection = _get_chroma_collection()
    n_results = RERANK_CANDIDATES if payload.use_reranker else TOP_K
    result = collection.query(
        query_embeddings=[query_vector],
        n_results=n_results,
        where={"contract_id": str(contract_id)},
        include=["metadatas", "documents", "distances"],
    )

    candidates = [
        {
            "clause_id": metadata["clause_id"],
            "heading_path": metadata["heading_path"],
            "category": metadata["category"] or None,
            "text": document,
            # Chroma's default index space is L2 distance (lower = more
            # similar); converted to a bounded, higher-is-more-similar
            # score so the API doesn't leak the underlying index's raw
            # distance units. Ordering is unaffected — this is a
            # monotonic transform of the same distance values Chroma
            # returned, not a separately-computed or hardcoded number.
            "vector_score": 1.0 / (1.0 + distance),
        }
        for metadata, document, distance in zip(
            result["metadatas"][0], result["documents"][0], result["distances"][0]
        )
    ]

    if not payload.use_reranker:
        # Unchanged pre-M21 M11 path: top-5 by raw vector similarity.
        results = [
            SearchResult(
                clause_id=c["clause_id"],
                heading_path=c["heading_path"],
                category=c["category"],
                text=c["text"],
                score=c["vector_score"],
            )
            for c in candidates
        ]
        return SearchResponse(results=results)

    # M24: fuse vector's top-20 with a BM25 top-20 (same contract scope)
    # via RRF (k=60) BEFORE reranking. Both bm25_index.fetch_contract_clauses()
    # and search_bm25() are used as-is here (not reimplemented) -- see
    # retrieval/bm25_index.py's own SCOPING section for why passing this
    # contract's real Postgres-fetched candidates, and only this
    # contract's, is what makes BM25 search here safe from cross-contract
    # leakage. A fresh db session is used for this Postgres read (the
    # ownership-check session above is already closed by this point).
    db2 = SessionLocal()
    try:
        bm25_source_candidates = fetch_contract_clauses(contract_id, db2)
        clause_rows = db2.query(Clause).filter(Clause.contract_id == contract_id).all()
    finally:
        db2.close()

    # Single, comprehensive metadata source for the reranked/gated path:
    # a fused candidate may have come from BM25 alone (never appeared in
    # vector's own top-20, so it's absent from Chroma's metadata returned
    # above) -- Postgres's real Clause rows are the one place guaranteed
    # to have heading_path/category/text for EVERY candidate that could
    # possibly appear in the fused list, regardless of which method(s)
    # surfaced it.
    metadata_by_id = {
        str(c.id): {"heading_path": c.heading_path, "category": c.category, "text": c.text}
        for c in clause_rows
    }

    vector_ranking = [(c["clause_id"], c["vector_score"]) for c in candidates]
    bm25_ranking = search_bm25(payload.query, bm25_source_candidates, n_results=RERANK_CANDIDATES)
    # RRF only looks at rank/order within each input list (see fusion.py's
    # own docstring) -- truncated back down to RERANK_CANDIDATES so the
    # reranker sees the same ~20-candidate count it always has, not a
    # larger vector+BM25 union.
    fused = reciprocal_rank_fusion(vector_ranking, bm25_ranking)[:RERANK_CANDIDATES]

    # Reranked path: the fused candidates are rescored by the cross-encoder
    # against the real query text, then cut down to the top TOP_K by that
    # sigmoid score (NOT by the fused RRF score, which is discarded for
    # ranking purposes once reranking is on -- it was only ever the
    # candidate-selection signal feeding into this step).
    reranked = rerank(payload.query, [(clause_id, metadata_by_id[clause_id]["text"]) for clause_id, _rrf_score in fused])

    confident_results: list[GatedSearchResult] = []
    low_confidence_results: list[GatedSearchResult] = []
    for clause_id, sigmoid_score in reranked[:TOP_K]:
        meta = metadata_by_id[clause_id]
        # THE CRITICAL TRACE (M20's whole reason for existing -- the "v3
        # bug" the spec explicitly flags): `sigmoid_score` here is
        # exactly the value rerank() just returned in the loop variable
        # above -- reranker.py's own post-sigmoid cross-encoder score,
        # confirmed correct in M21's independent review. It is NOT a
        # vector_score or an RRF fused score (neither is referenced again
        # past this point). gate() only ever sees sigmoid_score.
        is_confident, threshold = gate(meta["category"], sigmoid_score)
        gated_result = GatedSearchResult(
            clause_id=clause_id,
            heading_path=meta["heading_path"],
            category=meta["category"],
            text=meta["text"],
            score=sigmoid_score,
            threshold=threshold,
        )
        if is_confident:
            confident_results.append(gated_result)
        else:
            low_confidence_results.append(gated_result)

    return SearchResponse(confident_results=confident_results, low_confidence_results=low_confidence_results)
