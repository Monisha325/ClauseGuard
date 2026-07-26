"""M23: stage-2 classification fallback for clauses M8's heading-match
(classification/heading_match.py's classify_heading()) leaves
Unclassified. Computes each category's centroid embedding from a
bootstrap set of REAL, already-correctly-classified clause texts, then
classifies a heading-less clause by cosine similarity against those 5
centroids -- assigning the highest-similarity category only if it
exceeds SIMILARITY_THRESHOLD, else leaving the clause Unclassified (a
valid, expected outcome, same philosophy as stage 1's None result).

===========================================================================
EXAMPLE-SOURCING PLAN -- READ THIS BEFORE CHANGING CATEGORY_EXAMPLES BELOW
===========================================================================

This milestone's own spec explicitly flags "centroid embeddings computed
from too few/unrepresentative examples" as a known risk, and requires
documenting exact sourcing BEFORE writing centroid logic. Three candidate
sources were named: (1) heading_match.py's lexicon phrases, (2) the
M19/M22 fixture contract's real indexed clauses, (3) any other real,
already-classified clauses already sitting in Postgres from prior
milestone testing.

SOURCE ACTUALLY USED: (2) and (3), merged -- a live query against the
running Postgres database (2026-07, this milestone's build) found 51 real
persisted Clause rows across many contracts from prior milestone testing
(M10, M15, M16, M19 fixture, and others), NOT just the 7-row M19/M22
fixture. Every example below is copied VERBATIM from one of those real
rows (clause_id noted per example for traceability) -- this is real legal
clause text that M8's stage-1 heading-match already correctly classified,
not synthetic or fabricated text.

SOURCE DELIBERATELY NOT USED: (1), heading_match.py's lexicon phrases
("hold harmless", "governing law", etc.). These are short (1-4 word)
phrases, not full clause paragraphs -- embedding a 2-word phrase and
embedding a 300-character legal paragraph are different regimes for an
embedding model (length/context strongly affect what these vectors
represent), and every clause this fallback will ever be asked to classify
at real inference time is a full paragraph, not a short phrase. Mixing
short-phrase embeddings into a centroid meant to match full-paragraph
embeddings would risk skewing the centroid away from what it needs to
match against, not strengthen it. Given real full-clause examples were
available in sufficient quantity (see counts below), this tradeoff wasn't
worth it. This is a judgment call, not a mechanical inclusion of every
listed candidate source -- documented here so it can be revisited.

DATA QUALITY FILTERING (done BEFORE counting/using anything below): the
raw Postgres query returned some multi-clause-topic TEXT blobs -- chunker
merge/test artifacts, not single-topic clause text. One row labeled
"Limitation of Liability" (clause_id 423adf5b) was actually a 6-section
whole-contract dump (Indemnification + Limitation of Liability +
Termination + Governing Law + Confidentiality + Notices concatenated,
~759 chars, with the labeled category's own content only ~15% of the
text) -- EXCLUDED entirely, since using it would inject roughly 5x more
off-topic content than on-topic content into that category's centroid.
Three OTHER rows have a real clause's content as the clear majority of
the text with a SHORT secondary section appended (e.g. a Governing Law
paragraph followed by a short Expenses paragraph, or a Limitation of
Liability paragraph followed by a short Notices paragraph) -- these were
KEPT, since the labeled category's content is still unambiguously the
primary topic, but are flagged individually below with an "impurity"
note so this isn't silently glossed over.

PER-CATEGORY EXAMPLE COUNT AND SOURCE (verified by direct query against
this milestone's own running Postgres instance, not estimated):

  Limitation of Liability : 7 examples (6 clean, 2 with a short trailing
                             secondary-section tail -- see inline notes)
  Confidentiality         : 6 examples (5 clean, 1 with a short trailing
                             secondary-section tail -- see inline notes)
  Governing Law/Jurisdiction: 4 examples (3 clean, 1 with a short
                             trailing secondary-section tail)
  Indemnification         : 3 examples (all clean) -- *** THIN CATEGORY,
                             FLAGGED BY NAME: only 3 real examples exist
                             in Postgres today. Every one of the 3 is a
                             genuine, clean, correctly-labeled clause, so
                             this is a REAL bootstrap, not a fabricated
                             one -- but 3 is a small basis for a centroid
                             and should be one of the first categories
                             recalibrated once M18's real hand-labeling
                             produces more examples.
  Termination             : 3 examples (all clean) -- *** THIN CATEGORY,
                             FLAGGED BY NAME, same reasoning as
                             Indemnification above: only 3 real examples
                             exist in Postgres today.

THIS IS A BOOTSTRAP, NOT A CALIBRATED CLASSIFIER: all of the above is
built from whatever real, already-classified examples happen to exist in
this project's database RIGHT NOW, purely because M18's real hand-
labeling work (15-20+ real contracts) has not happened yet. This is
explicitly a stand-in, not a substitute for recalibrating
CATEGORY_EXAMPLES and re-deriving centroids once that real labeled data
exists -- especially for the two thin categories flagged above.

===========================================================================
"""

import hashlib
import json
import logging
import math
import time
from collections import deque
from pathlib import Path

from embeddings.rate_limit import MAX_RATE_LIMIT_RETRIES, embed_with_retry, estimate_tokens, pace
from embeddings.voyage_client import MODEL as _EMBED_MODEL
from embeddings.voyage_client import VoyageEmbeddingError

logger = logging.getLogger("clauseguard.classification.centroid_fallback")

SIMILARITY_THRESHOLD = 0.5

# Voyage free-tier pacing floor between consecutive LIVE embed_text()
# calls while building centroids from scratch (see embeddings/index.py
# and eval/retrieval_eval.py for the same confirmed 3 RPM constraint).
# Only relevant on a cold cache build (see _load_or_build_centroids) --
# every subsequent call in this process, and every subsequent process
# that finds a valid disk cache, pays none of this cost.
MIN_SECONDS_BETWEEN_VOYAGE_CALLS = 21

# Disk-persisted cache: the 5 category centroid vectors, keyed by a hash
# of CATEGORY_EXAMPLES so an edit to the example set below is detected
# (cache treated as stale, centroids recomputed) rather than silently
# serving stale centroids computed from a since-changed example set.
CENTROID_CACHE_PATH = Path(__file__).resolve().parent / ".centroid_cache.json"


# Real clause text, copied verbatim from Postgres (see EXAMPLE-SOURCING
# PLAN above for the query, filtering, and per-category counts/reasoning
# behind this exact set). clause_id noted per example for traceability
# back to the real row it came from.
CATEGORY_EXAMPLES: dict[str, list[str]] = {
    "Limitation of Liability": [
        # clause_id=7491c804-d05f-431e-a23a-4654f49702d1 -- clean
        "LIMITATION OF LIABILITY\nIn no event shall either party be liable to the other for any indirect, incidental, special, or consequential\ndamages arising out of or related to this Agreement, regardless of the theory of liability asserted, and the\naggregate liability of either party under this Agreement shall not exceed the total fees paid during the twelve\nmonths preceding the claim.",
        # clause_id=f0ef0ec7-dbeb-433f-a9d1-125bed2162de -- clean
        "In no event shall either party be liable for indirect, incidental, or consequential damages arising out of this Agreement.",
        # clause_id=18591322-b565-432b-ac62-4a1da051246e -- clean
        "LIMITATION OF LIABILITY\nClient's total liability under this Agreement shall not exceed the fees paid in the preceding twelve months. Consultant's liability, however, shall be unlimited and shall include all direct, indirect, incidental, special, and consequential damages of any kind, without regard to the theory of liability asserted.",
        # clause_id=10b70401-0f62-42be-b50a-8c7e21102679 -- IMPURITY: short
        # trailing "Governing Law" section appended after the Limitation
        # of Liability content, which is still the clear majority of the text.
        "SERVICES AGREEMENT\nLIMITATION OF LIABILITY\nClient's total liability under this Agreement shall not exceed the fees paid in the preceding twelve months. Consultant's liability, however, shall be unlimited and shall include all direct, indirect, incidental, special, and consequential damages of any kind, without regard to the theory of liability asserted.\nGOVERNING LAW\nThis Agreement shall be governed by and construed in accordance with the laws of the State of Delaware, without regard to its conflict of laws principles.",
        # clause_id=2615658e-fc2a-45a1-8c04-0a95c330a0c5 -- IMPURITY: short
        # trailing "Notices" section appended, LoL content still the majority.
        "M15 ASYNC TEST AGREEMENT\nLIMITATION OF LIABILITY\nClient's total liability under this Agreement shall not exceed the fees paid in the preceding twelve months. Consultant's liability, however, shall be unlimited and shall include all direct, indirect, incidental, special, and consequential damages of any kind, without regard to the theory of liability asserted.\nNOTICES\nAll notices required under this Agreement shall be sent by certified mail to the address set forth in the preamble.",
        # clause_id=964b2738-7867-42c7-9f56-75f2ad4149c9 -- clean
        "M16 POLLING TEST AGREEMENT\nLIMITATION OF LIABILITY\nClient's total liability under this Agreement shall not exceed the fees paid in the preceding twelve months. Consultant's liability, however, shall be unlimited and shall include all direct, indirect, incidental, special, and consequential damages of any kind, without regard to the theory of liability asserted.",
        # clause_id=784cf976-2191-44fd-838b-759cd8069003 -- clean (M19/M22 fixture)
        "2. Limitation of Liability\nNeither party shall be liable to the other for any indirect, incidental, special, or consequential damages arising out of or relating to this Agreement, and each party's total aggregate liability shall not exceed the total fees paid under this Agreement in the twelve months preceding the claim.",
    ],
    "Indemnification": [
        # THIN CATEGORY -- see EXAMPLE-SOURCING PLAN above: only 3 real
        # examples exist in Postgres today. All 3 are clean.
        # clause_id=db9e0df5-c70d-44a3-9a65-7344f082b994
        "REVIEWER-MODIFIED TEXT FOR M10 RE-EMBED VERIFICATION: INDEMNIFICATION\nEach party shall indemnify, defend, and hold harmless the other party and its officers, directors, employees,\nand agents from and against any and all third-party claims, damages, losses, and expenses, including\nreasonable attorney fees, arising out of or resulting from the indemnifying party breach of this Agreement,\nnegligence, or willful misconduct in connection with the performance of its obligations hereunder.",
        # clause_id=30f980c8-e762-4fca-9ce4-88d2e4517166
        "MASTER SERVICES AGREEMENT\nThis Agreement is entered into between Vendor and Customer.\nINDEMNIFICATION\nConsultant shall indemnify, defend, and hold harmless Client, its officers, directors, and affiliates from and against any and all claims, damages, losses, liabilities, and expenses of any kind whatsoever, whether or not arising from Consultant's negligence, willful misconduct, or breach of this Agreement, with no cap or limitation of any kind on the amount or duration of such indemnification obligation, which shall survive termination of this Agreement in perpetuity.",
        # clause_id=374677a0-d6c0-4684-a8b6-594eee17b7fe (M19/M22 fixture)
        "4. Indemnification\nThe Vendor shall defend, indemnify, and hold harmless the Client from and against any and all claims, losses, damages, or expenses arising out of the Vendor's negligent acts, errors, or omissions in connection with the performance of the Services.",
    ],
    "Termination": [
        # THIN CATEGORY -- see EXAMPLE-SOURCING PLAN above: only 3 real
        # examples exist in Postgres today. All 3 are clean.
        # clause_id=04e99daa-feb2-4944-84b2-b8ec4eb4ee14
        "TERMINATION\nEither party may terminate this Agreement for convenience upon sixty days prior written notice to the other\nparty, and either party may terminate this Agreement immediately for cause upon a material breach by the\nother party that remains uncured for thirty days following written notice of such breach.",
        # clause_id=64084019-5bbd-42a9-b102-d1759c3e1cf3 (near-duplicate wording of the above, different contract)
        "TERMINATION\nEither party may terminate this Agreement for convenience upon sixty (60) days prior written notice to the other party, and either party may terminate this Agreement immediately for cause upon a material breach by the other party that remains uncured for thirty (30) days following written notice of such breach.",
        # clause_id=2fd28bd4-a63c-48b9-a866-b48b1818e505 (M19/M22 fixture)
        "3. Termination\nEither party may terminate this Agreement upon thirty days' prior written notice to the other party if the other party materially breaches any provision of this Agreement and fails to cure such breach within the notice period.",
    ],
    "Governing Law / Jurisdiction": [
        # clause_id=a88569c1-e19c-41f8-922b-f1da8014e4e2 -- clean
        "GOVERNING LAW\nThis Agreement shall be governed by and construed in accordance with the laws of the State of Delaware,\nwithout regard to its conflict of laws principles, and each party irrevocably submits to the exclusive\njurisdiction of the state and federal courts located within the State of Delaware for any dispute arising\nhereunder.",
        # clause_id=6cc8d1a7-857d-4ddc-a562-d5987ed3ed17 -- clean (near-duplicate of the above, only whitespace/line-wrap differs -- a distinct real DB row from a different contract, kept as-is)
        "GOVERNING LAW\nThis Agreement shall be governed by and construed in accordance with the laws of the State of Delaware, without regard to its conflict of laws principles, and each party irrevocably submits to the exclusive jurisdiction of the state and federal courts located within the State of Delaware for any dispute arising hereunder.",
        # clause_id=879b4706-729d-4039-90d0-c7c947472014 -- clean (New York variant, adds real wording diversity)
        "REVIEW M15 MIXED-OUTCOME TEST\nGOVERNING LAW\nThis Agreement shall be governed by and construed in accordance with the laws of the State of New York, without regard to its conflict of laws principles, and each party irrevocably submits to the exclusive jurisdiction of the state and federal courts located within the State of New York for any dispute arising out of or relating to this Agreement or its subject matter.",
        # clause_id=def9c5ab-9030-44d5-90f8-654334ae1714 (M19/M22 fixture) --
        # IMPURITY: short trailing "Expenses" section appended, Governing
        # Law content still the clear majority of the text.
        "5. Governing Law\nThis Agreement shall be governed by and construed in accordance with the laws of the State of Delaware, without regard to its conflict of laws principles, and the parties consent to the exclusive jurisdiction of the courts located in Wilmington, Delaware.\n6. Expenses\nClient shall reimburse Vendor for reasonable travel expenses and other documented out-of-pocket costs incurred in connection with performance of the Services, invoiced to Client on a monthly basis.",
    ],
    "Confidentiality": [
        # clause_id=814cf9af-555e-42cc-bf20-515c36d496a7 -- clean
        "CONFIDENTIALITY\nEach party shall protect the confidential information of the other party using at least the same degree of\ncare it uses to protect its own confidential information of similar importance, and shall not disclose such\nconfidential information to any third party without the prior written consent of the disclosing party, except as\nrequired by applicable law.",
        # clause_id=358b04c3-757b-49e3-a92e-140eb014208b -- clean (real heading was "DATA PROTECTION", a stage-1 miss that presumably got its category from an earlier manual/test correction -- content is unambiguously Confidentiality)
        "Each party shall implement reasonable technical and organizational measures to protect Confidential Information from unauthorized access or disclosure.",
        # clause_id=f377e0be-053d-45d4-be86-ef724e55038c -- clean
        "Each party agrees to protect the other Confidential Information using reasonable care, and shall not disclose it to any third party without prior written consent.",
        # clause_id=4f06459d-4dd0-4bab-b78e-8c52637c8c27 -- IMPURITY: short
        # trailing "Payment Terms" section appended, Confidentiality content
        # still the clear majority of the text.
        "CONFIDENTIALITY\nEach party shall protect the confidential information of the other party using at least the same degree of care it uses to protect its own confidential information of similar importance, and shall not disclose such confidential information to any third party without the prior written consent of the disclosing party, except as required by applicable law.\nPAYMENT TERMS\nCustomer shall pay all invoiced amounts within thirty (30) days of receipt. Late payments accrue interest at 1.5% per month on the outstanding balance.",
        # clause_id=a2393291-29bb-4f89-9d6a-ddcc5f61da04 -- clean
        "REVIEW M15 TEST AGREEMENT\nCONFIDENTIALITY\nEach party shall maintain the confidentiality of the other party's proprietary information and shall not disclose it to any third party without prior written consent, except as required by law.",
        # clause_id=dd952865-e15c-4319-a431-75f9feacd114 -- clean (M19/M22 fixture)
        "M19 SMOKE-TEST FIXTURE CONTRACT (synthetic, not a real contract)\n1. Confidentiality\nEach party shall keep confidential all proprietary information disclosed by the other party during the course of this Agreement and shall not disclose such information to any third party without prior written consent.",
    ],
}


def _examples_hash() -> str:
    """Stable hash of CATEGORY_EXAMPLES's exact content, used to detect a
    stale disk cache (see CENTROID_CACHE_PATH) -- if this file's example
    set is ever edited, the hash changes, the old cache is recognized as
    no longer matching, and centroids are recomputed from the new set
    rather than silently serving centroids built from different text.
    """
    canonical = json.dumps(CATEGORY_EXAMPLES, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Real cosine similarity (normalized dot product), NOT a raw dot
    product -- see this milestone's own known-failure-mode warning: an
    unnormalized dot product has no fixed range, which would make the
    0.5 threshold meaningless. Returns a value in [-1, 1] in general;
    real embeddings of related legal text are expected to land in a
    sensible positive sub-range (confirmed empirically during this
    milestone's own testing, not merely assumed).
    """
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _average_vector(vectors: list[list[float]]) -> list[float]:
    n = len(vectors)
    dim = len(vectors[0])
    return [sum(v[i] for v in vectors) / n for i in range(dim)]


def _compute_centroids_live() -> dict[str, list[float]]:
    """Embeds every example in CATEGORY_EXAMPLES (real, live Voyage calls,
    paced at MIN_SECONDS_BETWEEN_VOYAGE_CALLS) and averages each
    category's vectors into its centroid. Only ever called on a cold
    cache (no valid disk cache found) -- see _load_or_build_centroids.

    POST-INCIDENT FIX (audit finding, 2026-07-10): the per-example call
    below now goes through embed_with_retry() (embeddings/rate_limit.py
    -- the same shared retry mechanism already used by this module's own
    per-clause centroid_similarities() and by embeddings/index.py and
    agent/suggest_negotiation.py) instead of a bare embed_text() call --
    a transient Voyage failure mid-bootstrap used to crash this entire
    function with no retry at all. The EXISTING pacing loop
    (MIN_SECONDS_BETWEEN_VOYAGE_CALLS, time.sleep below) is deliberately
    left as-is, not replaced with rate_limit.py's own deque-based pace()
    -- this loop's shape (a simple last_call timestamp, not a rolling-
    window deque) is already correct for this function's own strictly
    sequential, one-call-at-a-time bootstrap, and forcing the deque
    pattern here would be a mismatched rewrite for no real benefit, not
    an improvement. Only the missing RETRY layer was added; pacing itself
    is untouched.

    A genuinely persistent (retries-exhausted) Voyage failure still
    raises VoyageEmbeddingError, uncaught, same as before -- deliberately
    NOT degraded to a partial/empty centroid set. This is a startup/
    bootstrap path: failing loudly here (blocking classification until
    the real problem is resolved) is more appropriate than silently
    proceeding with incomplete or missing centroids, which would corrupt
    every subsequent stage-2 classification decision in a way that's far
    harder to detect than a loud failure at boot.
    """
    logger.info(
        "Building M23 centroid fallback from scratch: %d categories, %d total example texts -- this makes real, paced Voyage API calls and only happens once (result is cached to disk).",
        len(CATEGORY_EXAMPLES),
        sum(len(texts) for texts in CATEGORY_EXAMPLES.values()),
    )

    centroids: dict[str, list[float]] = {}
    last_call: float | None = None
    for category, texts in CATEGORY_EXAMPLES.items():
        vectors = []
        for text in texts:
            if last_call is not None:
                wait = MIN_SECONDS_BETWEEN_VOYAGE_CALLS - (time.monotonic() - last_call)
                if wait > 0:
                    time.sleep(wait)
            vectors.append(embed_with_retry(text, context=f"centroid bootstrap ({category})"))
            last_call = time.monotonic()
        centroids[category] = _average_vector(vectors)
        logger.info("Computed centroid for %r from %d example(s).", category, len(texts))

    return centroids


def _load_or_build_centroids() -> dict[str, list[float]]:
    """Disk-cache layer: loads the 5 centroid vectors from
    CENTROID_CACHE_PATH if present AND its stored examples_hash matches
    CATEGORY_EXAMPLES's current hash; otherwise builds them live (real,
    paced Voyage calls -- see _compute_centroids_live) and persists the
    result. This is what makes centroid computation a one-time cost
    across process restarts, not just within a single process -- see
    _get_centroids for the additional in-process memoization layer on
    top of this.
    """
    current_hash = _examples_hash()

    if CENTROID_CACHE_PATH.exists():
        with CENTROID_CACHE_PATH.open("r", encoding="utf-8") as f:
            cached = json.load(f)
        if cached.get("examples_hash") == current_hash:
            logger.info("Loaded M23 centroids from disk cache (%s) -- no live Voyage calls made.", CENTROID_CACHE_PATH)
            return cached["centroids"]
        logger.info("Disk-cached centroids are stale (CATEGORY_EXAMPLES changed) -- recomputing.")

    centroids = _compute_centroids_live()
    CENTROID_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CENTROID_CACHE_PATH.open("w", encoding="utf-8") as f:
        json.dump({"examples_hash": current_hash, "embed_model": _EMBED_MODEL, "centroids": centroids}, f)
    return centroids


# In-process memoization -- computed at most once per process (via
# _load_or_build_centroids, which itself may hit disk instead of Voyage;
# see that function's docstring). Every classify_by_centroid() call after
# the first one in this process reuses this dict directly, at zero
# recomputation cost.
_centroids: dict[str, list[float]] | None = None

# POST-INCIDENT ADDITION: persistent, module-level pacing history for
# every REAL, per-CLAUSE Voyage call centroid_similarities() below makes
# (NOT the one-time centroid-bootstrap calls in _compute_centroids_live(),
# which already has its own adequate one-time pacing loop and is
# deliberately left untouched -- out of scope for this fix, see below).
#
# Mirrors _centroids's own persistent-across-calls pattern just above,
# for the same reason: classify_clause() (ingestion/chunker.py's
# persist_chunks()) calls classify_by_centroid() -> centroid_similarities()
# once per unclassified clause, across MANY separate calls within one
# contract's persistence stage, and potentially across many different
# contracts within this same long-lived worker process. A history
# deque scoped to a single call (the way embeddings/index.py's own
# index_contract_clauses() correctly scopes ITS history to one contract's
# indexing loop) would forget about the previous clause's real Voyage
# call the moment that call returned -- exactly the gap that let multiple
# rapid, unpaced calls slip through and trip Voyage's real rate limit,
# reproduced live: calls 1-3 succeeded, call 4 raised
# voyageai.error.RateLimitError uncaught, crashing the whole contract's
# processing (contract CN_Sheet_Final.pdf, a real production failure).
_pacing_history: deque[tuple[float, int]] = deque()


def _get_centroids() -> dict[str, list[float]]:
    global _centroids
    if _centroids is None:
        _centroids = _load_or_build_centroids()
    return _centroids


def centroid_similarities(text: str) -> dict[str, float]:
    """Cosine similarity of `text`'s embedding against all 5 category
    centroids -- exposed as its own function (not just an internal detail
    of classify_by_centroid) so callers/tests can inspect every
    category's real similarity score, not just the winning one.

    POST-INCIDENT FIX: the per-clause embed_text() call below now goes
    through embeddings/rate_limit.py's pace()/embed_with_retry() -- the
    SAME already-verified pacing/retry protection embeddings/index.py's
    own index_contract_clauses() has always used for this identical
    Voyage account and rate limit, reused (not reimplemented) via that
    shared module. Previously this was a bare embed_text() call with no
    pacing and no retry at all; a contract with several unclassifiable
    headings could fire multiple rapid calls here (BEFORE
    index_contract_clauses() -- M10's own dedicated, already-paced
    embedding stage -- ever runs) and trip Voyage's real 3 RPM limit,
    crashing the whole pipeline. See _pacing_history's own comment above
    for why this call site needs its OWN persistent history rather than
    reusing index_contract_clauses()'s per-call-scoped one.

    This function itself still does NOT catch a persistent (retries-
    exhausted) VoyageEmbeddingError -- it still genuinely raises in that
    case, preserving its own documented contract ("callers/tests can
    inspect every category's real similarity score" -- a caller that
    truly needs to know about a hard failure here still can). The
    graceful Unclassified fallback lives one level up, in
    classify_by_centroid() below, which already has its own established
    "no confident category = Unclassified, not an error" philosophy to
    extend -- routes/contracts.py's own _get_similarities_cached() (the
    OTHER real caller of this function, for the live flagged-clauses API
    display) already wraps its own call in a broad try/except and was
    never at risk of crashing anything; this fix does not need to touch
    that call site at all.
    """
    centroids = _get_centroids()
    tokens = estimate_tokens(text)
    pace(_pacing_history, tokens, context="stage-2 centroid classification")
    vector = embed_with_retry(text, context="stage-2 centroid classification")
    _pacing_history.append((time.monotonic(), tokens))
    return {category: _cosine_similarity(vector, centroid) for category, centroid in centroids.items()}


def classify_by_centroid(text: str) -> str | None:
    """Stage-2 fallback classification for a clause whose heading didn't
    match stage 1's lexicon. Returns the highest-similarity category if
    its cosine similarity exceeds SIMILARITY_THRESHOLD, else None (the
    clause remains Unclassified -- a valid, expected outcome, same
    philosophy as stage 1's None result, NOT an error or a forced guess
    into the closest-but-still-wrong category).

    POST-INCIDENT ADDITION: also returns None (Unclassified) if
    centroid_similarities() raises VoyageEmbeddingError -- i.e. pacing
    and MAX_RATE_LIMIT_RETRIES retries (embeddings/rate_limit.py) were
    not enough, a real, persistent failure, not just transient rate-
    limiting. Deliberately a straightforward extension of this
    function's OWN existing philosophy, not a new concept: "could not
    reach a confident classification" was already a valid, expected,
    non-error outcome here; "could not even get a similarity score
    because the embedding provider is persistently unavailable" is the
    same kind of outcome, not a reason to crash the entire contract's
    processing over ONE clause's classification.
    """
    try:
        similarities = centroid_similarities(text)
    except VoyageEmbeddingError as exc:
        logger.error(
            "Stage-2 centroid fallback could not classify this clause -- "
            "a persistent Voyage failure survived pacing and %d retries: "
            "%s. Falling back to Unclassified for this one clause rather "
            "than crashing the whole contract's processing.",
            MAX_RATE_LIMIT_RETRIES, exc,
        )
        return None

    best_category = max(similarities, key=similarities.get)
    best_score = similarities[best_category]

    if best_score > SIMILARITY_THRESHOLD:
        logger.info("Stage-2 centroid fallback matched %r (similarity=%.4f).", best_category, best_score)
        return best_category

    logger.info("Stage-2 centroid fallback found no category above threshold (best=%r at %.4f) -- remains Unclassified.", best_category, best_score)
    return None
