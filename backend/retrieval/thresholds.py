"""M20: per-category confidence threshold table and gating logic,
applied to M21's reranked (sigmoid) scores.

THRESHOLD VALUES -- PROVENANCE, READ BEFORE CHANGING: the five values
below (Limitation of Liability 0.62, Indemnification 0.71, Termination
0.65, Governing Law / Jurisdiction 0.55, Confidentiality 0.68) are taken
directly from this milestone's own spec, as given by the project owner.
They are NOT independently re-verified against the real v4 architecture
doc -- no such doc file exists anywhere in this repository (confirmed via
a repo-wide search before writing this file), and no prior milestone in
this project has ever had direct access to it either (M8's category
names and M9/M12's model IDs all followed this same "given directly by
the user, not independently re-derived from a doc this session can't
see" pattern). If the real doc is ever produced, these five numbers
should be checked against it directly before being trusted further.

THE CRITICAL REQUIREMENT THIS FILE EXISTS TO ENFORCE (the "v3 bug" the
spec explicitly flags): these thresholds are calibrated against the
reranker's POST-SIGMOID score (retrieval/reranker.py's rerank(), a real
0-1 probability-like value, independently verified correct -- see that
module's own docstring) -- NEVER against raw cosine/L2 vector similarity.
gate() below takes a bare `score: float` argument and has no way to
enforce on its own which score the caller passes in; the actual
enforcement point is in routes/retrieval.py, which must only ever call
gate() with the sigmoid score from reranker.rerank(), never with
c["vector_score"] (the first-pass Chroma similarity). See that file's
own comment at the call site for the explicit trace.

CATEGORY NAMES: must exactly match classification/heading_match.py's own
CATEGORIES tuple (M8's real taxonomy) -- verified by reading that file
directly before writing this dict, not retyped from memory. If that
tuple is ever revised, this dict's keys must be kept in sync with it by
hand, the same manual-sync obligation heading_match.py's own docstring
already places on its own external-doc dependency.
"""

CATEGORY_THRESHOLDS: dict[str, float] = {
    "Limitation of Liability": 0.62,
    "Indemnification": 0.71,
    "Termination": 0.65,
    "Governing Law / Jurisdiction": 0.55,
    "Confidentiality": 0.68,
}


def gate(category: str | None, score: float) -> tuple[bool, float | None]:
    """Decide whether `score` counts as confident for `category`.

    Returns (is_confident, threshold_used):
      - is_confident: True if score >= the category's threshold.
      - threshold_used: the actual numeric threshold this score was
        measured against, so the caller can show its work -- or None if
        no category-specific threshold applied (see below), which the
        caller should render as "no threshold" rather than a fabricated
        number.

    `score` MUST be the post-sigmoid rerank score from
    retrieval/reranker.py's rerank() -- see this module's own docstring
    for why. This function has no way to verify that on its own; it
    trusts its caller.

    category=None (M8's real, expected "Unclassified" outcome -- a
    heading that matched none of the 5 known categories, not an error)
    has no category-specific bar to measure against, so it is ALWAYS
    routed to low_confidence (is_confident=False, threshold_used=None)
    regardless of how high `score` is. Inventing a "conservative
    default" threshold for this case would just be a fabricated number
    with no basis in the real threshold table -- this project's
    established philosophy is to surface genuine uncertainty rather than
    guess (the same reasoning behind M13's flagging_failed sentinel: a
    clearly-marked "we don't have a real answer here" beats a
    confident-looking but arbitrary one). An unrecognized category
    string (which M8's classify_heading() should never actually produce,
    since it only ever returns one of the 5 known names or None, but
    defensively handled anyway) is treated identically to None, for the
    same reason: no known threshold, no basis to gate on.
    """
    if category is None or category not in CATEGORY_THRESHOLDS:
        return False, None

    threshold = CATEGORY_THRESHOLDS[category]
    return score >= threshold, threshold
