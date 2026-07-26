"""M21: cross-encoder reranking stage between vector retrieval and
generation.

M28 ADDITION: rerank()'s actual model.predict() call is now wrapped in
observability/timing.py's timed_stage("rerank_inference", ...) -- a
real, persistent, always-on per-call latency measurement (M21's own
latency work below was a one-off benchmark script, not standing
instrumentation), so p50/p95 can be computed from real production calls
against the architecture doc's ~180ms/450ms targets. Only the actual
inference call is inside the timed block -- see that call site's own
comment for exactly what is and isn't included, and this milestone's own
testing for the concrete trace confirming it.

M28 CORRECTION (post-independent-review): any future rerank latency
benchmark in this project MUST use a pool of RERANK_CANDIDATES (20) real
candidates (routes/retrieval.py's actual production pool size, and the
same 20 M21's own truncation benchmark below already used) -- NOT an
arbitrary smaller pool. M28's own first benchmark used a pool of 7 (the
largest single real contract in the dev DB at the time), which
understated real p50 latency by ~37% (227.91ms measured vs 312.46ms at
the correct pool=20 scale) -- corrected in this milestone's final
report, not re-measured here in code.

MODEL CHOICE (confirmed, not guessed, before hardcoding): the v4
architecture doc referenced the "ms-marco-MiniLM" family without an
exact HuggingFace repo string. Checked directly against HuggingFace
before writing this file: cross-encoder/ms-marco-MiniLM-L6-v2 is the
current, actively-maintained repo name (80.8M downloads in the prior
month at the time of this check, no deprecation notice on the model
card) -- not the older "ms-marco-MiniLM-L-6-v2" (hyphen before the
layer count) naming some older docs/tutorials still reference. 6-layer
distilled MiniLM, 22.7M parameters -- small enough for real CPU
inference at this project's scale (see this milestone's own latency
measurements, not assumed from the parameter count alone).
sentence-transformers==5.6.0 (current stable release on PyPI at the
time of this check) provides the CrossEncoder class used below.

KNOWN, ALREADY-DOCUMENTED DOMAIN RISK (per M19's own independent review
and the v4 architecture doc itself -- NOT something this milestone
attempts to solve): ms-marco-MiniLM was trained on MS MARCO, real Bing
web-search queries against web passages -- not legal contract text.
Whether reranking with this model actually improves ranking quality on
ClauseGuard's real clause data, versus pure vector search, is an open,
unanswered question. This milestone's job is only building a REAL,
WORKING, TOGGLEABLE reranking stage with a mathematically correct
sigmoid transform -- M22's future A/B eval against labeled data is where
"does this actually help" gets measured, not here.

SIGMOID TRANSFORM (do not skip, do not assume it's already applied):
CrossEncoder.predict() for this model returns a RAW LOGIT by default --
any real number, frequently negative, NOT a probability. Feeding that
raw value into M20's future threshold table (built for 0-1 probability-
like scores) would silently produce nonsense gating the moment that
table exists. sigmoid(x) = 1 / (1 + e^-x) is applied explicitly here to
every raw score before it is ever returned or ranked on. This was
confirmed empirically against real model output during this milestone's
own testing (real scores observed in the 0-1 range, not merely assumed
correct because the formula looks right on paper). Also independently
re-verified: model.activation_fn for THIS model is Identity() (not
Sigmoid, despite num_labels=1 -- the model's own saved config overrides
sentence-transformers' generic num_labels-based default), and raw
predict() output was confirmed genuinely unbounded (e.g. -10.56, 1.39)
-- so this sigmoid call is not a double-transform.

TRUNCATION (max_length=MAX_SEQ_LENGTH, fixes a real, measured latency
defect -- see below): independent review found the originally-reported
~244ms p95 latency figure only held for unrealistically short (~41
token) synthetic test text; with realistic legal-clause-length text
(220-297 tokens, matching what a real long clause chunk looks like),
p95 was actually 1506-1688ms -- 3-4x over the ~450ms architecture-doc
target, with no truncation applied (the untruncated default is 512,
this model's own max, meaning long clauses were processed in full).
Root-caused to CPU self-attention cost scaling with input length, not
missing batching (CrossEncoder.predict() already receives the full
candidate list in one call -- confirmed by reading this file's own
rerank() below -- so batching was never the problem).

Fix: truncate to MAX_SEQ_LENGTH=128 tokens. Reasoning, not a blind
latency hack: (a) this model was trained on MS MARCO passages averaging
56-73 tokens (median 50, training-distribution max 362) -- 128 is
already >1.7x the training mean, well outside "aggressively short" for
what this model actually learned to score well; (b) legal clauses
front-load their topic (heading + core obligation sentence) in the
opening tokens -- empirically confirmed during this fix's own testing:
4 realistic clause templates (Indemnification/Limitation of Liability/
Confidentiality/Termination, 216-293 tokens each) reranked correctly
(4/4, matching clause chosen with a clear score margin) at BOTH
max_length=512 and max_length=128 against their natural queries -- (c)
measured latency at 128 tokens: p95=403.2ms across 20 realistic-length
candidates, under the ~450ms target; 256 (p95=796.7ms) and 180
(p95=518.1ms) were both tried first and were insufficient on their own.
This is a real tradeoff, not a free lunch: a clause whose semantically
distinguishing content appears ONLY after token 128 (unusual for how
these clauses are drafted, but not impossible for an unusually long,
back-loaded clause) would lose that signal. Accepted as the right
tradeoff for this milestone given the measured latency gap and the
empirical correctness check above; M22's future A/B eval against real
labeled data is the right place to catch a truncation-driven quality
regression for real, if one exists, since this fix's own quality check
used representative but still synthetic clause text.
"""

import math

from sentence_transformers import CrossEncoder

from observability.timing import timed_stage

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"
MAX_SEQ_LENGTH = 128  # see TRUNCATION note above for the reasoning and measurements behind this number

_model: CrossEncoder | None = None


def _get_model() -> CrossEncoder:
    """Lazily load and cache the cross-encoder model.

    Loaded once per process (module-level singleton), not once per
    request -- CrossEncoder(...) loads model weights from disk, which
    would be a large, needless latency hit on every single search call.
    The model is pre-downloaded into the Docker image at build time (see
    backend/Dockerfile) so this load never needs a live network call at
    request time either.
    """
    global _model
    if _model is None:
        _model = CrossEncoder(MODEL_NAME, max_length=MAX_SEQ_LENGTH)
    return _model


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def rerank(query: str, candidates: list[tuple[str, str]]) -> list[tuple[str, float]]:
    """Rerank (candidate_id, candidate_text) pairs against `query` using
    the cross-encoder.

    Returns [(candidate_id, sigmoid_score), ...] sorted descending by
    sigmoid_score (best/most-relevant first). sigmoid_score is always in
    [0, 1] -- a real transformed probability-like value, never a raw
    logit -- see the sigmoid transform note above.

    candidates: list of (candidate_id, candidate_text). The caller
    decides what candidate_id represents (here, a clause_id string) --
    this function only reasons about text pairs and scores.
    """
    if not candidates:
        return []

    model = _get_model()
    pairs = [(query, text) for _, text in candidates]

    # M28: timed block isolates ONLY the actual cross-encoder inference
    # call -- not _get_model() (a one-time lazy load, already excluded
    # since it happens before this block; a cached-singleton lookup on
    # every subsequent call anyway), not the pairs list comprehension
    # above (pure Python, negligible, and arguably part of "preparing
    # the request" rather than "reranking" itself), and not the sigmoid
    # transform + sort below (also pure Python, negligible). This
    # matches M21's own isolated benchmark methodology (which measured
    # this exact predict() call, not surrounding glue code) -- see this
    # milestone's own testing step 2 for the explicit trace confirming
    # no network/DB round-trip time is included here (there is none:
    # `candidates` is already-fetched text, passed in by the caller).
    with timed_stage("rerank_inference", n_candidates=len(candidates)):
        raw_scores = model.predict(pairs)

    scored = [
        (candidate_id, _sigmoid(float(raw_score)))
        for (candidate_id, _text), raw_score in zip(candidates, raw_scores)
    ]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored
