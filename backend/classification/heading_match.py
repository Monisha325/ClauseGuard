"""Stage-1 clause classification: heading_path pattern matching against a
fixed 5-category taxonomy.

PROVENANCE (important, read before changing the category set): the 5
category names below — Limitation of Liability, Indemnification,
Termination, Governing Law / Jurisdiction, Confidentiality — are sourced
directly from the v4 architecture doc's classification/threshold table,
as given by the project owner (the doc itself was not available in this
session; its category names were provided directly rather than guessed,
per explicit instruction not to fabricate them). The doc specifies only
category names and post-sigmoid thresholds, NOT a heading-phrase lexicon.
The lexicon below — which phrases match which category — is this
implementation's own construction, using realistic legal-drafting
judgment about how these clause types are commonly titled in real
contracts. If M20/M26's threshold table is ever revised, the CATEGORIES
tuple here must be kept in sync with it.

This is stage-1 only: pure pattern matching against heading_path, no
embeddings, no confidence score. A heading that doesn't match any
category's lexicon returns None — that is the correct, expected result at
this stage, not an error or a guess.

M23 ADDITION: classify_clause() below is the real two-stage cascade —
stage 1 (classify_heading(), unchanged, still callable on its own for
anyone who only wants heading-only classification), falling back to
stage 2 (classification/centroid_fallback.py's embedding-centroid
classifier) ONLY when stage 1 returns None. classify_heading() itself is
NOT modified by this addition — it stays pure heading-text matching, zero
embeddings/DB/network dependency, exactly as before. The new dependency
on centroid_fallback.py (and transitively, embeddings/voyage_client.py's
live API) is isolated to classify_clause(), not this module's original
function.
"""

import re

from classification.centroid_fallback import classify_by_centroid

CATEGORIES = (
    "Limitation of Liability",
    "Indemnification",
    "Termination",
    "Governing Law / Jurisdiction",
    "Confidentiality",
)

# Lexicon: for each category, a list of lowercase "root" phrases.
# Matching is word-boundary-bounded substring matching against a
# normalized heading (see _PHRASE_ENTRIES below), so a root phrase like
# "indemnification" also matches real compound headings that CONTAIN it
# as a whole word/phrase — "Indemnification and Hold Harmless", "Mutual
# Indemnification Obligations" — without needing to enumerate every
# compound form separately, while NOT matching as a fragment inside an
# unrelated word (e.g. "venue" no longer matches inside "revenue").
#
# Reasoning per category (own construction, not from the source doc):
#   - Limitation of Liability: the bare category phrase plus its common
#     singular/plural and on/of variants, plus the concrete mechanisms
#     these clauses usually describe (a liability cap, exclusion of
#     consequential/indirect/punitive damages, an aggregate liability
#     ceiling) since real headings often name the mechanism instead of
#     the abstract category ("Liability Cap" rather than "Limitation of
#     Liability"). Both "exclusion of damages" and "damages exclusion"
#     are included since word-boundary matching doesn't add word-order
#     tolerance on its own — a heading using the reversed phrasing needs
#     its own explicit entry.
#   - Indemnification: "indemnification"/"indemnity"/"indemnities" cover
#     the noun-form variants; "hold harmless" is included separately
#     since "Hold Harmless" alone (without the word "indemnif...") is a
#     real, common standalone heading for the same clause type.
#   - Termination: "termination"/"terminate" cover noun and verb forms
#     (neither is a substring of the other, so both are needed);
#     "cancellation"/"cancel" are included as a genuinely common
#     synonym in services/subscription-style agreements.
#   - Governing Law / Jurisdiction: covers both halves of the category
#     name (choice-of-law phrasing, and venue/jurisdiction phrasing)
#     plus "conflict of laws", a standard legal term of art for the same
#     concept. Deliberately does NOT include "arbitration" or "dispute
#     resolution" — those describe a related but distinct clause topic
#     (the *procedure* for resolving disputes) that real contracts often
#     title as its own separate section; conflating it here would
#     misclassify a genuinely different clause type.
#   - Confidentiality: covers "confidentiality" and the common
#     hyphenated/unhyphenated "non-disclosure" variants, plus "trade
#     secret" and "proprietary information"/"confidential information",
#     plus the "materials" variant of each ("confidential materials",
#     "proprietary materials") since real headings sometimes substitute
#     "Materials" for "Information" as the object being protected.
_LEXICON: dict[str, list[str]] = {
    "Limitation of Liability": [
        "limitation of liability",
        "limitations of liability",
        "limitation on liability",
        "limitations on liability",
        "liability limitation",
        "liability cap",
        "cap on liability",
        "exclusion of damages",
        "damages exclusion",
        "exclusion of consequential damages",
        "consequential damages",
        "limitation of damages",
        "limitation of remedies",
        "special or punitive damages",
        "punitive damages",
        "aggregate liability",
    ],
    "Indemnification": [
        "indemnification",
        "indemnity",
        "indemnities",
        "hold harmless",
    ],
    "Termination": [
        "termination",
        "terminate",
        "cancellation",
        "cancel",
    ],
    "Governing Law / Jurisdiction": [
        "governing law",
        "choice of law",
        "conflict of laws",
        "conflicts of law",
        "applicable law",
        "jurisdiction",
        "venue",
        "forum selection",
    ],
    "Confidentiality": [
        "confidentiality",
        "confidential information",
        "confidential materials",
        "non-disclosure",
        "nondisclosure",
        "non disclosure",
        "proprietary information",
        "proprietary materials",
        "trade secret",
    ],
}

# Tie-breaking priority, used only as the FINAL fallback when two
# DIFFERENT categories' matched phrases start at the exact same character
# position in the normalized heading (see classify_heading) — genuinely
# rare, since two different literal phrases can only share a starting
# index if one is a character-for-character prefix of what's at that
# position in the text. Alphabetical, since there's no principled reason
# to prefer one category over another beyond having SOME fixed,
# documented, deterministic order.
_CATEGORY_PRIORITY = {name: i for i, name in enumerate(sorted(CATEGORIES))}

# Word-boundary-bounded compiled pattern per (category, phrase), built
# once at import time. \b on both ends prevents a lexicon phrase from
# matching as a substring of an unrelated word — e.g. "venue" no longer
# matches inside "revenue" or "avenue", and "termination" no longer
# matches inside "determination" (both were confirmed false positives
# with the previous unbounded-substring matching). \b works correctly
# for multi-word phrases too (e.g. "hold harmless") since only the two
# outer edges need a boundary; the phrase's own internal spaces already
# only align with real word breaks.
_PHRASE_ENTRIES: list[tuple[str, str, re.Pattern]] = [
    (category, phrase, re.compile(r"\b" + re.escape(phrase) + r"\b"))
    for category, phrases in _LEXICON.items()
    for phrase in phrases
]

# Strips a single leading numbering/label decoration, e.g. "5.",
# "Section 5:", "ARTICLE V —", "(a)". Not strictly required for matching
# to succeed — matching is substring-based, so "indemnification" would
# still be found inside "5. indemnification" even unstripped — this
# exists to avoid incidental false matches from numbering fragments and
# to keep normalized text clean. The lookahead in the bare-numbering
# branch (before consuming the token) requires a real separator
# character or whitespace to follow, so a heading that merely STARTS
# with a roman-numeral-shaped letter (e.g. "Indemnification" starting
# with "I") is never mistaken for a numbering prefix.
_LEADING_TOKEN = re.compile(
    r"^(?:"
    r"(?:article|section|clause|item)\s+[ivxlcdm\d]+(?:\.\d+)*\s*[.:)\-–—]*"
    r"|\([a-z0-9ivxlcdm]+\)"
    r"|[ivxlcdm\d]+(?:\.\d+)*(?=[.:)\-–—]|\s)[.:)\-–—\s]*"
    r")\s*",
    re.IGNORECASE,
)


def _normalize(text: str) -> str:
    text = text.strip()
    for _ in range(3):
        stripped = _LEADING_TOKEN.sub("", text, count=1).strip()
        if stripped == text:
            break
        text = stripped
    text = text.strip(" .:;,-–—()[]<>")
    text = re.sub(r"\s+", " ", text)
    return text.lower()


def classify_heading(heading_path: str) -> str | None:
    """Match heading_path against the 5-category lexicon.

    Returns the matched category name, or None if nothing matches — None
    is the correct, expected result for an unrelated heading, not an
    error or a default guess.

    Matching is word-boundary-bounded (see _PHRASE_ENTRIES), not raw
    substring matching, so a lexicon phrase never matches as part of an
    unrelated word.

    Tie-breaking: if a heading matches lexicon phrases from more than one
    category (a real possibility with compound headings, e.g.
    "Indemnification and Limitation of Liability" matches both), the
    EARLIEST-STARTING matched phrase in the normalized heading wins — the
    primary topic of a real contract heading is conventionally stated
    first, so position is a better specificity signal than raw phrase
    length. (An earlier version of this function used "longest phrase
    wins" instead; independent review found that rule was actually just
    measuring which category's lexicon happens to contain wordier
    phrases, not genuine topical specificity — confirmed via constructed
    cases like "Notice of Termination Notwithstanding Any Exclusion of
    Consequential Damages...", where the heading is clearly primarily
    about termination, but the old rule picked Limitation of Liability
    purely because that phrase was longer. Earliest-position fixes THAT
    specific bias, but is not bias-free itself; see KNOWN LIMITATIONS
    below.) If two matched phrases from different categories start at the
    exact same character position (only possible if one phrase is a
    literal character-for-character prefix of what's at that position —
    genuinely rare), a fixed alphabetical category order breaks the tie,
    so the result is always deterministic regardless of dict iteration
    order or how many times this is called.

    KNOWN LIMITATIONS (confirmed, accepted for stage-1; not being chased
    with a 4th tie-breaking heuristic -- this is the expected ceiling of
    a heading-pattern heuristic, and M23's embedding-centroid fallback
    exists specifically to catch what stage-1 can't):

    1. Tie-breaking bias -- position-based tie-breaking systematically
       favors whichever category is named FIRST in a compound heading.
       This is wrong on legal-drafting qualifying-clause patterns where
       the earlier-named category is the subordinate/qualifying clause
       and the later-named one is the actual subject, e.g. "Subject to
       the Limitation of Liability set forth in Section 8, this Section
       9 governs Termination" or "Notwithstanding the Termination rights
       above, this Section addresses Indemnification" -- the heading's
       real topic is the second-named category, but earliest-position
       picks the first. Unbounded substring matching, then longest-match,
       then position-based tie-breaking have each fixed one such bias
       while introducing a different one; a 4th tie-break heuristic is
       expected to do the same, not resolve the issue.
    2. Word-order sensitivity -- matching is per-phrase substring
       matching, not word-order-tolerant. "exclusion of damages" and
       "damages exclusion" are separate lexicon entries precisely because
       matching one does not imply the other. Only the specific reversed
       variants explicitly added to the lexicon (e.g. "liability
       limitation" alongside "limitation of liability") are covered; any
       other un-added reordering of a multi-word phrase will not match.
    3. Inflected-form gaps -- word-boundary matching (\\b...\\b, added to
       stop false positives like "venue" inside "revenue") only matches
       whole words, so it misses inflected forms that the OLD unbounded
       matching happened to catch via shared prefixes. E.g. old unbounded
       matching let the root "cancel" match inside "cancellation",
       "canceled", "cancelled", and "cancelable" all as substrings; the
       new word-boundary matching only matches "cancel" and
       "cancellation" (both explicit lexicon entries) and misses
       "canceled" / "cancelled" / "cancelable" since none is a whole-word
       match for any lexicon entry. Not comprehensively addressed -- the
       lexicon covers common noun/verb pairs explicitly (e.g.
       "terminate"/"termination") but not every inflection of every root.
    """
    if not heading_path:
        return None

    normalized = _normalize(heading_path)
    if not normalized:
        return None

    best_category: str | None = None
    best_position: int | None = None
    best_priority = len(CATEGORIES)

    for category, _phrase, pattern in _PHRASE_ENTRIES:
        match = pattern.search(normalized)
        if match is None:
            continue
        position = match.start()
        priority = _CATEGORY_PRIORITY[category]
        if (
            best_position is None
            or position < best_position
            or (position == best_position and priority < best_priority)
        ):
            best_category = category
            best_position = position
            best_priority = priority

    return best_category


def classify_clause(heading_path: str, text: str) -> str | None:
    """The real two-stage classification cascade: stage 1
    (classify_heading(), heading-text pattern matching) first; stage 2
    (centroid_fallback.classify_by_centroid(), embedding-similarity
    matching) ONLY if stage 1 returns None.

    Stage 2 is never reached for a clause stage 1 already classified —
    classify_heading()'s result is returned immediately, short-circuiting
    before stage 2's embed_text() call (a real, live API call) is ever
    made. This is the concrete mechanism, not just documentation, behind
    "a clause with a recognizable heading should never reach stage 2":
    Python's `if ... is not None: return` short-circuits before the line
    calling classify_by_centroid() is ever evaluated.
    """
    category = classify_heading(heading_path)
    if category is not None:
        return category
    return classify_by_centroid(text)
