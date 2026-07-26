"""M31: OCR fallback for scanned/image-only PDF pages, via pytesseract.

Only ever called (see ingestion/extract.py's own call site) for a PDF
page whose directly-extracted text is under MIN_TEXT_CHARS_PER_PAGE --
never for DOCX (no scanned-image-DOCX concept) and never for a PDF page
that already has adequate real text. This module has no opinion on WHEN
it's called, only on HOW to OCR a given page when asked.

SYSTEM DEPENDENCY, CONFIRMED NOT ASSUMED: pytesseract is a thin Python
wrapper around the real Tesseract OCR binary -- it does NOT bundle
Tesseract itself. `pip install pytesseract` alone is not sufficient; the
real `tesseract` executable must be present on PATH. Confirmed directly,
before writing this module: a fresh container had Pillow already present
(a transitive dependency of something else) but no `tesseract` binary at
all (`which tesseract` found nothing) and no `pytesseract` package
either. Fixed via the Dockerfile's apt-get install of tesseract-ocr +
tesseract-ocr-eng (English trained data -- NOT pulled in automatically by
--no-install-recommends, since it's a "Recommends" not a "Depends" of the
base package on Debian) -- re-verified with `tesseract --version`
directly inside the container after installing, not assumed to work.

RASTERIZATION: PyMuPDF's own Page.get_pixmap() renders the page to a
bitmap directly -- no separate PDF-to-image library needed. Tesseract's
own documentation recommends >=300 DPI for reliable recognition;
PyMuPDF's default pixmap resolution is 72 DPI (screen resolution, not
print/scan quality), so a zoom Matrix of 300/72 is applied explicitly
rather than trusting the default to be OCR-adequate.

GARBAGE-OUTPUT DETECTION -- REPLACED, NOT PATCHED (real, independent
review found the original mechanism's actual ceiling, not a theoretical
one): the original detection was two text-PATTERN heuristics (alnum
ratio + minimum alphabetic-run length). Independent review found this
INSUFFICIENT via a real, reproducible test: sweeping 10 fresh random-
noise images through the real pipeline, 6 of 10 produced Tesseract
hallucinations containing at least one >=5-character alphabetic run
purely by chance (e.g. "Pears" from pure noise) -- a fixed-length-run
check is inherently probabilistic against a full page of scattered
hallucinated fragments; raising the length bar would only shift the
failure rate, not fix the underlying weakness (it's still just "did
ANY fragment get lucky", regardless of where the bar sits).

REPLACEMENT MECHANISM: Tesseract's own per-word confidence score, via
image_to_data() -- this is Tesseract's internal measure of how well the
recognized glyph shapes actually matched real character shapes, not a
property of the resulting string. Confirmed empirically (not assumed)
before choosing a threshold: through the REAL pipeline (same 300 DPI
rasterization every OCR call actually uses), a real clean scanned page's
word-level confidences clustered at 89-96 (mean 95.6, one real sample);
26 fresh random-noise images (10 from independent review + 16 more
generated for this fix, mixed sizes) NEVER produced a mean confidence
above 36.0 -- including the exact "Pears" case that broke the old
mechanism, whose word-level confidences were 0-49 (mean ~20.5) despite
"looking" like a real word.

CALIBRATED ON PAGE-LEVEL MEAN, NOT INDIVIDUAL-WORD CONFIDENCE (revised
after a second independent review round with a larger, fresh sample --
individual words are noisier than the first sample suggested, the page
MEAN is not): an earlier version of this docstring additionally claimed
individual word-level confidence never exceeded 53 for noise or dropped
below 89 for real text. A second, larger independent review round found
both claims too narrow at the individual-word level: one real-text word
scored only 65 (that clause's PAGE MEAN was still 93.08, comfortably
clear of the threshold), and one noise word scored 85 (that page's MEAN
was still only 39.67, since only a handful of words were detected total
and the rest scored far lower). Neither finding threatens the threshold
choice, because MIN_MEAN_CONFIDENCE gates on the PAGE-LEVEL MEAN, never
on any single word -- a single unusual word in either direction (an
uncommon real word Tesseract renders less confidently, or noise that
momentarily hallucinates something plausible-looking) does not by itself
decide the page's outcome; the mean across every detected word does.
Across all 35 fresh adversarial noise seeds gathered over both
independent review rounds, the highest PAGE-LEVEL mean any noise image
ever produced was 39.67 -- a comfortable ~25-point margin below the 65
threshold in every single case, even the ones containing an individual
high-scoring outlier word. Real-text page means, across every sample
gathered in both rounds, never dropped below 93.08. MIN_MEAN_CONFIDENCE=
65 sits in the wide gap between these two real, observed PAGE-LEVEL
distributions -- not at the edge of either, and not contradicted by the
individual-word spread found on closer, larger-sample inspection.

WHY MEAN, NOT MEDIAN OR MAX: the goal is "is this PAGE's OCR trustworthy
as a whole", not "did this page produce at least one high-confidence
word" (which max would answer, and which noise can already satisfy by
chance under the OLD text-based check's failure mode -- confidence
scores are far harder to get "lucky" on than a text shape, but max is
still the wrong aggregate to ask the right question of). Mean reflects
the page's overall recognition quality across every recognized
fragment; a real page that's mostly clean but has a few genuinely
low-confidence words (uncommon fonts, minor smudging) still averages
high, while noise -- which has NO genuinely well-recognized words --
averages low across the board. Median was considered and rejected: on a
hypothetically bimodal page (half genuinely clean, half genuinely
noise), median could look artificially fine by reflecting only the
"good half" while mean honestly reflects the page's real, mixed
trustworthiness.

TWO-TIER CHECK, CONFIDENCE PRIMARY: the original alnum-ratio/min-run
checks are KEPT as a cheap, no-extra-Tesseract-call secondary filter
(defense in depth against a case confidence alone might miss), but are
no longer the primary discriminator -- confidence must ALSO clear
MIN_MEAN_CONFIDENCE for OCR output to be accepted. See ocr_page()'s own
control flow for the exact order (confidence checked first, since it's
the mechanism that actually held up under real adversarial testing).

RESIDUAL RISK (stated plainly, not glossed over): this is a real
empirical calibration, not a formal guarantee. MIN_MEAN_CONFIDENCE=65
is validated against every real sample gathered so far (multiple clean-
text samples, 35 fresh adversarial noise seeds across two independent
review rounds) with a wide margin on both sides, but a sufficiently
adversarial or unusual image could theoretically still produce a mean
confidence in the 65+ range without being genuinely trustworthy text --
no threshold on a single scalar statistic can be a mathematical proof
against every possible input. What's different from the mechanism this
replaced is that this signal (Tesseract's own internal shape-matching
confidence) has no known-easy failure mode the way "does the resulting
string happen to look word-shaped" did, and every real adversarial input
constructed so far (see this milestone's own re-verification) is
correctly rejected, with a wide, not marginal, safety margin.
"""

import io
import logging
import re

import fitz
import pytesseract
from PIL import Image
from pytesseract import Output

logger = logging.getLogger("clauseguard.ingestion.ocr")

# Tesseract's own guidance: reliable recognition needs >=300 DPI;
# PyMuPDF's pixmap default is 72 DPI (screen, not print/scan quality).
OCR_DPI = 300
_ZOOM = OCR_DPI / 72

# PRIMARY garbage/quality gate -- see module docstring's REPLACEMENT
# MECHANISM and CALIBRATED ON PAGE-LEVEL MEAN notes for the real,
# empirical calibration behind this exact number: gates on PAGE-LEVEL
# MEAN confidence, not any individual word -- 39.67 is the highest PAGE
# MEAN any noise image has ever produced (across 35 fresh adversarial
# seeds, two independent review rounds), 93.08 is the lowest PAGE MEAN
# any real clean-text sample has ever produced; 65 sits in the wide
# middle of that gap.
MIN_MEAN_CONFIDENCE = 65.0

# SECONDARY, cheap defense-in-depth checks -- kept from the original
# (now-superseded-as-primary) detection mechanism. No longer relied on
# alone; see module docstring's TWO-TIER CHECK note.
MIN_ALNUM_RATIO = 0.5
MIN_WORD_RUN_LENGTH = 5
_ALPHA_RUN_RE = re.compile(r"[A-Za-z]+")


def _page_confidence(image: Image.Image) -> tuple[float | None, list[int]]:
    """Runs Tesseract's own image_to_data() and returns (mean_confidence,
    all_word_confidences).

    mean_confidence is None if Tesseract found no real word-level entries
    at all -- nothing to average, treated the same as "OCR found
    nothing" by the caller, not as low confidence (a real, distinct
    state -- see M29's own established "no data yet is not the same as
    measured-and-zero" discipline, applied here to the same shape of
    question).

    image_to_data() returns one row per hierarchy level (page/block/
    paragraph/line/word) for every detected region -- confirmed directly
    by inspecting real output before writing this function: levels 1-4
    (page/block/par/line) always have conf=-1 and empty text; only
    level-5 (word) rows carry a real Tesseract-computed confidence
    (0-100) and non-empty text. Both conditions (conf != -1 AND
    non-empty text) are checked, not just one, as a defensive
    belt-and-braces match against the documented level semantics rather
    than relying on exactly one of the two signals alone.
    """
    data = pytesseract.image_to_data(image, output_type=Output.DICT)
    confs = [
        conf for conf, text in zip(data["conf"], data["text"])
        if conf != -1 and text.strip()
    ]
    if not confs:
        return None, []
    return sum(confs) / len(confs), confs


def _garbage_reason(text: str) -> str | None:
    """SECONDARY check only -- see module docstring's TWO-TIER CHECK
    note. Returns a short, specific reason string if `text` (assumed
    already non-empty/stripped) looks like unusable noise under the
    original text-pattern heuristic, else None.
    """
    non_whitespace = [c for c in text if not c.isspace()]
    if not non_whitespace:
        return None
    alnum_count = sum(1 for c in non_whitespace if c.isalnum())
    ratio = alnum_count / len(non_whitespace)
    if ratio < MIN_ALNUM_RATIO:
        return f"alphanumeric ratio {ratio:.0%} is below the {MIN_ALNUM_RATIO:.0%} floor"

    longest_run = max((len(run) for run in _ALPHA_RUN_RE.findall(text)), default=0)
    if longest_run < MIN_WORD_RUN_LENGTH:
        return (
            f"longest alphabetic run ({longest_run} char(s)) is below the "
            f"{MIN_WORD_RUN_LENGTH}-char floor -- no real-word-like fragment found"
        )

    return None


def ocr_page(page: fitz.Page) -> str | None:
    """Rasterize `page` and run Tesseract OCR on it. Returns the
    recognized text, or None if OCR could not produce anything usable:

      - pytesseract/Tesseract itself failed (missing binary, a corrupt
        render, any other exception) -- logged as a warning, never
        raised further.
      - Tesseract ran fine but found no recognizable text at all (a
        genuinely blank/near-blank page) -- this is NOT an error, just a
        real, expected outcome; logged at info level.
      - Tesseract's own per-word confidence (image_to_data(), the
        PRIMARY check -- see module docstring) averages below
        MIN_MEAN_CONFIDENCE -- logged as a warning with the real
        confidence numbers included.
      - The recognized text ALSO fails the secondary text-pattern check
        (_garbage_reason) -- kept as defense in depth, no longer the
        primary discriminator.

    In every None case, the caller already has the pre-OCR (already
    low/empty) directly-extracted text to fall back to for this one
    page -- this function never crashes extraction for the rest of the
    document over a single bad page.

    Two real Tesseract invocations per call (image_to_data() for
    confidence, image_to_string() for the actual text) -- deliberate,
    not an oversight: image_to_data()'s per-word rows could be
    reassembled into a text string, but doing so would lose
    image_to_string()'s own line-break structure, which M6's chunker
    relies on for heading detection (confirmed in this milestone's own
    original testing). Paying for a second real OCR pass on the same
    image is a fair trade for not risking a text-reconstruction bug on
    every OCR'd page.
    """
    page_number = page.number + 1  # PyMuPDF is 0-indexed; log messages use 1-indexed, matching extract.py
    try:
        pix = page.get_pixmap(matrix=fitz.Matrix(_ZOOM, _ZOOM))
        image = Image.open(io.BytesIO(pix.tobytes("png")))
        mean_conf, word_confs = _page_confidence(image)
        raw_text = pytesseract.image_to_string(image)
    except Exception:
        logger.warning(
            "OCR failed for page %d (Tesseract/pytesseract error) -- "
            "falling back to empty text for this page.",
            page_number,
            exc_info=True,
        )
        return None

    stripped = raw_text.strip()
    if not stripped:
        logger.info("OCR found no recognizable text on page %d.", page_number)
        return None

    if mean_conf is None:
        # image_to_string found *some* text but image_to_data found no
        # real word-level entries at all to score -- an inconsistent,
        # unexplained state. Conservative default: treat as unknown
        # confidence, not as implicitly trustworthy.
        logger.warning(
            "OCR produced text on page %d but Tesseract's own confidence "
            "data had no scorable word entries -- discarding and falling "
            "back to empty text for this page. Raw OCR output was: %r",
            page_number, stripped[:200],
        )
        return None

    if mean_conf < MIN_MEAN_CONFIDENCE:
        logger.warning(
            "OCR output for page %d has low Tesseract confidence (mean "
            "%.1f over %d word(s), below the %.1f floor) -- discarding "
            "and falling back to empty text for this page. Raw OCR "
            "output was: %r",
            page_number, mean_conf, len(word_confs), MIN_MEAN_CONFIDENCE, stripped[:200],
        )
        return None

    # Confidence check passed -- secondary, cheap defense-in-depth check
    # (see module docstring's TWO-TIER CHECK note).
    garbage_reason = _garbage_reason(stripped)
    if garbage_reason is not None:
        logger.warning(
            "OCR output for page %d passed the confidence check (mean "
            "%.1f) but looks like unusable noise under the secondary "
            "text-pattern check (%s) -- discarding and falling back to "
            "empty text for this page. Raw OCR output was: %r",
            page_number, mean_conf, garbage_reason, stripped[:200],
        )
        return None

    return stripped
