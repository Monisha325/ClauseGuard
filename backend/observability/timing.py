"""M28: reusable timing instrumentation for production code paths.

GOAL: real, persistent, per-call latency measurements for the rerank
step (backend/retrieval/reranker.py) and the end-to-end contract
pipeline (backend/pipeline/run_contract.py), so p50/p95 can be computed
against the architecture doc's stated targets (~180ms/450ms rerank,
2.5s/5s end-to-end) -- not a one-off benchmark script (M21 already did
that once, for truncation-length tuning) but an always-on measurement
that accumulates real data from real runs.

DESIGN: timed_stage() is a context manager that wraps any code block,
measures its wall-clock duration with time.perf_counter() (monotonic,
immune to system-clock adjustments -- NOT time.time(), which a wall-clock
change could corrupt mid-measurement), and logs exactly ONE structured
line per call via the standard `logging` module -- the same mechanism
every other module in this codebase already uses (embeddings/index.py,
agent/flag_clause.py, etc.), so this rides on the existing, already-
captured (docker compose logs) observability path rather than inventing
a new storage mechanism. The logged line's MESSAGE body is a single JSON
object (stage name, duration_ms, a UTC ISO timestamp, and whatever extra
identifying context -- e.g. contract_id -- the caller passes as keyword
arguments) so it is trivially machine-parseable later, not just
human-readable prose.

PER-CALL, NOT A RUNNING AVERAGE: every real call produces its own log
line. This is deliberate -- a single running average (or even a running
mean/stddev) throws away the actual distribution, which is exactly what
p50/p95 need to be computed from real, accumulated per-call
measurements, not synthesized from a summary statistic after the fact.

OVERHEAD: one time.perf_counter() call on each side of the block
(nanosecond-scale), one dict construction, one json.dumps() call, and one
logger.info() call -- all pure in-process Python, no I/O beyond whatever
the logging handler already does for every other log line this codebase
already emits. This is negligible relative to the ~180-450ms rerank
latency or ~2.5-5s end-to-end pipeline latency being measured (verified
concretely, not just assumed, in this milestone's own testing step 5 --
see that comparison of summed per-stage durations against overall
wall-clock time for the same run).

SCOPE DISCIPLINE (the single highest-risk failure mode this milestone's
own spec flags): timed_stage() has NO opinion about what code the caller
puts inside its `with` block -- it is the CALLER's responsibility to wrap
ONLY the code that should count toward that stage's measured latency.
See retrieval/reranker.py's own call site for the specific, deliberate
choice of what is (and is NOT) inside the "rerank_inference" timed
block.
"""

import json
import logging
import math
import re
import time
from contextlib import contextmanager
from datetime import datetime, timezone

logger = logging.getLogger("clauseguard.observability.timing")

# The exact prefix every timed_stage() log line starts with -- used by
# parse_timing_log() below to find and extract timing lines out of a
# larger block of mixed log output (this project's containers log many
# OTHER things too; timing lines must be unambiguously identifiable).
_TIMING_PREFIX = "TIMING "

# Matches a timed_stage() log line's structured JSON body, allowing for
# whatever logging-format prefix (timestamp, logger name, level, etc.)
# a given log line has before the "TIMING " marker -- so this works
# whether reading raw `logger.info(...)` output or a fuller
# `docker compose logs` line with its own added prefix.
_TIMING_LINE_RE = re.compile(re.escape(_TIMING_PREFIX) + r"(\{.*\})\s*$")


@contextmanager
def timed_stage(stage: str, **context):
    """Times the wrapped block and logs exactly one structured line on
    exit (success or exception -- duration is always recorded, even if
    the block raised, since a stage that failed slowly is still real
    timing information, not something to discard).

    stage: a short, fixed name identifying what's being timed (e.g.
    "rerank_inference", "extraction", "end_to_end") -- this is the field
    percentiles get grouped/computed by later.

    **context: any extra identifying key=value pairs worth logging
    alongside the duration (e.g. contract_id=..., n_candidates=...) --
    purely for later correlation/debugging, never used in the timing
    math itself.
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        duration_ms = (time.perf_counter() - start) * 1000.0
        entry = {
            "stage": stage,
            "duration_ms": round(duration_ms, 3),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **context,
        }
        logger.info("%s%s", _TIMING_PREFIX, json.dumps(entry, default=str))


def compute_percentiles(durations_ms: list[float]) -> dict:
    """p50/p95 (plus n/min/max/mean) over a real list of duration_ms
    values, using the simple nearest-rank method (no external stats
    dependency, no interpolation subtleties to get wrong) -- per this
    milestone's own spec, "doesn't need to be fancy". Deterministic given
    the same input data.

    Returns {"n": 0, "p50": None, "p95": None, "min": None, "max": None,
    "mean": None} for an empty input -- explicitly, not a crash or a
    fabricated 0.0, since "no data yet" is a real, distinct state from
    "measured and it's zero".
    """
    if not durations_ms:
        return {"n": 0, "p50": None, "p95": None, "min": None, "max": None, "mean": None}

    data = sorted(durations_ms)
    n = len(data)

    def _percentile(p: float) -> float:
        idx = max(0, min(n - 1, math.ceil(p / 100 * n) - 1))
        return data[idx]

    return {
        "n": n,
        "p50": _percentile(50),
        "p95": _percentile(95),
        "min": data[0],
        "max": data[-1],
        "mean": sum(data) / n,
    }


def parse_timing_log(log_text: str, stage: str | None = None) -> list[float]:
    """Extracts duration_ms values from raw timed_stage() log output --
    the "simple script/function that parses the structured logs and
    computes percentiles" this milestone's spec asks for. Works directly
    against real `docker compose logs` text (each real log line is
    checked independently; non-timing lines are silently skipped, not
    treated as an error, since real log output is always a mix of many
    different messages). If `stage` is given, only entries for that
    exact stage name are returned -- e.g. parse_timing_log(text,
    stage="rerank_inference") to isolate just the rerank measurements
    out of a run that also logged "extraction"/"chunking"/etc. entries.
    """
    durations: list[float] = []
    for line in log_text.splitlines():
        match = _TIMING_LINE_RE.search(line)
        if not match:
            continue
        try:
            entry = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if stage is not None and entry.get("stage") != stage:
            continue
        duration = entry.get("duration_ms")
        if isinstance(duration, (int, float)):
            durations.append(float(duration))
    return durations
