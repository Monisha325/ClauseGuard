"""M36: an INTERNAL, app-level per-user daily contract-upload cap --
protecting against runaway cost/quota consumption from a single user
hammering THIS app's own upload endpoint. Deliberately a DIFFERENT,
additional layer from the EXTERNAL Groq/Voyage provider-side quotas
this project has repeatedly hit (M9/M10/M27's own retry/backoff logic
around 429s from those providers) -- this module has no opinion about,
and does not touch, provider-side quota state at all. It only answers
one question: has THIS user already uploaded too many contracts today,
according to OUR OWN app-level bookkeeping?

DEFINING "DAILY" -- explicitly flagged as a known risk by this
milestone's own spec, so the choice and reasoning are spelled out here
rather than left ambiguous:

Chosen: a fixed CALENDAR-DAY boundary, midnight UTC to midnight UTC --
NOT a rolling 24-hour window from each upload.

A rolling window is arguably the more "correct" definition (it can't be
gamed by uploading just before AND just after a fixed reset instant,
effectively doubling up around the boundary) and isn't meaningfully
harder to QUERY (`created_at >= now - 24h` is no more complex than
`created_at >= midnight_utc`). But it loses on the one thing this
milestone's own behavior requirement #4 explicitly demands: a
COMMUNICABLE, FIXED reset time in the 429 response. Under a rolling
window, "when does my limit reset" has no single answer -- it depends on
exactly when each of the user's own uploads in the trailing 24h window
individually ages out, a continuously shifting target that would require
tracking and exposing the user's own upload history just to answer "when
can I try again", and produces a confusing, constantly-changing reset
time even for a user doing nothing at all in the meantime. A calendar-day
boundary gives a single, fixed, honest answer -- "resets at
<next midnight UTC>" -- that doesn't change no matter when during the
day the user checks, which is a real, deliberate simplicity win over the
rolling window's marginal anti-gaming benefit. UTC (not a per-user local
timezone) is chosen because this app has no stored per-user timezone
anywhere to base a "local midnight" on, and inventing one (e.g. from
request IP geolocation) would be new, unrequested scope with its own
real failure modes (VPNs, proxies, travel) -- UTC is a real, unambiguous,
already-used-elsewhere-in-this-codebase (every `datetime.now(timezone.utc)`
call in every model's created_at) convention, not a new one.

Known, accepted consequence of this choice (not hidden): a user who
uploads at 23:59 UTC and again at 00:01 UTC gets 2 uploads in under 3
minutes of wall-clock time -- a real gaming vector at the exact boundary,
strictly worse than what a rolling window would allow. Accepted here
because the milestone's own spec explicitly prioritizes a clear,
communicable reset time (behavior requirement #4) and flags calendar-day
as an allowed choice ("your call, but be explicit") -- this is a
deliberate trade, not an oversight.

COUNTING PER USER, NOT GLOBALLY -- explicitly flagged as a known risk by
this milestone's own spec: check_daily_upload_limit() below ALWAYS
filters Contract.user_id == the authenticated caller's own user_id
(never a bare, unfiltered count across all users) -- see this module's
own test coverage for explicit confirmation that a different user's
upload volume has zero effect on this one.

WHAT COUNTS AS "AN UPLOAD" FOR THIS LIMIT (M36 behavior requirement #6):
every UploadEvent row that ever gets created for this user within the
window counts, REGARDLESS of that upload's resulting contract's eventual
status ("uploaded"/"processing"/"complete"/"failed", including a
pipeline-side failure like flagging_failed rows inside it) -- and,
critically, REGARDLESS of whether that contract has since been deleted
entirely (see the M36-FIX section below). Reasoning: this check runs
BEFORE routes/contracts.py's existing M32 file validation and BEFORE the
Celery task is even enqueued (see that route's own docstring) -- an
UploadEvent row is only ever created AFTER a real upload has already
passed M32's real byte-level size/MIME validation, at the exact same
point a Contract row is created (see that route's own docstring for
exactly where and why). A file that fails M32 validation (wrong type,
too large) never gets an UploadEvent row at all and so can never be
counted here -- exactly "only genuinely-accepted uploads count", with no
extra status filtering needed to achieve that. Conversely, a contract
that LATER fails somewhere in the pipeline (extraction error, Groq
flagging_failed, etc.) already consumed real downstream cost (storage, a
Celery task, likely real Voyage/Groq calls) by the time that failure
happens -- not counting it here would let a user retry an unlimited
number of times per day for contracts that happen to fail processing,
which is exactly the "runaway cost" scenario this milestone exists to
prevent, not a fairness carve-out worth adding.

M36-FIX (post-M37 holistic review -- a real, confirmed composition bug,
not a hypothetical): counting now happens against models/upload_event.py's
UploadEvent table, NOT live Contract rows as originally implemented.
M37 gave users a real DELETE endpoint that permanently removes Contract
rows (and cascades models/audit_log.py's own contract_id-scoped rows with
it, by that milestone's own deliberate design). Composed with the ORIGINAL
version of this check (which counted `Contract` rows directly), this was
a real, demonstrated exploit: upload a contract (spending real Groq/
Voyage quota the instant the pipeline ran), immediately DELETE it, and
the live Contract-row count drops back down -- freeing the just-spent
quota slot for another real upload, with zero surviving record (Contract
gone, audit_log cascaded away too) to even detect the pattern. UploadEvent
rows carry no foreign key to contracts.id at all (see that model's own
docstring for why) specifically so they are structurally incapable of
being cascaded away by a Contract deletion -- the ONLY correct behavior
for a counter whose entire job is "did this user cause real cost today",
a fact that must stay true forever once true, independent of whatever
the user does to the resulting contract afterward.
"""

import logging
import uuid
from datetime import datetime, timedelta, timezone

from config import settings
from models.upload_event import UploadEvent

logger = logging.getLogger("clauseguard.middleware.rate_limit")


class DailyUploadLimitExceededError(Exception):
    """Raised by check_daily_upload_limit() when `user_id` has already
    reached settings.daily_upload_limit_per_user contracts for the
    current UTC calendar day. Callers (routes/contracts.py) catch this
    and convert it into a specific 429 response -- never a generic error
    -- using `limit`/`reset_at` below to build that message, so the
    exact same numbers backing the exception are what the user sees.
    """

    def __init__(self, limit: int, reset_at: datetime):
        self.limit = limit
        self.reset_at = reset_at
        super().__init__(
            f"Daily upload limit of {limit} contract(s) reached. "
            f"Resets at {reset_at.isoformat()}."
        )


def _start_of_today_utc(now: datetime | None = None) -> datetime:
    """The current UTC calendar day's start (00:00:00.000000 UTC) --
    `now` is an injectable parameter purely so this milestone's own
    boundary-condition test can pass a fixed instant rather than
    depending on real wall-clock time to exercise a specific moment near
    midnight; every real call site below omits it and gets the genuine
    current time.
    """
    now = now if now is not None else datetime.now(timezone.utc)
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


def check_daily_upload_limit(user_id: uuid.UUID, db_session, now: datetime | None = None) -> None:
    """Raises DailyUploadLimitExceededError if `user_id` has already
    reached the configured daily limit for the current UTC calendar day
    (see this module's own docstring for the full "daily"/"per-user"/
    "what counts" reasoning). Returns None (no exception) if the user is
    still under the limit -- callers proceed exactly as before.

    Deliberately a single, cheap COUNT query against real UploadEvent rows
    (UploadEvent.user_id indexed -- see models/upload_event.py) -- this
    table is written ONCE, at upload-acceptance time, and never updated
    or deleted by anything else in this codebase (in particular, NOT by
    M37's DELETE endpoint -- see this module's own docstring, M36-FIX,
    for why that's the entire point).
    """
    limit = settings.daily_upload_limit_per_user
    window_start = _start_of_today_utc(now)
    window_end = window_start + timedelta(days=1)

    count = (
        db_session.query(UploadEvent)
        .filter(UploadEvent.user_id == user_id, UploadEvent.created_at >= window_start)
        .count()
    )

    if count >= limit:
        logger.warning(
            "Daily upload limit reached for user_id=%s: %d/%d contracts "
            "already uploaded since %s (resets %s).",
            user_id, count, limit, window_start.isoformat(), window_end.isoformat(),
        )
        raise DailyUploadLimitExceededError(limit, window_end)

    logger.info(
        "Daily upload limit check passed for user_id=%s: %d/%d contracts "
        "uploaded since %s.",
        user_id, count, limit, window_start.isoformat(),
    )
