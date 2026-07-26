"""OTP (one-time passcode) generation, hashing, expiry, and per-email
rate limiting for email verification at signup (M39, see
routes/auth.py).

SECRECY: mirrors the password-hashing discipline routes/auth.py
already established for hashed_password (M2) -- the actual 6-digit
code is NEVER stored in plaintext anywhere, only a bcrypt hash of it
(models/user.py's otp_hash). The plaintext code exists only
transiently, as issue_otp()'s return value, long enough for its one
caller (routes/auth.py) to hand it to services/email.py's
send_otp_email() -- it is never logged and never appears in any API
response.

RANDOMNESS: `secrets`, not `random` -- `random` is a Mersenne Twister,
not cryptographically secure, and a 6-digit code is exactly the kind
of low-entropy secret where a predictable generator would matter.

EXPIRY: 10 minutes. Long enough that real inbox delivery latency (a
few seconds to roughly a minute, in practice, for Brevo's free tier)
never causes a false "expired" complaint, short enough to bound how
long a leaked/intercepted code stays dangerous -- the same tradeoff
auth/jwt.py already makes with JWT_ACCESS_TOKEN_EXPIRE_MINUTES, tuned
here for a human typing a code from an email instead of a session.

VERIFY-ATTEMPT LIMIT: MAX_VERIFY_ATTEMPTS wrong guesses against the
CURRENT code invalidate it early (forces a resend) rather than leaving
it guessable for the rest of its 10-minute window. bcrypt's own
per-check cost already makes brute-forcing a 6-digit space slow
(bcrypt.checkpw is deliberately ~100ms+ per call), but that alone only
protects a single-threaded guesser -- this caps the honest number of
tries any client gets, full stop.

SEND RATE LIMITING (M39 requirement #2): mirrors M36's window+cap
*shape* (middleware/rate_limit.py) conceptually, but deliberately
implemented as plain mutable columns directly on the User row
(otp_last_sent_at/otp_send_count/otp_window_start) rather than a
second append-only event table like M36's own UploadEvent. That extra
table existed in M36 specifically to survive a Contract being deleted
(see models/upload_event.py's own docstring for the concrete exploit
it closes) -- no analogous exploit exists here, because this codebase
has no user-deletion endpoint anywhere (upload_event.py's own
docstring already relies on that same fact: "users are never deleted
anywhere in this codebase"). A User row's own counters therefore
cannot be reset by deleting and recreating "the same" account, so a
plain mutable counter is exactly as tamper-resistant as a separate
table would be here, for far less new schema.

Two independent, per-email limits:
  - OTP_COOLDOWN_SECONDS: minimum gap between two consecutive sends
    (blocks rapid resend-button mashing).
  - OTP_MAX_PER_WINDOW / OTP_WINDOW_MINUTES: a hard cap on total sends
    within a rolling window (blocks slow, patient draining of Brevo's
    shared 300/day quota against one target email).
"""

import secrets
from datetime import datetime, timedelta, timezone

import bcrypt

from models.user import User

OTP_LENGTH = 6
OTP_EXPIRY_MINUTES = 10
MAX_VERIFY_ATTEMPTS = 5

OTP_COOLDOWN_SECONDS = 60
OTP_MAX_PER_WINDOW = 5
OTP_WINDOW_MINUTES = 15

_DIGITS = "0123456789"


class OtpRateLimitedError(Exception):
    """Raised by check_and_record_send() when `user` has requested an
    OTP too recently (cooldown) or too many times in the current
    window (cap). `retry_after_seconds` is always a real, honest
    countdown derived from the actual limit that fired -- callers
    (routes/auth.py) surface it directly rather than a vague message.
    """

    def __init__(self, retry_after_seconds: int):
        self.retry_after_seconds = retry_after_seconds
        super().__init__(f"Too many OTP requests. Try again in {retry_after_seconds}s.")


def generate_otp() -> str:
    return "".join(secrets.choice(_DIGITS) for _ in range(OTP_LENGTH))


def _hash_otp(otp: str) -> str:
    return bcrypt.hashpw(otp.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def _verify_otp_hash(otp: str, otp_hash: str) -> bool:
    return bcrypt.checkpw(otp.encode("utf-8"), otp_hash.encode("utf-8"))


def check_and_record_send(user: User, now: datetime | None = None) -> None:
    """Raises OtpRateLimitedError if issuing a new OTP right now would
    violate either the cooldown or the rolling-window cap for `user`.
    Otherwise updates the rate-limit bookkeeping columns on `user` IN
    PLACE (advancing/resetting the window as needed) and returns
    normally -- caller (routes/auth.py) is responsible for committing
    the session, same convention as issue_otp()/verify_otp() below.

    Deliberately separate from issue_otp(): this is pure gate-keeping
    and bookkeeping, with no knowledge of the OTP's own value or
    expiry, so a caller can check-and-record a send attempt before
    doing any of the (comparatively expensive) hashing work below.
    """
    now = now or datetime.now(timezone.utc)

    if user.otp_last_sent_at is not None:
        elapsed = (now - user.otp_last_sent_at).total_seconds()
        if elapsed < OTP_COOLDOWN_SECONDS:
            raise OtpRateLimitedError(int(OTP_COOLDOWN_SECONDS - elapsed) + 1)

    if user.otp_window_start is None or now - user.otp_window_start >= timedelta(minutes=OTP_WINDOW_MINUTES):
        user.otp_window_start = now
        user.otp_send_count = 0

    if user.otp_send_count >= OTP_MAX_PER_WINDOW:
        window_end = user.otp_window_start + timedelta(minutes=OTP_WINDOW_MINUTES)
        raise OtpRateLimitedError(int((window_end - now).total_seconds()) + 1)

    user.otp_send_count += 1
    user.otp_last_sent_at = now


def issue_otp(user: User, now: datetime | None = None) -> str:
    """Generates a fresh OTP, hashes+stores it on `user` in place (with
    a fresh expiry and a reset attempt counter), and returns the
    PLAINTEXT code so the caller can pass it straight to
    services/email.py's send_otp_email() -- the only place downstream
    of this call the raw code is allowed to travel. Caller commits.
    """
    now = now or datetime.now(timezone.utc)
    otp = generate_otp()
    user.otp_hash = _hash_otp(otp)
    user.otp_expires_at = now + timedelta(minutes=OTP_EXPIRY_MINUTES)
    user.otp_attempt_count = 0
    return otp


def verify_otp(user: User, otp: str, now: datetime | None = None) -> bool:
    """Returns True and marks `user.is_verified = True` (clearing all
    pending-OTP state) if `otp` is correct, unexpired, and under the
    attempt cap. Returns False in every other case -- wrong code,
    expired, exhausted attempts, or no pending OTP at all. Callers MUST
    map every False outcome to the same generic "invalid or expired
    code" response (M39 requirement #4) -- never branch user-visible
    behavior on which specific case this was. Caller commits.
    """
    now = now or datetime.now(timezone.utc)

    if user.otp_hash is None or user.otp_expires_at is None:
        return False

    if now >= user.otp_expires_at:
        _clear_otp(user)
        return False

    if user.otp_attempt_count >= MAX_VERIFY_ATTEMPTS:
        _clear_otp(user)
        return False

    if not _verify_otp_hash(otp, user.otp_hash):
        user.otp_attempt_count += 1
        if user.otp_attempt_count >= MAX_VERIFY_ATTEMPTS:
            _clear_otp(user)
        return False

    user.is_verified = True
    _clear_otp(user)
    return True


def _clear_otp(user: User) -> None:
    user.otp_hash = None
    user.otp_expires_at = None
    user.otp_attempt_count = 0
