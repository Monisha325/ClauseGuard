import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, DateTime, Integer, String
from sqlalchemy.dialects.postgresql import UUID

from db import Base


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # M39: email-based OTP verification at signup. A freshly-created
    # account starts unverified and cannot log in (see routes/auth.py's
    # login()) until it completes services/otp.py's verify flow.
    is_verified = Column(Boolean, default=False, nullable=False)

    # The CURRENT pending OTP, stored only as a bcrypt hash -- same
    # secrecy discipline as hashed_password above; the plaintext code is
    # never persisted anywhere (see services/otp.py's own docstring).
    # All four columns below are nulled/reset together on every fresh
    # issue_otp() call and cleared on successful verification or expiry.
    otp_hash = Column(String, nullable=True)
    otp_expires_at = Column(DateTime(timezone=True), nullable=True)
    # Failed verify_otp() attempts against the CURRENT otp_hash. Hitting
    # services/otp.py's MAX_VERIFY_ATTEMPTS invalidates the code early
    # (forces a resend) rather than leaving a bounded-but-nonzero window
    # for a fast, parallelized guesser to exploit for the rest of the
    # 10-minute expiry.
    otp_attempt_count = Column(Integer, default=0, nullable=False)

    # Per-email OTP send rate limiting (services/otp.py's
    # check_and_record_send) -- deliberately plain mutable columns on
    # this row rather than a second append-only event table like M36's
    # UploadEvent, since no user-deletion endpoint exists anywhere in
    # this codebase to reset them via delete-and-recreate (see
    # models/upload_event.py's own docstring, which already relies on
    # that same fact). otp_window_start anchors a rolling window;
    # otp_send_count counts sends inside it; otp_last_sent_at enforces a
    # short cooldown between individual requests.
    otp_last_sent_at = Column(DateTime(timezone=True), nullable=True)
    otp_send_count = Column(Integer, default=0, nullable=False)
    otp_window_start = Column(DateTime(timezone=True), nullable=True)
