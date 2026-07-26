"""Brevo transactional email client (M39): sends the OTP verification
email at signup via Brevo's REST API
(https://api.brevo.com/v3/smtp/email), not raw SMTP -- the same
"vendor REST API over a lower-level protocol" choice this project
already made for Voyage/Groq (embeddings/voyage_client.py, agent/*),
for the same reason: simpler auth (one header, no SMTP
host/port/credential juggling) and a real JSON response to inspect on
failure instead of an SMTP status line.

API key: read from BREVO_API_KEY at CALL time, not import time --
identical pattern to VOYAGE_API_KEY (embeddings/voyage_client.py's own
docstring) so importing this module never crashes a process that
doesn't need it yet.

SENDER: Brevo rejects sends from an address that isn't a verified
sender on the account (Settings -> Senders & IP), so this is read from
BREVO_SENDER_EMAIL rather than hardcoded -- a fabricated sender domain
would just fail at send time with a real, confusing 400 from Brevo.

SECRECY: this module logs the OUTCOME of a send (recipient, Brevo's
HTTP status) but the OTP code itself is only ever used to build the
email body -- never interpolated into a log line, exception message,
or anything else that could end up in application logs.
"""

import logging
import os

import requests

logger = logging.getLogger("clauseguard.services.email")

BREVO_API_URL = "https://api.brevo.com/v3/smtp/email"
SENDER_NAME = "ClauseGuard"
REQUEST_TIMEOUT_SECONDS = 10


class EmailSendError(Exception):
    """Raised if BREVO_API_KEY/BREVO_SENDER_EMAIL are unset, or the
    Brevo API call fails for any reason (bad key, unverified sender,
    network error, rejected recipient, etc.)."""


def send_otp_email(to_email: str, otp: str) -> None:
    """Send `otp` to `to_email` as a plain, honest verification email.
    Raises EmailSendError on any failure -- callers (routes/auth.py)
    decide how to surface that to the user; this function never
    swallows a failure silently, since a signup that silently fails to
    send its OTP would leave a user stuck with no way to verify.
    """
    api_key = os.getenv("BREVO_API_KEY")
    if not api_key:
        raise EmailSendError(
            "BREVO_API_KEY is not set in the environment. Set it in "
            ".env (see .env.example) before calling send_otp_email()."
        )

    sender_email = os.getenv("BREVO_SENDER_EMAIL")
    if not sender_email:
        raise EmailSendError(
            "BREVO_SENDER_EMAIL is not set in the environment. Set it "
            "to a real, verified sender address from your Brevo "
            "account (Settings -> Senders & IP) in .env (see "
            ".env.example) before calling send_otp_email()."
        )

    payload = {
        "sender": {"name": SENDER_NAME, "email": sender_email},
        "to": [{"email": to_email}],
        "subject": "Your ClauseGuard verification code",
        "htmlContent": (
            "<p>Your ClauseGuard verification code is:</p>"
            f'<p style="font-size:28px;font-weight:700;letter-spacing:6px;">{otp}</p>'
            "<p>This code expires in 10 minutes. If you didn't request "
            "this, you can safely ignore this email.</p>"
        ),
    }

    try:
        response = requests.post(
            BREVO_API_URL,
            json=payload,
            headers={
                "api-key": api_key,
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        raise EmailSendError(f"Brevo API request failed: {exc}") from exc

    if response.status_code >= 300:
        # response.text is Brevo's own error body (e.g. invalid sender,
        # invalid key) -- never contains the OTP, which was never sent
        # to Brevo as a distinct field, only baked into htmlContent
        # above; truncated defensively anyway since it's third-party
        # text ending up in our own logs.
        raise EmailSendError(
            f"Brevo API returned {response.status_code} sending to "
            f"{to_email}: {response.text[:300]}"
        )

    # Logging only the status code here was a real gap: Brevo's success
    # body carries `messageId`, the ONE thing needed to look up this
    # exact send's actual delivery outcome later via Brevo's own
    # /v3/smtp/statistics/events API (a 201 here only means Brevo
    # ACCEPTED the request -- it says nothing about whether the message
    # was ultimately delivered, bounced, or blocked downstream). Logging
    # the full body -- not just messageId -- costs nothing extra
    # (response.text is already in memory) and means the NEXT
    # delivery-mystery doesn't require a fresh reproduction just to get
    # a messageId to query against.
    logger.info(
        "OTP email accepted by Brevo for %s (status=%d): %s",
        to_email, response.status_code, response.text,
    )
