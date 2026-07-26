import logging
import uuid

import bcrypt
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr, field_validator
from sqlalchemy.exc import IntegrityError

from auth.dependencies import get_current_user_id
from auth.jwt import create_access_token
from db import SessionLocal
from models.user import User
from services.email import EmailSendError, send_otp_email
from services.otp import OtpRateLimitedError, check_and_record_send, issue_otp, verify_otp

router = APIRouter()

logger = logging.getLogger("clauseguard.routes.auth")

_GENERIC_LOGIN_ERROR = "Invalid email or password"
_UNVERIFIED_LOGIN_ERROR = (
    "Please verify your email before logging in. Check your inbox for "
    "your verification code, or request a new one."
)
# M39 requirement #4: a wrong/expired/nonexistent OTP all look
# IDENTICAL to the caller -- never reveal which case it was.
_GENERIC_OTP_ERROR = "Invalid or expired code"
# Deliberately generic in the OTHER direction too: resend-otp always
# responds this way whether or not the email is registered, so this
# endpoint can't be used to enumerate accounts either.
_GENERIC_RESEND_MESSAGE = "If that email is registered and not yet verified, a new code has been sent."


class SignupRequest(BaseModel):
    email: EmailStr
    password: str

    @field_validator("password")
    @classmethod
    def password_min_length(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        return v


class SignupResponse(BaseModel):
    id: uuid.UUID
    email: EmailStr
    message: str = "Account created. Check your email for a 6-digit verification code."


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class MeResponse(BaseModel):
    user_id: uuid.UUID


class VerifyOtpRequest(BaseModel):
    email: EmailStr
    otp: str


class VerifyOtpResponse(BaseModel):
    message: str = "Email verified. You can now log in."


class ResendOtpRequest(BaseModel):
    email: EmailStr


class ResendOtpResponse(BaseModel):
    message: str = _GENERIC_RESEND_MESSAGE


def _hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def _verify_password(password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), hashed_password.encode("utf-8"))


def _issue_and_send_otp(db, user: User) -> None:
    """Shared by signup and resend-otp: runs the rate-limit gate, issues
    a fresh OTP, sends it via Brevo, and commits -- in that order, so a
    Brevo failure never leaves a silently-committed OTP the user was
    never actually told (see the rollback below).
    """
    check_and_record_send(user)
    otp = issue_otp(user)
    try:
        send_otp_email(user.email, otp)
    except EmailSendError:
        db.rollback()
        raise
    db.commit()


@router.post("/signup", response_model=SignupResponse, status_code=status.HTTP_201_CREATED)
def signup(payload: SignupRequest):
    db = SessionLocal()
    try:
        user = User(email=payload.email, hashed_password=_hash_password(payload.password), is_verified=False)
        db.add(user)
        try:
            db.flush()
        except IntegrityError:
            db.rollback()
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered")

        try:
            _issue_and_send_otp(db, user)
        except EmailSendError as exc:
            # The account row itself was never committed (flush only),
            # and _issue_and_send_otp already rolled back on failure --
            # signup fails outright rather than leaving an account with
            # no way to ever receive a code.
            logger.error("Signup OTP email failed for %s: %s", payload.email, exc)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Account created but the verification email could not be sent. Please try signing up again shortly.",
            )

        db.refresh(user)
        return SignupResponse(id=user.id, email=user.email)
    finally:
        db.close()


@router.post("/verify-otp", response_model=VerifyOtpResponse)
def verify_otp_route(payload: VerifyOtpRequest):
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email == payload.email).first()
        # No user, wrong code, expired code, exhausted attempts -- ALL
        # of these collapse to the same False / same HTTP response.
        if user is None or not verify_otp(user, payload.otp):
            if user is not None:
                db.commit()  # persist an incremented otp_attempt_count / clear on expiry
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=_GENERIC_OTP_ERROR)
        db.commit()
        return VerifyOtpResponse()
    finally:
        db.close()


@router.post("/resend-otp", response_model=ResendOtpResponse)
def resend_otp(payload: ResendOtpRequest):
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email == payload.email).first()
        if user is None or user.is_verified:
            # Same generic response either way -- don't reveal whether
            # the email exists or is already verified.
            return ResendOtpResponse()

        try:
            _issue_and_send_otp(db, user)
        except OtpRateLimitedError as exc:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Too many verification code requests. Try again in {exc.retry_after_seconds} seconds.",
            )
        except EmailSendError as exc:
            logger.error("Resend-OTP email failed for %s: %s", payload.email, exc)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="The verification email could not be sent. Please try again shortly.",
            )
        return ResendOtpResponse()
    finally:
        db.close()


@router.post("/login", response_model=LoginResponse)
def login(payload: LoginRequest):
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email == payload.email).first()
        if user is None or not _verify_password(payload.password, user.hashed_password):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=_GENERIC_LOGIN_ERROR)
        if not user.is_verified:
            # Only reached once the password has already been confirmed
            # correct, so this doesn't add a new email-enumeration
            # vector on top of the generic 401 above (M39 requirement
            # #6 explicitly wants a SPECIFIC message here, unlike the
            # deliberately-generic login/OTP errors elsewhere).
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=_UNVERIFIED_LOGIN_ERROR)
        return LoginResponse(access_token=create_access_token(user.id))
    finally:
        db.close()


@router.get("/me", response_model=MeResponse)
def get_me(user_id: uuid.UUID = Depends(get_current_user_id)):
    """Minimal protected route proving get_current_user_id works end-to-end."""
    return MeResponse(user_id=user_id)
