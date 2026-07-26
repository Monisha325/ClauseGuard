import uuid

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from auth.jwt import decode_access_token

_AUTH_ERROR_DETAIL = "Not authenticated"

_bearer_scheme = HTTPBearer(auto_error=False)


def get_current_user_id(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> uuid.UUID:
    """FastAPI dependency: validates the Bearer token and returns the caller's user_id.

    Every failure path (missing header, malformed header, bad signature,
    expired token, malformed claims) raises the same generic 401 so callers
    can't distinguish *why* auth failed.
    """
    if credentials is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=_AUTH_ERROR_DETAIL)

    try:
        payload = decode_access_token(credentials.credentials)
        return uuid.UUID(payload["user_id"])
    except (jwt.PyJWTError, KeyError, ValueError):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=_AUTH_ERROR_DETAIL)
