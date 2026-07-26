import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from db import check_db_connection, init_db, wait_for_db
from routes.auth import router as auth_router
from routes.contracts import router as contracts_router
from routes.retrieval import router as retrieval_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("clauseguard.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    wait_for_db()
    init_db()
    yield


app = FastAPI(title="ClauseGuard", lifespan=lifespan)

# M14: the frontend is a real browser app (Vite dev server), not curl --
# curl never enforces CORS, so this was invisible until now. Vite's
# default dev port (5173) was already held by other processes on this
# machine (Docker Desktop/WSL relay), so the dev server is pinned to
# 5174 instead (see frontend/vite.config.js) -- listed explicitly here
# rather than a wildcard so credentials (the Authorization header) can
# be sent.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(contracts_router)
app.include_router(retrieval_router)


@app.get("/health")
def health():
    if check_db_connection():
        return {"status": "ok", "db": "connected"}

    return JSONResponse(
        status_code=503,
        content={"status": "error", "db": "disconnected"},
    )
