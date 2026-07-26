import logging
import time

from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import declarative_base, sessionmaker

from config import settings

logger = logging.getLogger("clauseguard.db")

engine = create_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def wait_for_db() -> None:
    """Verify DB connectivity at startup with retry-with-backoff.

    Docker Compose only guarantees container start order, not Postgres
    readiness, so the first few connection attempts are expected to fail
    right after `docker-compose up`.
    """
    attempts = settings.db_connect_retries
    delay = settings.db_connect_retry_delay_seconds

    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Connected to DB on attempt %d/%d", attempt, attempts)
            return
        except OperationalError as exc:
            last_error = exc
            logger.warning(
                "DB not ready (attempt %d/%d): %s", attempt, attempts, exc
            )
            if attempt < attempts:
                time.sleep(delay)

    raise RuntimeError(
        f"Could not connect to database after {attempts} attempts"
    ) from last_error


def check_db_connection() -> bool:
    """Run a trivial query to confirm the DB is currently reachable."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except OperationalError:
        return False


def init_db() -> None:
    """Create tables for all registered models (M1 had none; M2 adds User; M4 adds Contract; M7 adds Clause; M13 adds FlaggedClause; M35 adds AuditLog; M36-FIX adds UploadEvent)."""
    import models.audit_log  # noqa: F401 — registers AuditLog with Base.metadata
    import models.clause  # noqa: F401 — registers Clause with Base.metadata
    import models.contract  # noqa: F401 — registers Contract with Base.metadata
    import models.flagged_clause  # noqa: F401 — registers FlaggedClause with Base.metadata
    import models.upload_event  # noqa: F401 — registers UploadEvent with Base.metadata
    import models.user  # noqa: F401 — registers User with Base.metadata

    Base.metadata.create_all(bind=engine)
    _migrate_user_otp_columns()
    _migrate_contract_stage_column()


def _migrate_user_otp_columns() -> None:
    """M39: this project has no migration tool (no Alembic anywhere --
    every prior schema addition was a brand-new TABLE, which
    create_all() above handles on its own). M39 is the first milestone
    to add COLUMNS to an already-existing table (`users`, live since
    M2), which create_all() deliberately never alters for an existing
    table. Rather than introduce a whole migrations framework for one
    milestone, this runs a small set of idempotent
    `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` statements at every
    startup -- safe to run against a fresh DB (columns already exist
    from create_all() above, so every ADD is a no-op) and against a
    pre-M39 DB with real existing user rows (this machine's own dev DB
    has both, from prior milestones' real testing).
    """
    statements = [
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS is_verified BOOLEAN NOT NULL DEFAULT FALSE",
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS otp_hash VARCHAR",
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS otp_expires_at TIMESTAMPTZ",
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS otp_attempt_count INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS otp_last_sent_at TIMESTAMPTZ",
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS otp_send_count INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS otp_window_start TIMESTAMPTZ",
    ]
    with engine.begin() as conn:
        for stmt in statements:
            conn.execute(text(stmt))
    logger.info("User OTP columns verified/migrated.")


def _migrate_contract_stage_column() -> None:
    """M40: same no-Alembic ADD COLUMN IF NOT EXISTS pattern as
    _migrate_user_otp_columns() above -- adding a column to the
    already-existing `contracts` table (live since M4).

    BACKFILL: real historical rows already sitting at a terminal
    status ("complete"/"failed") are backfilled to the SAME value for
    current_stage -- both are genuinely valid terminal current_stage
    values (see models/contract.py's own docstring), and this was
    already the honest, known outcome for those rows; leaving them
    NULL forever would just be a needless gap for old data that has a
    perfectly good real answer. Rows still "processing" from before
    this migration are deliberately left NULL -- which real stage they
    were in is genuinely unknown (that information was never
    recorded), and NULL is the honest way to represent "unknown", not
    a fabricated guess.
    """
    statements = [
        "ALTER TABLE contracts ADD COLUMN IF NOT EXISTS current_stage VARCHAR",
        "UPDATE contracts SET current_stage = status "
        "WHERE current_stage IS NULL AND status IN ('complete', 'failed')",
    ]
    with engine.begin() as conn:
        for stmt in statements:
            conn.execute(text(stmt))
    logger.info("Contract current_stage column verified/migrated.")
