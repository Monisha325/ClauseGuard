"""M15: a thin Celery wrapper around M13's run_contract_pipeline() --
no pipeline logic lives here. This task's only jobs are: (1) open its
own DB session, since this runs in a separate worker process/container
from the FastAPI app, not sharing the app's SessionLocal in-memory state;
(2) call the existing, already-verified pipeline function unchanged;
(3) guarantee Contract.status always ends up at "complete" or "failed",
never stuck at "processing", regardless of how the pipeline call turns
out; (4) log any failure with enough detail to diagnose it, since a
Celery task's exceptions don't surface anywhere else a human would
naturally look.
"""

import logging
import uuid

from db import SessionLocal
from models.clause import Clause  # noqa: F401 -- registers Clause with Base.metadata
from models.contract import Contract
from models.flagged_clause import FlaggedClause  # noqa: F401 -- registers FlaggedClause with Base.metadata
from models.user import User  # noqa: F401 -- registers User with Base.metadata
from pipeline.run_contract import run_contract_pipeline
from worker.celery_app import celery_app

logger = logging.getLogger("clauseguard.worker.tasks")


def _mark_status(db, contract_id: uuid.UUID, status_value: str) -> None:
    """Update Contract.status, with its OWN error handling separate from
    the pipeline call -- if this fails too (DB down, etc.), that's a
    distinctly logged failure mode rather than an uncaught exception that
    would leave no clue why the contract is stuck at "processing".

    M40: deliberately does NOT touch Contract.current_stage when
    status_value == "failed" in the ordinary case -- run_contract_pipeline()
    already stamped current_stage to whatever real stage was running when
    it raised (see that function's own _set_current_stage() calls), and
    overwriting that here would destroy the exact "how far did it
    genuinely get" signal this milestone exists to expose. The ONE
    narrow exception: if current_stage is still None (the pipeline
    raised before ever reaching its first real stage -- e.g. the
    Contract row itself couldn't be found), there is no real progress to
    preserve, so current_stage is set to "failed" here as a fallback --
    purely so the field is never left at a bare, ambiguous None
    alongside a genuine terminal status="failed".
    """
    try:
        contract = db.query(Contract).filter(Contract.id == contract_id).one_or_none()
        if contract is not None:
            contract.status = status_value
            if status_value == "failed" and contract.current_stage is None:
                contract.current_stage = "failed"
            db.commit()
        else:
            logger.error(
                "process_contract_task: contract_id=%s not found when "
                "trying to record final status=%s",
                contract_id, status_value,
            )
    except Exception:
        logger.exception(
            "process_contract_task: FAILED to record final status=%s for "
            "contract_id=%s -- contract may be stuck at its previous status",
            status_value, contract_id,
        )


@celery_app.task(name="process_contract")
def process_contract_task(contract_id_str: str) -> None:
    contract_id = uuid.UUID(contract_id_str)
    db = SessionLocal()
    try:
        try:
            run_contract_pipeline(contract_id, db)
        except Exception:
            # The pipeline raised something it didn't already handle
            # itself (M13's own per-clause flagging_failed handling
            # doesn't raise -- this is a genuine unhandled failure, e.g.
            # extraction crashing on a corrupt file, a DB error, etc.).
            # Roll back first: an exception mid-pipeline can leave the
            # session in a failed-transaction state, which would make
            # the status-update query below fail too if not cleared.
            db.rollback()
            logger.exception(
                "process_contract_task failed for contract_id=%s", contract_id
            )
            _mark_status(db, contract_id, "failed")
            return

        _mark_status(db, contract_id, "complete")
    finally:
        db.close()
