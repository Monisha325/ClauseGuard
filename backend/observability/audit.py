"""M35: write_audit_entry() -- the ONLY way any code in this codebase
writes an AuditLog row (models/audit_log.py). Two deliberate design
choices, both driven directly by this milestone's own explicitly-flagged
risk ("audit-write failures must never crash or interrupt the main
pipeline"):

1. OWN, INDEPENDENT DB SESSION, never the caller's db_session. The real
   pipeline (pipeline/run_contract.py) holds ONE db_session open across
   its entire run -- extraction, chunking, persistence, embedding, the
   flagging loop, and a single final commit of flagged_rows. If audit
   writes shared that session and one failed mid-flush, SQLAlchemy would
   leave the session in a state that requires an explicit rollback
   before ANY further use -- meaning an audit-write failure could
   silently poison the pipeline's own later persist_chunks()/commit()
   calls unless every single call site were made defensively aware of
   audit-write failures too. Opening a fresh SessionLocal() per audit
   entry (commit or rollback, always closed) makes an audit-write
   failure PROVABLY isolated: it can only ever affect its own throwaway
   session, never the caller's.

2. NEVER RAISES. Every exception opening, writing, or committing an
   audit entry is caught here, logged clearly (exc_info=True, so the
   real cause is never silently dropped), and swallowed -- the audit log
   is explicitly a secondary concern (per this milestone's own spec),
   not load-bearing. A lost audit entry is a real gap worth a loud log
   line; it must never be a reason a real clause fails to get flagged.

contract_id/clause_id are the real Contract/Clause UUIDs -- callers must
already have these in scope (pipeline/run_contract.py's own loop always
does; agent/validate.py/agent/flag_clause.py only have them if the
caller threaded them in as optional parameters -- see those modules'
own docstrings for why they default to None and skip the write entirely
rather than writing a row with a null contract_id).
"""

import logging
import uuid

from db import SessionLocal
from models.audit_log import AuditLog

# AuditLog.contract_id/clause_id are real FKs to contracts/clauses -- but
# SQLAlchemy only resolves a ForeignKey against another table once THAT
# table's own model has been imported somewhere in this process (which
# registers it on Base.metadata). Every real caller in this codebase
# (pipeline/run_contract.py, agent/validate.py) already imports
# models.contract/models.clause before ever calling write_audit_entry(),
# but this module has no business depending on caller import order for
# its own correctness -- confirmed the hard way (a bare script that
# imported only observability.audit hit a real
# sqlalchemy.exc.NoReferencedTableError on session.commit()). Importing
# both directly here guarantees write_audit_entry() works regardless of
# what has or hasn't been imported elsewhere first.
import models.clause  # noqa: F401 -- registers Clause with Base.metadata
import models.contract  # noqa: F401 -- registers Contract with Base.metadata

logger = logging.getLogger("clauseguard.observability.audit")

# Event types actually written by this codebase (see pipeline/run_contract.py
# and agent/validate.py for the real call sites). Free-text, not a DB-level
# enum (same convention as FlaggedClause.severity/Contract.status) -- these
# constants exist purely so every call site and every future query spells
# the same event the same way, not to enforce anything at the DB layer.
EVENT_VALIDATION_FAILURE = "validation_failure"
EVENT_VALIDATION_RETRY_SUCCESS = "validation_retry_success"
EVENT_VALIDATION_NEEDS_MANUAL_REVIEW = "validation_needs_manual_review"
EVENT_API_ERROR_RETRY = "api_error_retry"
EVENT_API_ERROR_FAILED = "api_error_failed"
EVENT_GROUNDING_FAILURE = "grounding_failure"
EVENT_GROUNDING_ERROR = "grounding_error"
EVENT_FINAL_FLAG_DECISION = "final_flag_decision"


def write_audit_entry(
    contract_id: uuid.UUID,
    event_type: str,
    details: str,
    clause_id: uuid.UUID | None = None,
) -> None:
    """Write one AuditLog row on its own independent session (see module
    docstring for why). Never raises -- any failure is logged and
    swallowed, so the caller's own pipeline/flagging logic is completely
    unaffected regardless of what goes wrong here.
    """
    try:
        session = SessionLocal()
    except Exception:
        logger.error(
            "write_audit_entry: could not open a DB session -- audit "
            "entry LOST (contract_id=%s clause_id=%s event_type=%s).",
            contract_id, clause_id, event_type, exc_info=True,
        )
        return

    try:
        session.add(
            AuditLog(
                contract_id=contract_id,
                clause_id=clause_id,
                event_type=event_type,
                details=details,
            )
        )
        session.commit()
    except Exception:
        logger.error(
            "write_audit_entry: failed to write audit entry -- audit "
            "entry LOST, pipeline continuing (contract_id=%s clause_id=%s "
            "event_type=%s details=%r).",
            contract_id, clause_id, event_type, details, exc_info=True,
        )
        try:
            session.rollback()
        except Exception:
            logger.error("write_audit_entry: rollback also failed.", exc_info=True)
    finally:
        try:
            session.close()
        except Exception:
            logger.error("write_audit_entry: session close also failed.", exc_info=True)
