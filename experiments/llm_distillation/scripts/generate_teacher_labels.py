"""Phase 2(B) — HANDOFF SCRIPT, run this yourself, not in the sandbox.

Calls the REAL, unmodified backend/agent/flag_clause.py -> real Groq API
(llama-3.3-70b-versatile) on each of the 450 clauses in
data/distillation_input_clauses.jsonl, and writes teacher-labeled output to
data/teacher_labels.jsonl.

WHY THIS CAN'T RUN IN THE SANDBOX: the sandbox's network egress allowlist
does not include api.groq.com (only pypi/npm/github/anthropic domains).
This is the same constraint that pushed LoRA training to a Colab notebook
in the previous experiment.

HOW TO RUN THIS:
1. From your real ClauseGuard checkout, with the backend's venv active
   (`pip install -r requirements.txt` -- groq, pydantic-settings, and
   python-dotenv, which pydantic-settings depends on directly, are all
   already in there; no extra installs needed) and your real .env at the
   PROJECT ROOT (same file docker-compose already reads for
   `${GROQ_API_KEY}`-style substitution -- NOT a separate backend/.env).
2. Run from the PROJECT ROOT (this matters: config.py's
   `SettingsConfigDict(env_file=".env")` resolves ".env" relative to
   the CURRENT WORKING DIRECTORY, not this script's location):
     `python experiments/llm_distillation/scripts/generate_teacher_labels.py`
   This script adds backend/ to sys.path itself (see BACKEND_DIR below),
   so it does NOT need CWD to be backend/ for imports to resolve --
   only the .env lookup cares about CWD, which is why project-root is
   the one CWD that works for both.
3. Postgres does NOT need to be running. flag_clause() is called below
   with contract_id=clause_id=None (see agent/validate.py's own
   docstring), which makes validate_with_retry() skip every
   write_audit_entry() call entirely -- no DB session is ever opened.
   config.py/db.py still need SOME value for every required env var to
   import without error (POSTGRES_USER etc.) -- your existing .env from
   docker-compose already has these; you do not need Postgres itself
   running for THIS script.
4. This will very likely NOT finish in one run -- see the TPD budget
   note below. Just rerun the exact same command again after your daily
   Groq quota resets (00:00 UTC, per this repo's own established
   calendar-day convention -- see backend/middleware/rate_limit.py). The
   script skips everything already in data/teacher_labels.jsonl and picks
   up where it left off, so re-running is always safe.
5. Bring data/teacher_labels.jsonl (and this run's printed summary) back
   for Phase 3/4.

RATE LIMITING (free tier: 30 RPM / 1,000 RPD / 12K TPM / 100K TPD, per
flag_clause.py's own module docstring):
- RPM: a flat REQUEST_INTERVAL_SECONDS=2.5 sleep between calls keeps this
  well under 30/min even accounting for the occasional validation retry
  (flag_clause() can make 2 real Groq requests for one clause -- see
  agent/validate.py -- so worst case this is ~24 req/min, still under 30).
- TPD (the REAL binding constraint, not RPD -- see Phase 1 findings):
  this script tracks cumulative total_tokens actually used THIS RUN via a
  logging.Handler attached to flag_clause.py's own
  "clauseguard.agent.flag_clause" logger (that module already logs
  `usage=%s` -- the real Groq response.usage object -- on every
  successful call; this just reads that same real value back out rather
  than re-implementing the Groq call or guessing token counts). When
  cumulative usage crosses DAILY_TOKEN_BUDGET (95,000, a safety margin
  under the real 100,000 TPD cap), the script stops cleanly and reports
  progress -- it does NOT keep calling until a real 429 forces it to stop.

FAILURE HANDLING (disclosed, not hidden): if flag_clause() raises
FlagClauseError (API/infra failure) or NeedsManualReviewError (model
failed schema validation twice), that row is written with
label_status="flagging_failed" / "needs_manual_review" and an "error"
field containing the real exception message -- NEVER a fabricated
severity/explanation/citation. These rows should be excluded from LoRA
training data (Phase 3) and reported as real failures in the eventual
REPORT.md, not silently dropped.
"""

import json
import logging
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# Load .env into the REAL process environment before importing agent.flag_clause.
# config.py's pydantic Settings() also parses .env, but only for its OWN fields
# (JWT_SECRET_KEY etc.) -- it does NOT export values into os.environ as a side
# effect. flag_clause.py reads GROQ_API_KEY via plain os.getenv(), which only
# sees real process env vars, so without this explicit load it stays unset
# even with a correct .env file. Must run BEFORE the agent.flag_clause import
# below, and .env must be in the current working directory (project root).
load_dotenv()

# --- backend import root (see docstring point 2) ---
BACKEND_DIR = Path(__file__).resolve().parents[3] / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from agent.flag_clause import MODEL, FlagClauseError, flag_clause  # noqa: E402
from agent.validate import NeedsManualReviewError  # noqa: E402

INPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "distillation_input_clauses.jsonl"
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "teacher_labels.jsonl"

REQUEST_INTERVAL_SECONDS = 2.5
DAILY_TOKEN_BUDGET = 95_000  # safety margin under Groq free tier's 100K TPD


class _TokenUsageCapture(logging.Handler):
    """Reads the real Groq response.usage object back out of
    flag_clause.py's own existing log line (see module docstring) --
    does not call Groq or estimate tokens itself, just observes the
    real value flag_clause.py already computed and logged.
    """

    def __init__(self):
        super().__init__()
        self.last_total_tokens = None

    def emit(self, record):
        # flag_clause.py logs: "... usage=%s", ..., ..., ..., response.usage
        if record.args and len(record.args) >= 4:
            usage = record.args[3]
            total = getattr(usage, "total_tokens", None)
            if total is not None:
                self.last_total_tokens = total


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    input_rows = load_jsonl(INPUT_PATH)
    if not input_rows:
        print(f"No input rows found at {INPUT_PATH} -- run "
              f"select_distillation_subset.py first.")
        return

    already_done = {
        row["distill_id"] for row in load_jsonl(OUTPUT_PATH)
        if row.get("label_status") == "ok"
    }
    remaining = [r for r in input_rows if r["distill_id"] not in already_done]

    print(f"{len(already_done)}/{len(input_rows)} already labeled from a "
          f"previous run. {len(remaining)} remaining.")

    if not remaining:
        print("All clauses already labeled. Nothing to do.")
        return

    usage_capture = _TokenUsageCapture()
    logging.getLogger("clauseguard.agent.flag_clause").addHandler(usage_capture)
    logging.getLogger("clauseguard.agent.flag_clause").setLevel(logging.INFO)

    cumulative_tokens_this_run = 0
    n_labeled = 0
    n_failed = 0

    with open(OUTPUT_PATH, "a", encoding="utf-8") as out_f:
        for row in remaining:
            if cumulative_tokens_this_run >= DAILY_TOKEN_BUDGET:
                print(
                    f"\nHit today's token budget ({DAILY_TOKEN_BUDGET} "
                    f"tokens) after {n_labeled} labeled + {n_failed} failed "
                    f"this run. Stopping cleanly -- rerun this same script "
                    f"after your Groq quota resets (00:00 UTC) to continue."
                )
                break

            usage_capture.last_total_tokens = None
            record = {
                "distill_id": row["distill_id"],
                "text": row["text"],
                "category": row["category"],
                "contract": row["contract"],
                "split": row["split"],
                "teacher_model": MODEL,
                "label_source": "teacher_generated",  # NOT human ground truth
            }

            try:
                result = flag_clause(row["text"])  # real Groq call, real pipeline
                record["label_status"] = "ok"
                record["severity"] = result.severity
                record["explanation"] = result.explanation
                record["citation"] = result.citation
                n_labeled += 1
            except NeedsManualReviewError as exc:
                record["label_status"] = "needs_manual_review"
                record["error"] = str(exc)
                n_failed += 1
            except FlagClauseError as exc:
                record["label_status"] = "flagging_failed"
                record["error"] = str(exc)
                n_failed += 1

            out_f.write(json.dumps(record) + "\n")
            out_f.flush()

            if usage_capture.last_total_tokens is not None:
                cumulative_tokens_this_run += usage_capture.last_total_tokens

            print(
                f"[{row['distill_id']}] status={record['label_status']} "
                f"cumulative_tokens_this_run={cumulative_tokens_this_run}"
            )

            time.sleep(REQUEST_INTERVAL_SECONDS)

    print(
        f"\nThis run: {n_labeled} labeled, {n_failed} failed, "
        f"~{cumulative_tokens_this_run} tokens used."
    )
    total_done = len(already_done) + n_labeled + n_failed
    print(f"Total progress: {total_done}/{len(input_rows)}")
    if total_done < len(input_rows):
        print("Not done yet -- rerun this script tomorrow (after quota reset) to continue.")
    else:
        print("All clauses processed. Bring data/teacher_labels.jsonl back for Phase 3/4.")


if __name__ == "__main__":
    main()
