import { useEffect, useId, useRef, useState } from "react";
import { motion } from "framer-motion";
import {
  AlertCircle,
  AlertOctagon,
  AlertTriangle,
  Boxes,
  CheckCircle2,
  ChevronRight,
  FileCheck2,
  FileText,
  HelpCircle,
  History as HistoryIcon,
  LayoutGrid,
  ListChecks,
  LogOut,
  Quote,
  Scissors,
  ShieldAlert,
  ShieldCheck,
  Tags,
  Table2,
  UploadCloud,
  UserCircle,
  XCircle,
} from "lucide-react";
import { uploadContract, getContractStatus, getFlaggedClauses } from "./api.js";
import ContractHistory from "./ContractHistory.jsx";

const POLL_INTERVAL_MS = 3000;
// A 404 on the first poll or two is tolerated rather than shown as a
// scary fatal error -- see the matching comment in
// backend/routes/contracts.py's upload route for the full reasoning:
// the Contract row is committed before the upload response is even
// returned, so this window is believed to be unreachable in practice,
// but the tolerance costs nothing and guards against that reasoning
// ever becoming wrong later. Capped so a GENUINELY wrong/deleted
// contract_id doesn't just poll forever in silence.
const NOT_FOUND_TOLERANCE = 3;

const SEVERITY_LABELS = {
  high: "HIGH",
  medium: "MEDIUM",
  low: "LOW",
  flagging_failed: "FLAGGING FAILED",
  needs_manual_review: "NEEDS MANUAL REVIEW",
};

// Real iconography per severity -- NOT a new color system (see index.css:
// these still ride the exact same red/amber/green/grey/blue tokens this
// project already committed to), just a shape alongside the color so the
// signal doesn't rely on hue alone.
const SEVERITY_ICONS = {
  high: AlertOctagon,
  medium: AlertTriangle,
  low: CheckCircle2,
  flagging_failed: XCircle,
  needs_manual_review: HelpCircle,
};

function SeverityBadge({ severity }) {
  const known = severity in SEVERITY_LABELS;
  const label = known ? SEVERITY_LABELS[severity] : SEVERITY_LABELS.flagging_failed;
  const Icon = known ? SEVERITY_ICONS[severity] : SEVERITY_ICONS.flagging_failed;
  return (
    <span className="severity-badge" data-severity={known ? severity : "flagging_failed"}>
      <Icon size={12} strokeWidth={2.5} />
      {label}
    </span>
  );
}

// M29: severity (Groq's real risk assessment) and classification
// confidence (how sure stage 1/2 were about which of the 5 categories a
// clause belongs to) are two INDEPENDENT signals -- flag_clause() never
// receives `category` at all (see backend/agent/flag_clause.py), so a
// clause's risk assessment is never influenced by how confident its
// category label was. This means a clause CAN be both genuinely flagged
// (real risk) AND low-confidence in its category at the same time -- not
// a contradiction, just two separate questions. Design decision for that
// overlap (unchanged from the original): severity takes VISUAL PRIORITY
// -- a real, Groq-flagged risk is more actionable than a
// classification-confidence question, so the clause stays in the
// "Flagged" section rather than being pulled into "Possible Risk", but a
// low-confidence badge renders alongside its severity badge so the
// signal is never silently dropped, just deprioritized.

// Which of the 4 UI buckets a clause belongs to -- unchanged logic from
// the original: every clause maps to EXACTLY one bucket, checked in this
// exact order (needs_manual_review first, since it would otherwise also
// match the "flagged" catch-all).
function bucketOf(clause) {
  if (clause.severity === "needs_manual_review") return "needs_manual_review";
  if (clause.severity !== "low") return "flagged";
  return clause.classification_confidence === "low" ? "possible_risk" : "no_flag";
}

// Severity rank purely for the compact TABLE view's sort order (below) --
// real risk floats to the top so "scan quickly" actually means
// something; the CARD view keeps its own separate, bucket-grouped order
// unchanged.
const SEVERITY_RANK = { high: 0, medium: 1, needs_manual_review: 2, flagging_failed: 3, low: 4 };

function ClassificationNote({ clause }) {
  let provenance;
  if (clause.classification_stage === "heading_match") {
    provenance = "matched by heading";
  } else if (clause.classification_stage === "centroid_fallback") {
    provenance = `matched by similarity (${clause.classification_similarity.toFixed(4)})`;
  } else {
    // Not asserting "(below 0.5 threshold)" here on purpose: this
    // similarity is RECOMPUTED live for display, and real testing showed
    // it can occasionally land on the opposite side of 0.5 from the
    // persisted Unclassified decision (embedding recomputation isn't
    // perfectly reproducible run-to-run right at the boundary) -- the
    // persisted category is always the source of truth for bucketing
    // (Unclassified is always LOW confidence regardless of what this
    // number says), so this caption just reports the live number without
    // implying a guaranteed relationship to the 0.5 cutoff.
    provenance =
      clause.classification_similarity != null
        ? `unclassified — best similarity ${clause.classification_similarity.toFixed(4)}`
        : "unclassified";
  }
  const confidenceLabel = clause.classification_confidence === "low" ? "LOW confidence" : "HIGH confidence";
  return (
    <p className="classification-note">
      Category: {clause.category || "(none)"} — {provenance} · {confidenceLabel}
    </p>
  );
}

// Replaces the native <details>/<summary> (which snaps open/closed with
// no transition in every browser) with a real, animated expand/collapse
// -- a controlled <button aria-expanded> + a grid-rows-driven body (see
// index.css's .disclosure-body / 0fr->1fr technique) so the citation
// panel eases open instead of snapping. Semantics preserved: still a
// single interactive trigger, still keyboard-operable (a real <button>,
// tab/Enter/Space all work natively), still announces its open/closed
// state via aria-expanded the way <details>'s own open attribute would.
//
// "Evidence" treatment: the citation is the actual, verifiable proof
// behind an AI-generated risk call -- styled as a real quoted exhibit
// (Quote icon, blockquote-style left rule) rather than a plain <pre>
// tucked behind a collapsed link, per this pass's own "here's the proof,
// not an afterthought" goal.
function Disclosure({ summary, children }) {
  const [open, setOpen] = useState(false);
  const bodyId = useId();
  return (
    <div className="disclosure">
      <button
        type="button"
        className="disclosure-trigger"
        aria-expanded={open}
        aria-controls={bodyId}
        onClick={() => setOpen((o) => !o)}
      >
        <ChevronRight size={14} className="disclosure-caret" />
        {summary}
      </button>
      <div id={bodyId} className={`disclosure-body ${open ? "open" : ""}`}>
        <div className="disclosure-body-inner">{children}</div>
      </div>
    </div>
  );
}

const cardListVariants = {
  hidden: {},
  visible: { transition: { staggerChildren: 0.05 } },
};
const cardItemVariants = {
  hidden: { opacity: 0, y: 10 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.35, ease: [0.2, 0.7, 0.3, 1] } },
};

function FlaggedClauseCard({ clause }) {
  const failed = clause.severity === "flagging_failed";
  const needsManualReview = clause.severity === "needs_manual_review";
  const isRealRisk = clause.severity !== "low" && !needsManualReview;
  const showOverlapBadge = isRealRisk && clause.classification_confidence === "low";
  return (
    <motion.div className="clause-card" data-severity={clause.severity} variants={cardItemVariants}>
      <div className="clause-card-head">
        <strong>
          {clause.heading_path}
          {clause.category ? ` (${clause.category})` : ""}
        </strong>
        <span className="badge-row">
          <SeverityBadge severity={clause.severity} />
          {showOverlapBadge && (
            <span
              className="low-confidence-badge"
              title="Groq flagged real risk here, but the category label itself is a low-confidence guess -- see the classification note below."
            >
              <AlertCircle size={11} strokeWidth={2.5} /> CATEGORY LOW CONFIDENCE
            </span>
          )}
        </span>
      </div>
      {failed ? (
        <p className="clause-explanation warn">
          <AlertTriangle size={14} className="inline-icon" /> Automated flagging failed for this clause after
          retries — no risk assessment was produced. Raw clause text is shown below for manual review.
        </p>
      ) : needsManualReview ? (
        <p className="clause-explanation warn">
          <HelpCircle size={14} className="inline-icon" /> Groq's response for this clause failed schema
          validation twice in a row (the original attempt and one retry) — no risk assessment was produced.
          Raw clause text is shown below for manual review.
        </p>
      ) : (
        <p className="clause-explanation">{clause.explanation}</p>
      )}
      <ClassificationNote clause={clause} />
      <Disclosure
        summary={
          <>
            <Quote size={13} /> View citation
          </>
        }
      >
        <div className="evidence-block">
          <Quote size={16} className="evidence-mark" aria-hidden="true" />
          <pre>{clause.citation}</pre>
        </div>
      </Disclosure>
    </motion.div>
  );
}

const SECTION_COPY = {
  flagged: { title: "Flagged — Real Risk", tone: "flagged", icon: ShieldAlert },
  possible_risk: {
    title: "Possible Risk — Needs Review",
    subtitle:
      "Groq did not flag these as high-severity risk, but the category they were filed under is a low-confidence classification -- worth a second look.",
    tone: "possible_risk",
    icon: AlertCircle,
  },
  needs_manual_review: {
    title: "Needs Manual Review — Assessment Failed",
    subtitle:
      "Groq's response for these clauses failed schema validation twice in a row -- no automated risk assessment was produced at all. Review the raw clause text directly.",
    tone: "needs_manual_review",
    icon: HelpCircle,
  },
  no_flag: { title: "No Flag", tone: "no_flag", icon: CheckCircle2 },
};

// M29: renders nothing (not even an empty header) when this bucket has
// no clauses, so an empty section never shows up as a confusing blank
// heading.
function ClauseSection({ bucket, clauses }) {
  if (clauses.length === 0) return null;
  const copy = SECTION_COPY[bucket];
  return (
    <div className="section" data-tone={copy.tone}>
      <span className="section-heading" data-tone={copy.tone}>
        <copy.icon size={14} strokeWidth={2.5} />
        {copy.title} ({clauses.length})
      </span>
      {copy.subtitle && <p className="section-subtitle">{copy.subtitle}</p>}
      <motion.div initial="hidden" animate="visible" variants={cardListVariants}>
        {clauses.map((clause) => (
          <FlaggedClauseCard key={clause.clause_id} clause={clause} />
        ))}
      </motion.div>
    </div>
  );
}

// Glanceable "dashboard before the detail list" -- real counts per
// bucket (further split into high/medium within "flagged", since that
// distinction is the most actionable one) so a long contract's overall
// shape reads in one line before scrolling into individual cards. Zero-
// count pills are omitted, same principle ClauseSection already applies
// to whole empty sections.
const SUMMARY_PILLS = [
  { tone: "high", label: "high-risk", match: (c) => c.severity === "high" },
  { tone: "medium", label: "medium-risk", match: (c) => c.severity === "medium" },
  { tone: "possible_risk", label: "possible risk", match: (c) => bucketOf(c) === "possible_risk" },
  { tone: "needs_manual_review", label: "needs review", match: (c) => bucketOf(c) === "needs_manual_review" },
  { tone: "failed", label: "flagging failed", match: (c) => c.severity === "flagging_failed" },
  { tone: "clean", label: "clean", match: (c) => bucketOf(c) === "no_flag" },
];

function SummaryStrip({ clauses }) {
  const counts = SUMMARY_PILLS.map((p) => ({ ...p, count: clauses.filter(p.match).length })).filter(
    (p) => p.count > 0
  );
  if (counts.length === 0) return null;
  return (
    <div className="summary-strip">
      {counts.map((p) => (
        <span className="summary-pill" data-tone={p.tone} key={p.tone}>
          <span className="count">{p.count}</span> {p.label}
        </span>
      ))}
    </div>
  );
}

// A real analytical read on a contract's overall shape -- a proportion
// bar (real percentages of this contract's own clauses, not a
// fabricated "risk score") plus a legend, sitting right above the detail
// list. Coarser than the 6-way SummaryStrip above on purpose: a visual
// proportion read is clearer with 4 segments than 6 slivers.
const PROPORTION_BUCKETS = [
  { bucket: "flagged", tone: "high", label: "Flagged" },
  { bucket: "possible_risk", tone: "possible_risk", label: "Possible risk" },
  { bucket: "needs_manual_review", tone: "needs_manual_review", label: "Needs review" },
  { bucket: "no_flag", tone: "clean", label: "Clean" },
];

function RiskProportion({ clauses }) {
  const total = clauses.length;
  if (total === 0) return null;
  const segments = PROPORTION_BUCKETS.map((p) => ({
    ...p,
    count: clauses.filter((c) => bucketOf(c) === p.bucket).length,
  })).filter((p) => p.count > 0);

  return (
    <div className="risk-proportion">
      <div className="risk-proportion-bar">
        {segments.map((s, i) => (
          <motion.div
            key={s.bucket}
            className="risk-proportion-segment"
            data-tone={s.tone}
            initial={{ width: 0 }}
            animate={{ width: `${(s.count / total) * 100}%` }}
            transition={{ duration: 0.7, delay: i * 0.08, ease: [0.2, 0.7, 0.3, 1] }}
            title={`${s.label}: ${s.count} of ${total} (${Math.round((s.count / total) * 100)}%)`}
          />
        ))}
      </div>
      <div className="risk-proportion-legend">
        {segments.map((s) => (
          <span className="risk-proportion-legend-item" key={s.bucket}>
            <span className="risk-proportion-dot" data-tone={s.tone} />
            {s.label} · {Math.round((s.count / total) * 100)}%
          </span>
        ))}
      </div>
    </div>
  );
}

// A compact, flat, sortable-by-eye alternative to the grouped card view
// -- every clause in ONE table, ranked by real severity so risk floats
// to the top, for reviewing a long contract without scrolling through
// dense cards for everything. The card view (ClauseSection, above)
// remains the default and is unchanged; this is an additional, opt-in
// way to look at the exact same data.
function ResultsTable({ clauses }) {
  const sorted = [...clauses].sort((a, b) => SEVERITY_RANK[a.severity] - SEVERITY_RANK[b.severity]);
  return (
    <div className="results-table-wrap">
      <table className="results-table">
        <thead>
          <tr>
            <th>Clause</th>
            <th>Severity</th>
            <th>Confidence</th>
            <th>Explanation</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((c) => (
            <tr key={c.clause_id} data-severity={c.severity}>
              <td className="results-table-heading">
                {c.heading_path}
                {c.category && <span className="results-table-category">{c.category}</span>}
              </td>
              <td>
                <SeverityBadge severity={c.severity} />
              </td>
              <td className="results-table-confidence" data-confidence={c.classification_confidence}>
                {c.classification_confidence === "low" ? "Low" : "High"}
              </td>
              <td className="results-table-explanation">{c.explanation}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ElapsedTimer() {
  const [seconds, setSeconds] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setSeconds((s) => s + 1), 1000);
    return () => clearInterval(id);
  }, []);
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return (
    <span>
      {mins}:{secs.toString().padStart(2, "0")}
    </span>
  );
}

// M40: a REAL, live-tracked pipeline stepper -- backed by
// Contract.current_stage (backend/models/contract.py), which
// run_contract_pipeline() (backend/pipeline/run_contract.py) now stamps
// and commits at the START of each real stage, exposed by GET
// /contracts/{id}/status. STAGE_ORDER's string values are the exact,
// literal current_stage values the backend writes -- this is a direct
// reflection of real backend state, not an animation or a guess.
const STAGE_ORDER = ["extracting", "chunking", "persisting", "embedding", "flagging"];

const PIPELINE_STAGES = [
  { label: "Extract", icon: FileText },
  { label: "Chunk", icon: Scissors },
  { label: "Classify", icon: Tags },
  { label: "Embed", icon: Boxes },
  { label: "Flag risk", icon: ShieldAlert },
];

// Maps each real STAGE_ORDER value to its matching PIPELINE_STAGES label
// -- used by the failed-state card to name exactly which real stage was
// running when the pipeline died.
const STAGE_LABELS = Object.fromEntries(STAGE_ORDER.map((key, i) => [key, PIPELINE_STAGES[i].label]));

// Returns how many real stages have been passed: STAGE_ORDER.length (all
// done) for "complete"; the real index of currentStage within
// STAGE_ORDER; or -1 for anything else -- null/undefined (no status
// polled yet, or the worker hasn't reached its first real stage yet) and
// the narrow "failed-before-any-stage" fallback value "failed" (see
// Contract.current_stage's own docstring) both honestly mean "no real
// stage confirmed reached yet", so both correctly render as all-pending
// rather than guessing which stage was actually running.
function stageIndex(currentStage) {
  if (currentStage === "complete") return STAGE_ORDER.length;
  return STAGE_ORDER.indexOf(currentStage);
}

function ProcessingPhase({ currentStage }) {
  const index = stageIndex(currentStage);
  return (
    <div className="pipeline-stepper">
      {PIPELINE_STAGES.map((stage, i) => {
        const state = i < index ? "done" : i === index ? "active" : "pending";
        return (
          <div className="pipeline-step" key={stage.label}>
            <motion.div
              className="pipeline-step-icon"
              data-state={state}
              animate={state === "active" ? { scale: [1, 1.14, 1] } : { scale: 1 }}
              transition={state === "active" ? { duration: 1.6, repeat: Infinity, ease: "easeInOut" } : {}}
            >
              <stage.icon size={18} strokeWidth={2} />
            </motion.div>
            <span className="pipeline-step-label" data-state={state}>
              {stage.label}
            </span>
            {i < PIPELINE_STAGES.length - 1 && (
              <span className="pipeline-step-connector" data-state={i < index ? "done" : "pending"} />
            )}
          </div>
        );
      })}
    </div>
  );
}

export default function Dashboard({ token, userEmail, onLogout }) {
  // M38: "upload" is both "upload a new contract" AND "view one specific
  // contract's live/complete state" -- the same view, just seeded either
  // by a fresh upload (handleUpload) or by picking an existing contract
  // from history (openContract, below). "history" is the new "My
  // Contracts" list. Kept as sibling top-level views (not nested
  // routing) -- this app has no URL-based routing at all (see App.jsx's
  // own docstring on scope), so a plain view switch is the simplest
  // thing that actually satisfies "real navigation, not a dead end."
  const [view, setView] = useState("upload"); // "upload" | "history"
  const [uploadState, setUploadState] = useState("idle"); // idle | processing | loading | done | failed | error
  const [uploadError, setUploadError] = useState(null);
  const [contractId, setContractId] = useState(null);
  const [filename, setFilename] = useState(null);
  const [statusInfo, setStatusInfo] = useState(null); // { status, clauses_persisted, clauses_flagged }
  const [flaggedClauses, setFlaggedClauses] = useState([]);
  const [chosenFileName, setChosenFileName] = useState(null);
  const [resultsView, setResultsView] = useState("cards"); // "cards" | "table"
  const fileInputRef = useRef(null);

  // Interval id lives in a ref, not state -- state changes would fire on
  // a re-render, which we don't need, and a ref is what clearInterval
  // needs to actually stop it. notFoundCountRef tracks consecutive
  // tolerated 404s (see NOT_FOUND_TOLERANCE above).
  const pollIntervalRef = useRef(null);
  const notFoundCountRef = useRef(0);

  function stopPolling() {
    if (pollIntervalRef.current !== null) {
      clearInterval(pollIntervalRef.current);
      pollIntervalRef.current = null;
    }
  }

  // Guarantees polling never outlives this component, even though in
  // this single-page app it never actually unmounts during normal use --
  // still the correct safeguard against a leaked interval.
  useEffect(() => {
    return () => stopPolling();
  }, []);

  async function pollOnce(id, authToken) {
    try {
      const result = await getContractStatus(id, authToken);
      notFoundCountRef.current = 0; // a real response resets the tolerance window
      setStatusInfo(result);

      if (result.status === "complete") {
        stopPolling();
        const flagged = await getFlaggedClauses(id, authToken);
        setFlaggedClauses(flagged.flagged_clauses);
        setUploadState("done");
      } else if (result.status === "failed") {
        stopPolling();
        setUploadState("failed");
      }
      // else "processing" (or any other non-terminal value): keep polling.
    } catch (err) {
      if (err.status === 404 && notFoundCountRef.current < NOT_FOUND_TOLERANCE) {
        notFoundCountRef.current += 1;
        return; // tolerate briefly, keep polling -- not shown to the user
      }
      stopPolling();
      setUploadError(err.message || "Status check failed");
      setUploadState("error");
    }
  }

  function startPolling(id, authToken) {
    stopPolling();
    notFoundCountRef.current = 0;
    pollIntervalRef.current = setInterval(() => pollOnce(id, authToken), POLL_INTERVAL_MS);
  }

  async function handleUpload(e) {
    e.preventDefault();
    const file = fileInputRef.current?.files?.[0];
    if (!file) return;

    setUploadState("processing");
    setUploadError(null);
    setStatusInfo(null);
    setFlaggedClauses([]);
    setResultsView("cards");
    try {
      // Returns near-instantly now (M15/M16) -- status='processing',
      // no results yet. The real pipeline runs in the Celery worker;
      // polling (below) is how the UI finds out when it's done.
      const result = await uploadContract(file, token);
      setContractId(result.id);
      setFilename(result.filename);
      startPolling(result.id, token);
    } catch (err) {
      setUploadError(err.message || "Upload failed");
      setUploadState("error");
    }
  }

  function handleUploadAnother() {
    stopPolling();
    setView("upload");
    setUploadState("idle");
    setContractId(null);
    setFilename(null);
    setStatusInfo(null);
    setFlaggedClauses([]);
    setUploadError(null);
    setChosenFileName(null);
    setResultsView("cards");
    if (fileInputRef.current) fileInputRef.current.value = "";
  }

  // M38: open a PREVIOUSLY uploaded contract from the "My Contracts"
  // history list -- reuses this SAME component's existing processing/
  // done/failed views rather than building a second results renderer.
  // The history list's own `status` is a snapshot from whenever that
  // list was fetched, which could be stale by the time a user actually
  // clicks a row (e.g. a "processing" contract that finished in the
  // meantime) -- get_contract_status is re-checked here as the real,
  // current source of truth before deciding which view to land on,
  // rather than trusting the list response blindly.
  async function openContract(contract) {
    stopPolling();
    setView("upload");
    setContractId(contract.id);
    setFilename(contract.filename);
    setUploadError(null);
    setChosenFileName(null);
    setResultsView("cards");
    if (fileInputRef.current) fileInputRef.current.value = "";

    if (contract.status === "processing") {
      // Route straight to the existing live-polling view -- there are no
      // real results to show yet, so a static fetch would have nothing
      // to render.
      setStatusInfo(null);
      setFlaggedClauses([]);
      setUploadState("processing");
      startPolling(contract.id, token);
      return;
    }

    // A brief, honest "loading" state (distinct from "processing" --
    // that specifically means the backend PIPELINE is still running,
    // which isn't true here) while the real, persisted results for THIS
    // specific contract_id are fetched from the database.
    setUploadState("loading");
    try {
      const [freshStatus, flagged] = await Promise.all([
        getContractStatus(contract.id, token),
        getFlaggedClauses(contract.id, token),
      ]);
      setStatusInfo(freshStatus);
      if (freshStatus.status === "processing") {
        // Genuinely changed since the list was fetched -- fall back to
        // live polling rather than showing (real but momentarily empty)
        // results for a contract that isn't done yet.
        setFlaggedClauses([]);
        setUploadState("processing");
        startPolling(contract.id, token);
      } else if (freshStatus.status === "failed") {
        setUploadState("failed");
      } else {
        setFlaggedClauses(flagged.flagged_clauses);
        setUploadState("done");
      }
    } catch (err) {
      setUploadError(err.message || "Failed to load this contract");
      setUploadState("error");
    }
  }

  return (
    <div className="app-shell">
      <header className="topbar chrome">
        <div className="brand">
          <span className="brand-mark chrome">
            <ShieldCheck size={20} strokeWidth={2.25} />
            ClauseGuard
          </span>
        </div>
        <nav className="topbar-nav">
          <button
            type="button"
            className={`nav-link ${view === "upload" ? "active" : ""}`}
            onClick={handleUploadAnother}
          >
            <UploadCloud size={16} /> Upload
          </button>
          <button
            type="button"
            className={`nav-link ${view === "history" ? "active" : ""}`}
            onClick={() => setView("history")}
          >
            <HistoryIcon size={16} /> My Contracts
          </button>
        </nav>
        <div className="user-chip">
          <span className="user-chip-avatar">
            <UserCircle size={16} />
          </span>
          <span className="user-chip-email">{userEmail || "Signed in"}</span>
          <button type="button" className="btn-ghost chrome" onClick={onLogout}>
            <LogOut size={14} /> Log out
          </button>
        </div>
      </header>

      <main className="dashboard">
        {view === "history" ? (
          <ContractHistory token={token} onSelectContract={openContract} onUploadNew={handleUploadAnother} />
        ) : (
          <>
            {(uploadState === "idle" || uploadState === "error") && (
              <>
                {/* Landing/idle intro -- a brief, honest explanation of what
                    happens next, so a freshly-logged-in user's first screen
                    doesn't read as an empty dropzone with no context. The
                    dropzone card below keeps its own short copy too, for
                    anyone who skips straight past this. */}
                <div className="dashboard-intro">
                  <h1>Analyze a contract</h1>
                  <p>
                    Drop a contract below — ClauseGuard extracts every clause, classifies it, and flags real risk
                    with an explanation for each flag, so you know exactly why.
                  </p>
                </div>
                <form onSubmit={handleUpload}>
                  <div className="upload-card">
                    <div className="upload-icon">
                      <UploadCloud size={22} />
                    </div>
                    <h2>Upload a contract</h2>
                    <p>PDF or DOCX, up to a few pages of real clause text.</p>
                    <div className="file-input-row">
                      <input
                        type="file"
                        id="contract-file"
                        className="file-picker-input"
                        ref={fileInputRef}
                        required
                        accept=".pdf,.docx"
                        onChange={(e) => setChosenFileName(e.target.files?.[0]?.name || null)}
                      />
                      <label htmlFor="contract-file" className="file-picker-label">
                        Choose file
                      </label>
                      <span className="file-picker-name">{chosenFileName || "No file chosen"}</span>
                      <button type="submit" className="btn btn-primary">
                        Analyze contract
                      </button>
                    </div>
                  </div>
                </form>
                {uploadState === "error" && (
                  <div className="form-message error" style={{ marginTop: 16 }}>
                    <span className="msg-icon">⚠</span>
                    {uploadError}
                  </div>
                )}
              </>
            )}

            {uploadState === "processing" && (
              <div className="processing-card">
                <div className="processing-title-row">
                  <span className="pulse-dot" />
                  <strong>Processing {filename}…</strong>
                </div>
                <ProcessingPhase currentStage={statusInfo?.current_stage} />
                <p className="processing-disclaimer">
                  This reflects the pipeline's real current stage, polled every {POLL_INTERVAL_MS / 1000}s — it
                  can lag a few seconds behind the actual backend state, but it isn't a guess or a fixed animation.
                </p>
                <p>
                  The full pipeline runs against real APIs with real rate limits in a background worker, so a
                  multi-clause contract can take a few minutes. No need to keep waiting on this request — it
                  already returned; this page will update automatically.
                </p>
                <div className="processing-meta">
                  <span>
                    Elapsed: <strong><ElapsedTimer /></strong>
                  </span>
                  {statusInfo && (
                    <span>
                      <strong>{statusInfo.clauses_persisted}</strong> clause(s) persisted ·{" "}
                      <strong>{statusInfo.clauses_flagged}</strong> flagged so far
                    </span>
                  )}
                </div>
              </div>
            )}

            {uploadState === "loading" && (
              <div className="loading-card">
                <span className="pulse-dot" />
                <span>Loading {filename}…</span>
              </div>
            )}

            {uploadState === "failed" && (
              <div className="failed-card">
                <span className="failed-icon" aria-hidden="true">⚠</span>
                <div className="failed-card-body">
                  <p>
                    <strong>Processing failed.</strong> Something went wrong analyzing {filename} on the server and
                    it could not be completed. Check the worker's logs for details, or try uploading again.
                  </p>
                  {STAGE_ORDER.includes(statusInfo?.current_stage) && (
                    <>
                      <p className="failed-progress-label">
                        Real progress before failing — stopped during <strong>{STAGE_LABELS[statusInfo.current_stage]}</strong>:
                      </p>
                      <ProcessingPhase currentStage={statusInfo.current_stage} />
                    </>
                  )}
                  <button type="button" className="btn btn-primary" onClick={handleUploadAnother}>
                    Upload another contract
                  </button>
                </div>
              </div>
            )}

            {uploadState === "done" && (
              <div>
                <div className="results-header">
                  <h2>
                    <FileCheck2 size={20} className="inline-icon" /> Results for {filename}
                  </h2>
                  {statusInfo && (
                    <span className="results-stat">
                      {statusInfo.clauses_persisted} clauses processed · {statusInfo.clauses_flagged} flagged
                    </span>
                  )}
                </div>
                {flaggedClauses.length === 0 ? (
                  <div className="empty-state">
                    <div className="empty-state-icon" aria-hidden="true">✓</div>
                    <h3>Nothing to show here</h3>
                    <p>This contract completed processing, but no clauses were returned to display.</p>
                  </div>
                ) : (
                  <>
                    <RiskProportion clauses={flaggedClauses} />
                    <div className="results-toolbar">
                      <SummaryStrip clauses={flaggedClauses} />
                      <div className="view-toggle" role="group" aria-label="Results view">
                        <button
                          type="button"
                          className={resultsView === "cards" ? "active" : ""}
                          onClick={() => setResultsView("cards")}
                        >
                          <LayoutGrid size={14} /> Cards
                        </button>
                        <button
                          type="button"
                          className={resultsView === "table" ? "active" : ""}
                          onClick={() => setResultsView("table")}
                        >
                          <Table2 size={14} /> Table
                        </button>
                      </div>
                    </div>
                    {resultsView === "table" ? (
                      <ResultsTable clauses={flaggedClauses} />
                    ) : (
                      <>
                        {/* M29: every clause in flaggedClauses maps to EXACTLY one
                            of these buckets via bucketOf() -- nothing filtered
                            out, nothing double-counted. See bucketOf()'s own
                            docstring for the exact rule and the flagged+low-
                            confidence overlap handling. */}
                        <ClauseSection
                          bucket="flagged"
                          clauses={flaggedClauses.filter((c) => bucketOf(c) === "flagged")}
                        />
                        <ClauseSection
                          bucket="possible_risk"
                          clauses={flaggedClauses.filter((c) => bucketOf(c) === "possible_risk")}
                        />
                        <ClauseSection
                          bucket="needs_manual_review"
                          clauses={flaggedClauses.filter((c) => bucketOf(c) === "needs_manual_review")}
                        />
                        <ClauseSection
                          bucket="no_flag"
                          clauses={flaggedClauses.filter((c) => bucketOf(c) === "no_flag")}
                        />
                      </>
                    )}
                  </>
                )}
                <button type="button" className="btn btn-secondary" onClick={handleUploadAnother} style={{ marginTop: 8 }}>
                  <ListChecks size={15} /> Upload another contract
                </button>
              </div>
            )}
          </>
        )}
      </main>
    </div>
  );
}
