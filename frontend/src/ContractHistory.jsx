import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import {
  AlertTriangle,
  CheckCircle2,
  Clock,
  FileStack,
  Loader2,
  ShieldAlert,
  UploadCloud,
  XCircle,
} from "lucide-react";
import { getContracts } from "./api.js";

// M38: reuses the SAME small-badge visual language the results view
// already established for clause severity (severity-badge) -- not a new
// color system, just the same pill shape/weight applied to a contract's
// overall STATUS instead of a clause's severity. "processing" reuses
// --primary (the same color the live processing card's own pulse-dot
// uses), "complete" reuses --success, "failed" reuses --danger -- all
// three are EXISTING tokens from index.css, none invented for this.
const STATUS_LABELS = {
  processing: "Processing",
  complete: "Complete",
  failed: "Failed",
};
const STATUS_ICONS = {
  processing: Loader2,
  complete: CheckCircle2,
  failed: XCircle,
};

function StatusBadge({ status }) {
  const known = status in STATUS_LABELS;
  const Icon = known ? STATUS_ICONS[status] : XCircle;
  return (
    <span className="status-badge" data-status={known ? status : "complete"}>
      <Icon size={12} strokeWidth={2.5} className={status === "processing" ? "spin" : ""} />
      {known ? STATUS_LABELS[status] : status}
    </span>
  );
}

function formatDate(isoString) {
  return new Date(isoString).toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

// The quick per-contract summary -- reuses the EXACT SAME .summary-pill
// component styling the single-contract results view's own SummaryStrip
// uses (Dashboard.jsx), just fed the coarser 3-bucket counts
// GET /contracts itself returns (see backend/routes/contracts.py's
// ContractSummary docstring for why that summary is deliberately
// coarser than the full per-clause detail view's own 4-bucket split).
function HistoryItemSummary({ contract }) {
  if (contract.status !== "complete") return <span className="history-item-empty-note">—</span>;
  const { clauses_flagged, clauses_needs_review, clauses_no_flag } = contract;
  if (clauses_flagged === 0 && clauses_needs_review === 0 && clauses_no_flag === 0) {
    return <span className="history-item-empty-note">No clauses</span>;
  }
  return (
    <span className="history-item-summary">
      {clauses_flagged > 0 && (
        <span className="summary-pill" data-tone="high">
          <span className="count">{clauses_flagged}</span> flagged
        </span>
      )}
      {clauses_needs_review > 0 && (
        <span className="summary-pill" data-tone="needs_manual_review">
          <span className="count">{clauses_needs_review}</span> review
        </span>
      )}
      {clauses_no_flag > 0 && (
        <span className="summary-pill" data-tone="clean">
          <span className="count">{clauses_no_flag}</span> clean
        </span>
      )}
    </span>
  );
}

const rowVariants = {
  hidden: { opacity: 0, y: 8 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.3, ease: [0.2, 0.7, 0.3, 1] } },
};

function ContractRow({ contract, onSelect }) {
  return (
    <motion.tr
      className="history-row"
      data-status={contract.status}
      onClick={() => onSelect(contract)}
      variants={rowVariants}
      tabIndex={0}
      role="button"
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          onSelect(contract);
        }
      }}
    >
      <td className="history-row-filename">{contract.filename}</td>
      <td className="history-row-date">{formatDate(contract.created_at)}</td>
      <td>
        <StatusBadge status={contract.status} />
      </td>
      <td>
        <HistoryItemSummary contract={contract} />
      </td>
    </motion.tr>
  );
}

// M38/M39: fetches up to MAX_FETCH contracts ONCE -- both the stat
// strip below and the table body render from this SAME real fetched
// set, so "N contracts reviewed, N risks caught" are genuinely computed
// numbers (sum of real clauses_flagged across every loaded contract),
// never fabricated copy. MAX_FETCH matches the backend's own real
// page-size ceiling (routes/contracts.py's MAX_PAGE_SIZE) -- this
// project's documented scale (a capstone-scale demo, not a production
// SaaS with thousands of contracts per user) makes one full fetch the
// simplest honest option; if a user genuinely has more than that, the
// note below says so rather than silently under-reporting.
const MAX_FETCH = 100;

function StatStrip({ contracts, total }) {
  const flaggedTotal = contracts.reduce((sum, c) => sum + c.clauses_flagged, 0);
  const reviewTotal = contracts.reduce((sum, c) => sum + c.clauses_needs_review, 0);
  const processingCount = contracts.filter((c) => c.status === "processing").length;
  return (
    <div className="stat-strip">
      <div className="stat-card">
        <FileStack size={20} />
        <div>
          <strong>{total}</strong>
          <span>contract{total === 1 ? "" : "s"} reviewed</span>
        </div>
      </div>
      <div className="stat-card" data-tone="high">
        <ShieldAlert size={20} />
        <div>
          <strong>{flaggedTotal}</strong>
          <span>risk{flaggedTotal === 1 ? "" : "s"} flagged</span>
        </div>
      </div>
      <div className="stat-card" data-tone="needs_manual_review">
        <AlertTriangle size={20} />
        <div>
          <strong>{reviewTotal}</strong>
          <span>need review</span>
        </div>
      </div>
      {processingCount > 0 && (
        <div className="stat-card" data-tone="processing">
          <Clock size={20} />
          <div>
            <strong>{processingCount}</strong>
            <span>processing now</span>
          </div>
        </div>
      )}
    </div>
  );
}

// M38: the real "My Contracts" history -- every contract this user has
// ever uploaded, most recent first, backed by GET /contracts. Rendered
// as a real table (a "document management" layout, per this pass's own
// goal) rather than a card list -- clicking any row calls
// onSelectContract (Dashboard.jsx's openContract), which reuses the
// EXISTING processing/done/failed rendering this app already has for a
// fresh upload, just seeded from a past contract_id instead. This
// component does NOT reimplement that results view.
export default function ContractHistory({ token, onSelectContract, onUploadNew }) {
  const [state, setState] = useState("loading"); // loading | loaded | error
  const [contracts, setContracts] = useState([]);
  const [total, setTotal] = useState(0);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      setState("loading");
      try {
        const result = await getContracts(token, { limit: MAX_FETCH, offset: 0 });
        if (cancelled) return;
        setContracts(result.contracts);
        setTotal(result.total);
        setState("loaded");
      } catch (err) {
        if (cancelled) return;
        setError(err.message || "Failed to load your contracts");
        setState("error");
      }
    }
    load();
    return () => {
      cancelled = true;
    };
  }, [token]);

  if (state === "loading") {
    return (
      <div className="loading-card">
        <span className="pulse-dot" />
        <span>Loading your contracts…</span>
      </div>
    );
  }

  if (state === "error") {
    return (
      <div className="form-message error">
        <span className="msg-icon">⚠</span>
        {error}
      </div>
    );
  }

  if (contracts.length === 0) {
    return (
      <div className="empty-state">
        <div className="empty-state-icon" aria-hidden="true">
          <FileStack size={22} />
        </div>
        <h3>No contracts yet</h3>
        <p>Once you upload a contract, it will show up here so you can come back to it any time.</p>
        <button type="button" className="btn btn-primary" onClick={onUploadNew} style={{ marginTop: 8 }}>
          <UploadCloud size={15} /> Upload your first contract
        </button>
      </div>
    );
  }

  return (
    <div>
      <div className="results-header">
        <h2>My Contracts</h2>
        <span className="results-stat">{total} total</span>
      </div>
      <StatStrip contracts={contracts} total={total} />
      {total > MAX_FETCH && (
        <p className="history-note">
          Showing the {MAX_FETCH} most recent of {total} contracts.
        </p>
      )}
      <div className="history-table-wrap">
        <table className="history-table">
          <thead>
            <tr>
              <th>Filename</th>
              <th>Uploaded</th>
              <th>Status</th>
              <th>Summary</th>
            </tr>
          </thead>
          <motion.tbody initial="hidden" animate="visible" transition={{ staggerChildren: 0.03 }}>
            {contracts.map((contract) => (
              <ContractRow key={contract.id} contract={contract} onSelect={onSelectContract} />
            ))}
          </motion.tbody>
        </table>
      </div>
    </div>
  );
}
