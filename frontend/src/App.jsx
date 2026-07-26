import { useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import Landing from "./Landing.jsx";
import AuthPage from "./AuthPage.jsx";
import Dashboard from "./Dashboard.jsx";

// Token storage choice: plain React state (in-memory) on purpose, not
// localStorage/sessionStorage. Reasoning: (1) this app's own scope is a
// single continuous session (auth -> upload -> view results) with no
// "survive a page refresh" requirement -- routing/multi-page/history are
// all explicitly out of scope; (2) in-memory state has the smallest
// attack surface for a bearer token -- nothing persists for an XSS
// payload to read later, and it's automatically gone on refresh/tab
// close; (3) if a later milestone needs the session to survive a
// refresh, the better real-world upgrade is an httpOnly cookie set by
// the backend, not moving this token into localStorage -- JS-readable
// storage of a bearer token is a tradeoff to make deliberately, not a
// default to fall into. UNCHANGED from the original M14 design -- only
// this file's own screen-switching grew a real "landing" step in front
// of auth, plus page-level transitions (framer-motion) between the
// three top-level screens.
//
// userEmail: captured at the moment of login/signup (AuthPage already
// has it in a local field value) and threaded down to Dashboard purely
// for presentation (the branded header showing who's signed in) -- NOT
// a new API call, NOT a backend change. /me (backend/routes/auth.py)
// only returns user_id, not email, so this is the one honest way to
// show a real email without altering any response shape.
const PAGE_TRANSITION = {
  initial: { opacity: 0 },
  animate: { opacity: 1 },
  exit: { opacity: 0 },
  transition: { duration: 0.22, ease: [0.2, 0.7, 0.3, 1] },
};

export default function App() {
  const [screen, setScreen] = useState("landing"); // "landing" | "auth"
  const [token, setToken] = useState(null);
  const [userEmail, setUserEmail] = useState(null);

  function handleAuthenticated(newToken, email) {
    setToken(newToken);
    setUserEmail(email);
  }

  function handleLogout() {
    setToken(null);
    setUserEmail(null);
    setScreen("landing");
  }

  const view = token ? "dashboard" : screen;

  return (
    <AnimatePresence mode="wait">
      {view === "landing" && (
        <motion.div key="landing" {...PAGE_TRANSITION}>
          <Landing onGetStarted={() => setScreen("auth")} />
        </motion.div>
      )}
      {view === "auth" && (
        <motion.div key="auth" {...PAGE_TRANSITION}>
          <AuthPage onAuthenticated={handleAuthenticated} onBack={() => setScreen("landing")} />
        </motion.div>
      )}
      {view === "dashboard" && (
        <motion.div key="dashboard" {...PAGE_TRANSITION}>
          <Dashboard token={token} userEmail={userEmail} onLogout={handleLogout} />
        </motion.div>
      )}
    </AnimatePresence>
  );
}
