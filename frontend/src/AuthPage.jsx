import { useLayoutEffect, useRef, useState } from "react";
import { ArrowLeft, Lock, Mail, MailCheck, ShieldCheck } from "lucide-react";
import { login, resendOtp, signup, verifyOtp } from "./api.js";

// Login's own behavior below is a direct, unmodified port of the
// original handleLogin -- same call, same token handoff (now also
// threading the real email up to App.jsx for header display, see that
// file's own docstring), same error surfacing.

// pydantic validation errors on /signup (e.g. password too short) come
// back as {"detail": [{"msg": "...", ...}, ...]}, not a plain string --
// api.js's shared parseJsonOrThrow (used by every endpoint) already
// JSON.stringifies non-string detail so no other caller breaks; this
// unpacks that shape ONLY here, at the UI layer, for a readable message,
// without touching the shared parsing logic every other call relies on.
function readableAuthError(err) {
  if (!err?.message) return "Something went wrong. Please try again.";
  try {
    const parsed = JSON.parse(err.message);
    if (Array.isArray(parsed) && parsed[0]?.msg) return parsed[0].msg;
  } catch {
    // Not JSON -- a plain detail string (e.g. "Email already registered",
    // "Invalid email or password") -- use it as-is.
  }
  return err.message;
}

export default function AuthPage({ onAuthenticated, onBack }) {
  const [mode, setMode] = useState("login"); // "login" | "signup" | "otp"
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(null);
  const [signupSuccess, setSignupSuccess] = useState(null);
  const loginTabRef = useRef(null);
  const signupTabRef = useRef(null);
  const [indicatorStyle, setIndicatorStyle] = useState({});

  // M39: OTP entry screen shown right after signup, before the
  // existing "switch to login" success flow. `email` above (already
  // captured by the signup form) is reused as-is for verify/resend --
  // no separate "pending email" state needed.
  const [otp, setOtp] = useState("");
  const [otpSubmitting, setOtpSubmitting] = useState(false);
  const [otpError, setOtpError] = useState(null);
  const [resendMessage, setResendMessage] = useState(null);
  const [resendSubmitting, setResendSubmitting] = useState(false);

  // Measures the ACTIVE tab's real position/width so the sliding
  // underline lines up exactly, instead of a hardcoded 50% split that
  // would drift if the two tab labels ever have different widths.
  function updateIndicator(targetMode) {
    const el = targetMode === "login" ? loginTabRef.current : signupTabRef.current;
    if (el) {
      setIndicatorStyle({ transform: `translateX(${el.offsetLeft}px)`, width: `${el.offsetWidth}px` });
    }
  }

  // Re-measure whenever the tabs are actually on screen -- both on
  // first paint (useLayoutEffect, not useEffect, so it never flashes
  // at 0/0) AND whenever mode transitions back to "login"/"signup"
  // from "otp" (the tabs unmount entirely during the OTP screen, so
  // their refs aren't attached again until this fires post-render).
  useLayoutEffect(() => {
    if (mode === "login" || mode === "signup") {
      updateIndicator(mode);
    }
  }, [mode]);

  function switchMode(next) {
    setMode(next);
    setError(null);
    if (next !== "login") {
      setSignupSuccess(null);
    }
  }

  async function handleLogin(e) {
    e.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      const result = await login(email, password);
      onAuthenticated(result.access_token, email);
    } catch (err) {
      setError(readableAuthError(err));
    } finally {
      setSubmitting(false);
    }
  }

  async function handleSignup(e) {
    e.preventDefault();
    setError(null);
    setSignupSuccess(null);
    setSubmitting(true);
    try {
      await signup(email, password);
      setPassword("");
      setOtp("");
      setOtpError(null);
      setResendMessage(null);
      setMode("otp");
    } catch (err) {
      setError(readableAuthError(err));
    } finally {
      setSubmitting(false);
    }
  }

  async function handleVerifyOtp(e) {
    e.preventDefault();
    setOtpError(null);
    setOtpSubmitting(true);
    try {
      await verifyOtp(email, otp);
      setOtp("");
      setSignupSuccess("Email verified — log in below.");
      setMode("login");
    } catch (err) {
      setOtpError(readableAuthError(err));
    } finally {
      setOtpSubmitting(false);
    }
  }

  async function handleResendOtp() {
    setOtpError(null);
    setResendMessage(null);
    setResendSubmitting(true);
    try {
      const result = await resendOtp(email);
      setResendMessage(result.message);
    } catch (err) {
      setOtpError(readableAuthError(err));
    } finally {
      setResendSubmitting(false);
    }
  }

  const isLogin = mode === "login";
  const isOtp = mode === "otp";

  return (
    <div className="auth-page">
      <div className="auth-split">
        <div className="auth-brand-panel">
          {onBack && (
            <button type="button" className="auth-back-link" onClick={onBack}>
              <ArrowLeft size={16} /> Back
            </button>
          )}
          <div className="auth-brand-content">
            <span className="brand-mark chrome">
              <ShieldCheck size={22} strokeWidth={2.25} />
              ClauseGuard
            </span>
            <h2>AI-assisted contract risk review</h2>
            <p>
              Every clause extracted, classified, and checked for real risk — with an explanation you
              can actually verify against the contract's own text.
            </p>
            <ul className="auth-brand-list">
              <li>Real reasoning, cited against the actual clause</li>
              <li>Your uploads and history stay private to your account</li>
              <li>Uncertain results are labeled, never guessed silently</li>
            </ul>
          </div>
        </div>

        <div className="auth-card">
        <div className="auth-card-inner">
          {isOtp ? (
            <div className="otp-panel">
              <div className="otp-icon-circle">
                <MailCheck size={22} strokeWidth={2.25} />
              </div>
              <h3>Check your email</h3>
              <p className="otp-hint">
                We sent a 6-digit code to <strong>{email}</strong>. Enter it below to verify your account.
              </p>
              <form onSubmit={handleVerifyOtp}>
                <div className="field">
                  <label htmlFor="otp-code">Verification code</label>
                  <input
                    id="otp-code"
                    type="text"
                    inputMode="numeric"
                    autoComplete="one-time-code"
                    maxLength={6}
                    value={otp}
                    onChange={(e) => setOtp(e.target.value.replace(/\D/g, "").slice(0, 6))}
                    required
                    className="otp-input"
                  />
                </div>
                <button
                  type="submit"
                  className="btn btn-primary btn-block"
                  disabled={otpSubmitting || otp.length !== 6}
                >
                  {otpSubmitting ? "Verifying…" : "Verify email"}
                </button>
                {otpError && (
                  <div className="form-message error">
                    <span className="msg-icon">⚠</span>
                    {otpError}
                  </div>
                )}
                {resendMessage && (
                  <div className="form-message success">
                    <span className="msg-icon">✓</span>
                    {resendMessage}
                  </div>
                )}
              </form>
              <p className="auth-switch-hint">
                Didn't get a code?{" "}
                <button type="button" onClick={handleResendOtp} disabled={resendSubmitting}>
                  {resendSubmitting ? "Sending…" : "Resend code"}
                </button>
              </p>
              <p className="auth-switch-hint">
                <button type="button" onClick={() => switchMode("signup")}>Use a different email</button>
              </p>
            </div>
          ) : (
          <>
          <div className="auth-tabs">
            <button
              ref={loginTabRef}
              type="button"
              className={`auth-tab ${isLogin ? "active" : ""}`}
              onClick={() => switchMode("login")}
            >
              Log in
            </button>
            <button
              ref={signupTabRef}
              type="button"
              className={`auth-tab ${!isLogin ? "active" : ""}`}
              onClick={() => switchMode("signup")}
            >
              Sign up
            </button>
            <span className="auth-tab-indicator" style={indicatorStyle} />
          </div>

          <div className="auth-form-panel">
            {isLogin ? (
              <form key="login" onSubmit={handleLogin} className="auth-form-enter">
                <div className="field">
                  <label htmlFor="login-email">Email</label>
                  <div className="input-with-icon">
                    <Mail size={16} className="input-icon" aria-hidden="true" />
                    <input
                      id="login-email"
                      type="email"
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      required
                      autoComplete="email"
                    />
                  </div>
                </div>
                <div className="field">
                  <label htmlFor="login-password">Password</label>
                  <div className="input-with-icon">
                    <Lock size={16} className="input-icon" aria-hidden="true" />
                    <input
                      id="login-password"
                      type="password"
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      required
                      autoComplete="current-password"
                    />
                  </div>
                </div>
                <button type="submit" className="btn btn-primary btn-block" disabled={submitting}>
                  {submitting ? "Logging in…" : "Log in"}
                </button>
                {signupSuccess && (
                  <div className="form-message success">
                    <span className="msg-icon">✓</span>
                    {signupSuccess}
                  </div>
                )}
                {error && (
                  <div className="form-message error">
                    <span className="msg-icon">⚠</span>
                    {error}
                  </div>
                )}
              </form>
            ) : (
              <form key="signup" onSubmit={handleSignup} className="auth-form-enter">
                <div className="field">
                  <label htmlFor="signup-email">Email</label>
                  <div className="input-with-icon">
                    <Mail size={16} className="input-icon" aria-hidden="true" />
                    <input
                      id="signup-email"
                      type="email"
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      required
                      autoComplete="email"
                    />
                  </div>
                </div>
                <div className="field">
                  <label htmlFor="signup-password">Password</label>
                  <div className="input-with-icon">
                    <Lock size={16} className="input-icon" aria-hidden="true" />
                    <input
                      id="signup-password"
                      type="password"
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      required
                      autoComplete="new-password"
                      minLength={8}
                    />
                  </div>
                  <p className="field-hint">At least 8 characters.</p>
                </div>
                <button type="submit" className="btn btn-primary btn-block" disabled={submitting}>
                  {submitting ? "Creating account…" : "Create account"}
                </button>
                {error && (
                  <div className="form-message error">
                    <span className="msg-icon">⚠</span>
                    {error}
                  </div>
                )}
              </form>
            )}
          </div>

          <p className="auth-switch-hint">
            {isLogin ? (
              <>
                New here? <button type="button" onClick={() => switchMode("signup")}>Create an account</button>
              </>
            ) : (
              <>
                Already have an account? <button type="button" onClick={() => switchMode("login")}>Log in</button>
              </>
            )}
          </p>
          </>
          )}
        </div>
        </div>
      </div>
    </div>
  );
}
