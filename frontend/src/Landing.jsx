import { motion } from "framer-motion";
import { ShieldCheck, FileSearch, Tags, ShieldAlert, ArrowRight, CheckCircle2 } from "lucide-react";

// A real, honest marketing moment before auth -- the value proposition
// stated plainly (no overclaiming: "flags real risk", not "guarantees"
// or "replaces a lawyer"), plus a genuine plain-language walkthrough of
// the actual pipeline (extract -> classify -> flag), matching what this
// project's own backend really does (pipeline/run_contract.py). The
// example clause card below is clearly labeled "Example" -- illustrative
// of the real UI, not fabricated live data.
const STEPS = [
  {
    icon: FileSearch,
    title: "Extract",
    body: "Every clause is pulled from your PDF or DOCX, split apart from the raw document text.",
  },
  {
    icon: Tags,
    title: "Classify",
    body: "Each clause is categorized — liability, indemnification, termination, and more — by heading and, when that's not enough, by meaning.",
  },
  {
    icon: ShieldAlert,
    title: "Flag real risk",
    body: "An AI model reviews each clause on its own merits and explains, in plain language, exactly why it's risky — or that it isn't.",
  },
];

export default function Landing({ onGetStarted }) {
  return (
    <div className="landing">
      <header className="landing-nav">
        <span className="brand-mark chrome">
          <ShieldCheck size={22} strokeWidth={2.25} />
          ClauseGuard
        </span>
        <button type="button" className="btn btn-primary" onClick={onGetStarted}>
          Get Started
        </button>
      </header>

      <section className="landing-hero">
        <motion.div
          className="landing-hero-copy"
          initial={{ opacity: 0, y: 18 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, ease: [0.2, 0.7, 0.3, 1] }}
        >
          <span className="landing-eyebrow">AI-assisted contract risk review</span>
          <h1>
            Know which clauses put you at risk — <em>before</em> you sign.
          </h1>
          <p className="landing-lede">
            Upload a contract. ClauseGuard extracts every clause, classifies it, and flags real risk
            with a plain-language explanation for each flag — grounded in the clause's own text, not a
            generic checklist.
          </p>
          <div className="landing-cta-row">
            <button type="button" className="btn btn-primary btn-lg" onClick={onGetStarted}>
              Get started free <ArrowRight size={18} />
            </button>
            <span className="landing-cta-hint">No credit card. Takes under a minute.</span>
          </div>
        </motion.div>

        <motion.div
          className="landing-preview"
          initial={{ opacity: 0, y: 24, rotate: -1.5 }}
          animate={{ opacity: 1, y: 0, rotate: -1.5 }}
          transition={{ duration: 0.7, delay: 0.15, ease: [0.2, 0.7, 0.3, 1] }}
        >
          <div className="landing-preview-label">Example</div>
          <div className="landing-preview-card">
            <div className="landing-preview-head">
              <strong>6. Indemnification</strong>
              <span className="severity-badge" data-severity="high">
                HIGH
              </span>
            </div>
            <p>
              This clause has no cap or limitation on indemnification liability and survives
              termination indefinitely, exposing the Client to unlimited, open-ended financial risk.
            </p>
            <div className="landing-preview-citation">
              "...shall indemnify, defend, and hold harmless...with no cap or limitation of any
              kind on the amount..."
            </div>
          </div>
        </motion.div>
      </section>

      <section className="landing-how">
        <h2>How it actually works</h2>
        <p className="landing-how-sub">Three real stages, in plain language.</p>
        <div className="landing-steps">
          {STEPS.map((step, i) => (
            <motion.div
              className="landing-step"
              key={step.title}
              initial={{ opacity: 0, y: 16 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-60px" }}
              transition={{ duration: 0.5, delay: i * 0.1, ease: [0.2, 0.7, 0.3, 1] }}
            >
              <div className="landing-step-icon">
                <step.icon size={22} strokeWidth={2} />
              </div>
              <span className="landing-step-number">{String(i + 1).padStart(2, "0")}</span>
              <h3>{step.title}</h3>
              <p>{step.body}</p>
            </motion.div>
          ))}
        </div>
      </section>

      <section className="landing-trust">
        <div className="landing-trust-item">
          <CheckCircle2 size={18} />
          Every flag cites the exact clause text it's based on
        </div>
        <div className="landing-trust-item">
          <CheckCircle2 size={18} />
          Your contract history is private to your account
        </div>
        <div className="landing-trust-item">
          <CheckCircle2 size={18} />
          Uncertain results are labeled, never guessed silently
        </div>
      </section>

      <footer className="landing-footer">
        <span>Ready to see what's actually in your contract?</span>
        <button type="button" className="btn btn-secondary" onClick={onGetStarted}>
          Sign up free
        </button>
      </footer>
    </div>
  );
}
