// Thin wrappers around the ClauseGuard backend. Plain fetch() is used on
// purpose, not axios: fetch has NO default timeout at all (axios defaults
// to 0/no-timeout too, but fetch needs zero configuration to get that).
// Upload itself returns near-instantly now (M15/M16 -- see uploadContract
// below), so this no longer matters for the upload call the way it used
// to under M14's synchronous flow, but there's still no reason to add a
// timeout that could bite later.

const BASE_URL = "http://localhost:8010";

async function parseJsonOrThrow(response) {
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = body && body.detail ? body.detail : response.statusText;
    const err = new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    // Attached so callers (M16's status-polling loop) can distinguish a
    // 404 -- tolerated briefly during the theoretical race-condition
    // window right after upload -- from any other failure, which should
    // stop polling and surface as a real error immediately.
    err.status = response.status;
    throw err;
  }
  return body;
}

export async function signup(email, password) {
  // New function -- POST /signup has existed on the backend since M2 but
  // never had a frontend caller until now. Mirrors login()'s own shape
  // exactly (same request pattern, same parseJsonOrThrow error handling)
  // rather than inventing a different convention for the one new endpoint.
  const response = await fetch(`${BASE_URL}/signup`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });
  return parseJsonOrThrow(response); // { id, email }
}

export async function login(email, password) {
  const response = await fetch(`${BASE_URL}/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });
  return parseJsonOrThrow(response); // { access_token, token_type }
}

// M39: email OTP verification, added after signup. Same fetch/
// parseJsonOrThrow shape as every other call in this file.
export async function verifyOtp(email, otp) {
  const response = await fetch(`${BASE_URL}/verify-otp`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, otp }),
  });
  return parseJsonOrThrow(response); // { message }
}

export async function resendOtp(email) {
  const response = await fetch(`${BASE_URL}/resend-otp`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email }),
  });
  return parseJsonOrThrow(response); // { message }
}

export async function uploadContract(file, token) {
  const formData = new FormData();
  formData.append("file", file);

  // M16: this now returns near-instantly (status='processing') -- the
  // actual pipeline runs in a separate Celery worker (M15). The lack of
  // a timeout here is no longer load-bearing for a multi-minute wait the
  // way it was in M14, but is kept anyway since there's no reason to
  // impose one.
  const response = await fetch(`${BASE_URL}/contracts/upload`, {
    method: "POST",
    headers: { Authorization: `Bearer ${token}` },
    body: formData,
  });
  return parseJsonOrThrow(response); // { id, filename, status }
}

export async function getContractStatus(contractId, token) {
  const response = await fetch(`${BASE_URL}/contracts/${contractId}/status`, {
    method: "GET",
    headers: { Authorization: `Bearer ${token}` },
  });
  return parseJsonOrThrow(response); // { id, status, clauses_persisted, clauses_flagged }
}

export async function getFlaggedClauses(contractId, token) {
  const response = await fetch(`${BASE_URL}/contracts/${contractId}/flagged-clauses`, {
    method: "GET",
    headers: { Authorization: `Bearer ${token}` },
  });
  return parseJsonOrThrow(response); // { contract_id, status, flagged_clauses: [...] }
}

// M38 ("My Contracts" history): lists the authenticated user's own past
// uploads, most recent first. Mirrors every other call in this file --
// same fetch/Authorization/parseJsonOrThrow pattern, no new convention.
export async function getContracts(token, { limit, offset } = {}) {
  const params = new URLSearchParams();
  if (limit != null) params.set("limit", limit);
  if (offset != null) params.set("offset", offset);
  const qs = params.toString();
  const response = await fetch(`${BASE_URL}/contracts${qs ? `?${qs}` : ""}`, {
    method: "GET",
    headers: { Authorization: `Bearer ${token}` },
  });
  return parseJsonOrThrow(response); // { contracts: [...], limit, offset, total }
}
