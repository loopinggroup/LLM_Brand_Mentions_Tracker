"""
Request/response evidence recording for Langdock support tickets.

Langdock support can only diagnose an Agent-API "model is not available" 400 from a
matching pair out of ONE run: the full GET /agent/v1/models response with its timestamp
and the key identity used, and the POST /agent/v1/chat/completions sent immediately
afterwards with its exact body, status code and unmodified response body.

brand_visibility.log cannot supply that: the GET response is parsed and dropped, the POST
body is never written, and error bodies go through _strip_html() (600 chars). So the pair
is recorded here instead, into its own JSONL file, in a form that can be sent on as-is.

Two rules this module exists to enforce:
  1. The API key is NEVER written — only key_fingerprint(), which is stable across calls
     and therefore proves "same key on GET and POST" without disclosing anything.
  2. Response bodies are recorded VERBATIM. Support explicitly asked for the unmodified
     body; truncating or prettifying it is what made the existing log useless to them.

Used by app.py (production runs) and support/langdock_repro.py (standalone reproduction).
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import uuid
from datetime import datetime, timezone


def key_fingerprint(api_key: str | None) -> str:
    """
    Stable, non-reversible identifier for an API key, safe to send to support.

    Format: "sha256:<12 hex>/…<last 4 chars of the key>". The hash alone proves two
    requests used the same key; the last 4 characters let a human match it against the
    key in their own key management without revealing the secret.
    """
    if not api_key:
        return "sha256:<none>"
    digest = hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:12]
    return f"sha256:{digest}/…{api_key[-4:]}"


def redact(text: str | None, api_key: str | None) -> str | None:
    """
    Safety net: strips the raw key should it ever be echoed back inside a body or header.
    Everything else is left byte-for-byte intact — support needs the unmodified body.
    """
    if not text or not api_key:
        return text
    return text.replace(api_key, "<REDACTED_API_KEY>")


def _stamps() -> dict:
    now = datetime.now(timezone.utc).astimezone()
    return {
        "ts_local": now.isoformat(timespec="milliseconds"),
        "ts_utc":   now.astimezone(timezone.utc).isoformat(timespec="milliseconds"),
    }


class EvidenceRecorder:
    """
    Appends one JSON object per HTTP request to a JSONL file.

    Every record carries run_id + seq, so a POST can be tied to the GET that produced its
    model ID even when several runs interleave in the same file. Writes are guarded by a
    lock because collection runs the Agent API from a thread pool.
    """

    def __init__(self, path: str, run_id: str | None = None) -> None:
        self.path   = path
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self._seq   = 0
        self._lock  = threading.Lock()
        self.records: list[dict] = []

    def record(self, kind: str, **fields) -> dict:
        """
        Writes one entry. `kind` is the step being captured, e.g. "GET /agent/v1/models"
        or "POST /agent/v1/chat/completions". Returns the stored record.
        """
        with self._lock:
            self._seq += 1
            entry = {"run_id": self.run_id, "seq": self._seq, "kind": kind, **_stamps(), **fields}
            self.records.append(entry)
            try:
                d = os.path.dirname(self.path)
                if d:
                    os.makedirs(d, exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            except OSError:
                # Evidence recording must never take a run down with it.
                pass
            return entry


def response_headers(resp) -> dict:
    """
    All response headers, verbatim.

    Deliberately not filtered: the instance-identifying ones (x-request-id, cf-ray,
    x-vercel-id, server, date) are exactly what shows that two contradictory catalog
    answers came from different backends, and which of them is worth keeping differs per
    CDN. Response headers carry no credentials — the request's Authorization header is
    never recorded anywhere in this module.
    """
    try:
        return dict(resp.headers)
    except Exception:
        return {}
