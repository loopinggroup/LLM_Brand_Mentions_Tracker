#!/usr/bin/env python3
"""
Standalone reproduction of the Langdock Agent API "model is not available" 400.

Produces exactly the material Langdock support asked for, out of ONE run and with ONE key:

    A) GET  /agent/v1/models          → timestamp, key identity, status, full raw body
    B) POST /agent/v1/chat/completions → sent immediately after A, for each data[].id
                                         taken VERBATIM from A's response; exact request
                                         body, status code, unmodified response body
    C) GET  /agent/v1/models           → same key, same host, at the end of the run, so
                                         catalog drift within one run is visible in the
                                         same document

Each model ID is tried several times, because the failure is intermittent: the same ID
against the same endpoint with the same key returns both 200 and 400 depending on which
backend answers. A single attempt per ID would not show that.

Costs are negligible: the prompt is "ping" and, on a 200, the stream is closed as soon as
the status line and the first events have arrived — the run never waits for a full answer.

No Streamlit, no project imports beyond langdock_evidence.py, so support can be handed the
script itself if they want to run it on their side.

Usage:
    python support/langdock_repro.py                    # key from LANGDOCK_API_KEY or key.txt
    python support/langdock_repro.py --repeats 8        # more attempts per Anthropic model
    python support/langdock_repro.py --all-models       # every model, not just Anthropic + controls
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import uuid
from collections import defaultdict
from datetime import datetime

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langdock_evidence import (  # noqa: E402
    EvidenceRecorder,
    key_fingerprint,
    redact,
    response_headers,
)

AGENT_MODELS_URL = "https://api.langdock.com/agent/v1/models"
AGENT_URL        = "https://api.langdock.com/agent/v1/chat/completions"

REPO_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVIDENCE_DIR = os.path.join(REPO_ROOT, "support", "evidence")

# Only headers are needed to establish the status code; a 200 stream is abandoned right
# after that, so the read timeout never has to cover a full generation.
TIMEOUT = (10, 45)

# Max characters of a 200 stream kept in the transcript. A 400 body is ALWAYS stored in
# full — it is the piece of evidence the ticket is about.
OK_BODY_CHARS = 2000


# ---------------------------------------------------------------------------
# Key handling
# ---------------------------------------------------------------------------
def load_api_key() -> str:
    """Reads the key from LANGDOCK_API_KEY, else from key.txt ('KEY = value' or bare)."""
    env = os.environ.get("LANGDOCK_API_KEY", "").strip()
    if env:
        return env

    path = os.path.join(REPO_ROOT, "key.txt")
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # "LANGDOCK_API_KEY = sk-..." or just "sk-..."
                return line.split("=", 1)[1].strip() if "=" in line else line
    except OSError:
        pass

    sys.exit("No API key found. Set LANGDOCK_API_KEY or put the key in key.txt.")


# ---------------------------------------------------------------------------
# Step A / C — GET /agent/v1/models
# ---------------------------------------------------------------------------
def fetch_catalog(rec: EvidenceRecorder, api_key: str, label: str) -> tuple[list[str], str]:
    """
    Records the GET verbatim and returns (ids, raw_body). IDs are data[].id, untouched.
    """
    t0 = time.time()
    try:
        r = requests.get(
            AGENT_MODELS_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=TIMEOUT,
        )
    except requests.exceptions.RequestException as e:
        rec.record(
            "GET /agent/v1/models",
            label=label,
            url=AGENT_MODELS_URL,
            key_fingerprint=key_fingerprint(api_key),
            error=str(e),
        )
        return [], ""

    raw = redact(r.text, api_key) or ""
    rec.record(
        "GET /agent/v1/models",
        label=label,
        url=AGENT_MODELS_URL,
        key_fingerprint=key_fingerprint(api_key),
        http_status=r.status_code,
        elapsed_s=round(time.time() - t0, 3),
        response_headers=response_headers(r),
        response_body_raw=raw,          # complete and unmodified, as requested
    )

    ids: list[str] = []
    try:
        for m in (r.json().get("data") or []):
            if isinstance(m, dict) and m.get("id"):
                ids.append(m["id"])     # verbatim — never rewritten
    except ValueError:
        pass
    return ids, raw


# ---------------------------------------------------------------------------
# Step B — POST /agent/v1/chat/completions
# ---------------------------------------------------------------------------
def build_payload(model: str) -> dict:
    """
    Same body shape the production app sends (app.py, call_langdock_agent) — inline agent
    object with capabilities.webSearch, UI-message parts, streaming — with a trivial
    prompt. `model` is the string as it came out of GET /agent/v1/models.
    """
    return {
        "agent": {
            "name":         "Langdock support reproduction",
            "instructions": "Reply with the single word: pong.",
            "model":        model,
            "capabilities": {"webSearch": True},
        },
        "messages": [{"id": "msg_0", "role": "user", "parts": [{"type": "text", "text": "ping"}]}],
        "stream":   True,
    }


def post_completion(rec: EvidenceRecorder, api_key: str, model: str, attempt: int,
                    catalog_seq: int) -> int | None:
    """
    Sends one completion request and records the full triple. Returns the status code.
    `catalog_seq` ties this POST to the GET whose catalog supplied `model`.
    """
    payload = build_payload(model)
    t0 = time.time()

    try:
        with requests.post(
            AGENT_URL,
            json=payload,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            timeout=TIMEOUT,
            stream=True,
        ) as r:
            if r.status_code >= 400:
                body      = redact(r.text, api_key) or ""
                truncated = False
            else:
                # Success only needs to be evidenced, not completed: take the opening of
                # the stream and drop the connection so the run stays fast and cheap.
                chunks, size = [], 0
                for line in r.iter_lines(decode_unicode=True):
                    if line:
                        chunks.append(line)
                        size += len(line)
                    if size >= OK_BODY_CHARS:
                        truncated = True
                        break
                else:
                    truncated = False
                body = redact("\n".join(chunks), api_key) or ""

            rec.record(
                "POST /agent/v1/chat/completions",
                url=AGENT_URL,
                key_fingerprint=key_fingerprint(api_key),
                model_requested=model,
                from_catalog_seq=catalog_seq,
                attempt=attempt,
                request_body=payload,              # exact body as sent
                http_status=r.status_code,
                elapsed_s=round(time.time() - t0, 3),
                response_headers=response_headers(r),
                response_body_raw=body,            # unmodified (400s always complete)
                response_body_truncated=truncated,
            )
            return r.status_code

    except requests.exceptions.RequestException as e:
        rec.record(
            "POST /agent/v1/chat/completions",
            url=AGENT_URL,
            key_fingerprint=key_fingerprint(api_key),
            model_requested=model,
            from_catalog_seq=catalog_seq,
            attempt=attempt,
            request_body=payload,
            error=str(e),
            elapsed_s=round(time.time() - t0, 3),
        )
        return None


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def _is_anthropic(model_id: str) -> bool:
    return "claude" in model_id.lower() or "anthropic" in model_id.lower()


def _available_models_from_body(body: str) -> list[str]:
    """Pulls the catalog out of an 'available models are: …' 400 body."""
    m = re.search(r"available models are:\s*([^\"}]+)", body or "", re.IGNORECASE)
    return [p.strip().strip(".\"'} ") for p in m.group(1).split(",")] if m else []


def _family(model_id: str) -> str:
    """
    Strips deployment decoration so two spellings of the SAME model compare equal:
        claude-opus-4-8@default                     → claude-opus-4-8
        eu.anthropic.claude-opus-4-5-20251101-v1:0  → claude-opus-4-5

    Used for the report only — never to choose a value to send. Its whole purpose is to
    show that a rejected ID and an offered ID denote one model under two spellings.
    """
    m = model_id.strip().lower()
    m = m.split("@", 1)[0]
    m = re.sub(r"^(eu|us|apac|global)\.", "", m)
    m = re.sub(r"^(anthropic|openai|google|meta|amazon)\.", "", m)
    m = re.sub(r"-(eu|us|apac|global)$", "", m)
    m = re.sub(r"-v\d+(:\d+)?$", "", m)
    m = re.sub(r"-\d{8}$", "", m)
    return m


def _variant_offered(requested: str, offered: list[str]) -> list[str]:
    """Entries in `offered` that are the same model as `requested`, spelled differently."""
    fam = _family(requested)
    return [o for o in offered if _family(o) == fam and o != requested]


def write_report(rec: EvidenceRecorder, path: str, api_key: str, jsonl_path: str) -> None:
    gets  = [r for r in rec.records if r["kind"].startswith("GET")]
    posts = [r for r in rec.records if r["kind"].startswith("POST")]

    tally: dict[str, dict[int | str, int]] = defaultdict(lambda: defaultdict(int))
    for p in posts:
        tally[p["model_requested"]][p.get("http_status") or "error"] += 1

    out: list[str] = []
    w = out.append

    w("# Langdock Agent API — Reproduktion 400 „Model provided (…) is not available“")
    w("")
    w(f"- **Run-ID:** `{rec.run_id}`")
    w(f"- **Erstellt:** {datetime.now().astimezone().isoformat(timespec='seconds')}")
    w(f"- **Key-Kennung:** `{key_fingerprint(api_key)}` — identisch für **alle** unten "
      "aufgeführten Requests (GET wie POST). Der Key selbst ist nirgends enthalten.")
    w(f"- **Hosts:** `{AGENT_MODELS_URL}` und `{AGENT_URL}` — beides `api.langdock.com`, "
      "ein Prozess, eine Session.")
    w(f"- **Vollständiger Mitschnitt (JSONL, ungekürzt):** `{os.path.basename(jsonl_path)}`")
    w("")
    w("Alle `model`-Werte stammen unverändert aus `data[].id` des ersten GET. Es findet "
      "keinerlei Umschreibung statt — kein Kürzen, kein Ergänzen oder Entfernen von "
      "`@default`/Versionsständen, keine Anzeigenamen.")
    w("")

    # --- Ergebnisübersicht ---------------------------------------------------
    w("## 1. Ergebnisübersicht")
    w("")
    w("| Modell-ID (verbatim aus `data[].id`) | 200 | 400 | sonstige |")
    w("|---|---|---|---|")
    for model in sorted(tally):
        counts = tally[model]
        other  = sum(v for k, v in counts.items() if k not in (200, 400))
        w(f"| `{model}` | {counts.get(200, 0)} | {counts.get(400, 0)} | {other or ''} |")
    w("")

    flapping = [m for m, c in tally.items() if c.get(200, 0) and c.get(400, 0)]
    if flapping:
        w("**Befund 1 — dieselbe ID liefert 200 *und* 400.** Für die folgenden IDs "
          "antwortet derselbe Endpunkt mit demselben Key und derselben, unveränderten "
          "ID unterschiedlich:")
        w("")
        for m in sorted(flapping):
            w(f"- `{m}` — {tally[m].get(200,0)}× 200, {tally[m].get(400,0)}× 400")
        w("")
        w("Eine Schreibweise, die die API grundsätzlich nicht akzeptiert, käme auf 0× 200. "
          "Der Fehler hängt damit nicht am gesendeten String, sondern daran, welche Instanz "
          "die jeweilige Anfrage beantwortet.")
        w("")

    # Der direkteste Beleg: die ablehnende Antwort bietet dasselbe Modell selbst an —
    # nur anders geschrieben. Damit ist „die ID wurde verändert“ als Ursache ausgeschlossen.
    mismatches = []
    for p in posts:
        if p.get("http_status") == 400:
            offered = _available_models_from_body(p.get("response_body_raw", ""))
            variants = _variant_offered(p["model_requested"], offered)
            if variants:
                mismatches.append((p["seq"], p["model_requested"], variants))

    if mismatches:
        w(f"**Befund {2 if flapping else 1} — die ablehnende Antwort bietet dasselbe Modell "
          "unter anderer Schreibweise an.** Die angefragte ID stammt unverändert aus "
          "`GET /agent/v1/models`; die Instanz, die den POST beantwortet, kennt sie nicht, "
          "listet in derselben Antwort aber eine andere Schreibweise desselben Modells:")
        w("")
        w("| seq | gesendet (aus `data[].id`) | von der 400-Antwort stattdessen angeboten |")
        w("|---|---|---|")
        for seq, requested, variants in mismatches:
            w(f"| {seq} | `{requested}` | " + ", ".join(f"`{v}`" for v in variants) + " |")
        w("")
        w("Die beiden Spalten bezeichnen jeweils dasselbe Modell. Der Katalog aus dem GET "
          "und der Katalog der antwortenden Completion-Instanz stimmen also nicht überein.")
        w("")

    if not flapping and not mismatches:
        w("**Befund:** In diesem Durchlauf weder gemischte Ergebnisse pro ID noch "
          "abweichende Schreibweisen in den 400-Antworten. Siehe Abschnitt 3.")
        w("")

    # --- Katalog-Differenzen -------------------------------------------------
    w("## 2. Katalog-Vergleich innerhalb desselben Durchlaufs")
    w("")
    catalogs: list[tuple[str, set[str]]] = []
    for g in gets:
        try:
            ids = {m["id"] for m in (json.loads(g.get("response_body_raw") or "{}").get("data") or [])
                   if isinstance(m, dict) and m.get("id")}
        except ValueError:
            ids = set()
        catalogs.append((f"GET #{g['seq']} ({g['ts_local']})", ids))

    for label, ids in catalogs:
        w(f"- **{label}** — {len(ids)} Modelle")
    if len(catalogs) >= 2:
        first, last = catalogs[0][1], catalogs[-1][1]
        only_first, only_last = sorted(first - last), sorted(last - first)
        if only_first or only_last:
            w("")
            w("Unterschiede zwischen erstem und letztem Abruf:")
            for m in only_first:
                w(f"  - nur im ersten Abruf: `{m}`")
            for m in only_last:
                w(f"  - nur im letzten Abruf: `{m}`")
        else:
            w("")
            w("Beide Abrufe listen dieselben IDs.")

    # Kataloge, die die 400-Bodies selbst aufzählen — der direkteste Beleg.
    error_catalogs = []
    for p in posts:
        if p.get("http_status") == 400:
            avail = _available_models_from_body(p.get("response_body_raw", ""))
            if avail:
                error_catalogs.append((p["seq"], p["ts_local"], p["model_requested"], avail))
    if len(error_catalogs) >= 2:
        w("")
        w("Die 400-Bodies zählen ihrerseits einen Katalog auf. Diese Aufzählungen sind "
          "zwischen den Antworten nicht identisch:")
        w("")
        for seq, ts, model, avail in error_catalogs[:6]:
            uniq = [a for a in avail if _is_anthropic(a)]
            w(f"- `seq {seq}` {ts} (angefragt: `{model}`) → Anthropic-Einträge: "
              + ", ".join(f"`{a}`" for a in uniq))
    w("")

    # --- Vollständige Paare --------------------------------------------------
    w("## 3. Vollständige Request-/Response-Paare")
    w("")
    w("Reihenfolge und Zeitstempel wie gesendet. `seq` entspricht dem Eintrag im JSONL.")
    w("")

    for r in rec.records:
        w(f"### seq {r['seq']} — {r['kind']}")
        w("")
        w(f"- **Zeitpunkt:** {r['ts_local']}  (UTC: {r['ts_utc']})")
        w(f"- **Key-Kennung:** `{r['key_fingerprint']}`")
        w(f"- **URL:** `{r['url']}`")
        if "model_requested" in r:
            w(f"- **Gesendeter `model`-Wert:** `{r['model_requested']}` "
              f"(unverändert aus `data[].id` von `seq {r['from_catalog_seq']}`, "
              f"Versuch {r['attempt']})")
        if "error" in r:
            w(f"- **Transportfehler:** `{r['error']}`")
            w("")
            continue
        w(f"- **HTTP-Status:** `{r['http_status']}`  ({r['elapsed_s']} s)")
        for h in ("date", "x-request-id", "x-vercel-id", "cf-ray", "server"):
            for k, v in (r.get("response_headers") or {}).items():
                if k.lower() == h:
                    w(f"- **`{k}`:** `{v}`")
        w("")
        if "request_body" in r:
            w("**Gesendeter Body:**")
            w("")
            w("```json")
            w(json.dumps(r["request_body"], ensure_ascii=False, indent=2))
            w("```")
            w("")
        body = r.get("response_body_raw", "")
        if r.get("http_status") == 400 and "model_requested" in r:
            variants = _variant_offered(r["model_requested"], _available_models_from_body(body))
            if variants:
                w(f"- ⚠️ Die Antwort lehnt `{r['model_requested']}` ab und listet gleichzeitig "
                  + ", ".join(f"`{v}`" for v in variants)
                  + " — dasselbe Modell, andere Schreibweise.")
                w("")
        note = " (Anfang des Streams; bei 200 nach dem Statusnachweis abgebrochen)" \
            if r.get("response_body_truncated") else ""
        w(f"**Response-Body, unverändert{note}:**")
        w("")
        w("```")
        w(body if body else "<leer>")
        w("```")
        w("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repeats", type=int, default=5,
                    help="Versuche pro Anthropic-Modell (Standard: 5)")
    ap.add_argument("--control-repeats", type=int, default=2,
                    help="Versuche pro Nicht-Anthropic-Kontrollmodell (Standard: 2)")
    ap.add_argument("--controls", type=int, default=2,
                    help="Anzahl Nicht-Anthropic-Kontrollmodelle (Standard: 2)")
    ap.add_argument("--all-models", action="store_true",
                    help="Alle Modelle testen statt Anthropic + Kontrollgruppe")
    args = ap.parse_args()

    api_key = load_api_key()
    run_id  = uuid.uuid4().hex[:12]
    stamp   = datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    jsonl   = os.path.join(EVIDENCE_DIR, f"langdock_repro_{stamp}.jsonl")
    report  = os.path.join(EVIDENCE_DIR, f"langdock_repro_{stamp}.md")

    rec = EvidenceRecorder(jsonl, run_id=run_id)
    print(f"Run {run_id} | Key {key_fingerprint(api_key)}")
    print(f"Transcript: {jsonl}\n")

    # --- A ---------------------------------------------------------------
    print("[A] GET /agent/v1/models …")
    ids, _ = fetch_catalog(rec, api_key, label="vor den Completion-Requests")
    catalog_seq = rec.records[-1]["seq"]
    if not ids:
        sys.exit("GET /agent/v1/models lieferte keine IDs — siehe Transcript.")
    print(f"    {len(ids)} Modelle: {', '.join(ids)}\n")

    if args.all_models:
        plan = [(m, args.repeats) for m in ids]
    else:
        anthropic = [m for m in ids if _is_anthropic(m)]
        controls  = [m for m in ids if not _is_anthropic(m)][:args.controls]
        plan = [(m, args.repeats) for m in anthropic] + \
               [(m, args.control_repeats) for m in controls]

    total = sum(n for _, n in plan)
    print(f"[B] POST /agent/v1/chat/completions — {len(plan)} Modelle, {total} Requests")
    done = 0
    for model, repeats in plan:
        for attempt in range(1, repeats + 1):
            status = post_completion(rec, api_key, model, attempt, catalog_seq)
            done += 1
            flag = "OK " if status == 200 else f"!! {status}"
            print(f"    [{done:>3}/{total}] {flag}  {model}  (Versuch {attempt}/{repeats})")

    # --- C ---------------------------------------------------------------
    print("\n[C] GET /agent/v1/models (Kontrollabruf am Ende) …")
    fetch_catalog(rec, api_key, label="nach den Completion-Requests")

    write_report(rec, report, api_key, jsonl)

    # Letzte Sicherung: der Key darf in keiner Ausgabedatei stehen.
    for path in (jsonl, report):
        with open(path, encoding="utf-8") as f:
            if api_key in f.read():
                sys.exit(f"ABBRUCH: API-Key steht in {path} — Datei NICHT versenden.")

    print(f"\nFertig.\n  Bericht:    {report}\n  Transcript: {jsonl}")
    print("  Key-Prüfung: bestanden (kein Klartext-Key in den Ausgabedateien).")


if __name__ == "__main__":
    main()
