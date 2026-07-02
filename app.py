"""
LLM Brand Visibility — Streamlit App (Langdock Edition)
=========================================================
Based on brand_monitor.py and analyze_csv.py.
Uses the Langdock API directly via requests (no OpenAI SDK needed).

Installation:
    pip install streamlit requests pandas plotly

Start:
    streamlit run app.py
"""

import csv
import json
import logging
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Logging
# Uses a guard to prevent duplicate handlers when Streamlit reruns the script.
# Without this, each Streamlit rerun adds another handler and every log line
# gets written multiple times.
# ---------------------------------------------------------------------------
log = logging.getLogger("brand_visibility")
if not log.handlers:
    log.setLevel(logging.INFO)
    _fh = logging.FileHandler("brand_visibility.log")
    _fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    log.addHandler(_fh)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="LLM Brand Visibility",
    page_icon="📊",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session state
# Streamlit reruns the whole script on every user interaction.
# Session state persists values across those reruns.
# ---------------------------------------------------------------------------
def init_state():
    defaults = {
        "step":           1,
        "config":         {},
        "questions":      [],
        "raw_answers":    [],   # phase 1 output — Q&A only, no brand analysis
        "results":        [],   # phase 2 output — with brands + sentiment
        "stop_requested": False,
        "timing":         {},   # phase wall times + per-call elapsed data
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ---------------------------------------------------------------------------
# Langdock API configuration
# ---------------------------------------------------------------------------
LANGDOCK_REGION     = os.environ.get("LANGDOCK_REGION", "eu")
LANGDOCK_URL        = f"https://api.langdock.com/openai/{LANGDOCK_REGION}/v1/chat/completions"
ANTHROPIC_URL       = f"https://api.langdock.com/anthropic/{LANGDOCK_REGION}/v1/messages"
REQUEST_TIMEOUT     = 120   # gpt-5-mini (reasoning) regularly takes 30-56s; 60s caused unnecessary retries
MAX_TOKENS          = 2000  # answer calls — default; user can raise up to 16000 in the UI
ANALYSIS_MAX_TOKENS = 4000  # brand analysis JSON — reasoning models burn thinking tokens, 1000 was too tight
QUESTION_MAX_TOKENS = 16000 # question generation: reasoning model uses token budget for internal thinking too

# ---------------------------------------------------------------------------
# Available models grouped by provider
# ---------------------------------------------------------------------------
AVAILABLE_MODELS = {
    "OpenAI": [
        "gpt-5-mini",
        "gpt-5.1",
        "gpt-5.2",
        "gpt-5.2-pro",
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.5",
        "o3",
        "o4-mini",
    ],
    "Anthropic": [
        "claude-fable-5",
        "claude-opus-4-8",
        "claude-opus-4-7",
        "claude-opus-4-6",
        "claude-sonnet-4-6",
        "claude-haiku-4-5",
    ],
    "Meta (via Langdock)": [
        "langdock-llama-3.3-70b-2",
    ],
}

_MODEL_OPTIONS = (
    [f"{provider} — {m}" for provider, models in AVAILABLE_MODELS.items() for m in models]
    + ["Benutzerdefiniert..."]
)

def _model_id_from_option(option: str) -> str:
    return option.split(" — ", 1)[1] if " — " in option else option

def _is_anthropic_model(model: str) -> bool:
    return model.startswith("claude-")


# ---------------------------------------------------------------------------
# Core API call wrapper
# Single function for all Langdock requests.
# Returns the response text on success, or None on any failure.
# Logs the full HTTP response body on error so failures are diagnosable.
# Does NOT call st.error() — error surfacing is handled by the caller.
# ---------------------------------------------------------------------------
def call_langdock(
    api_key: str,
    messages: list[dict],
    model: str,
    max_tokens: int = MAX_TOKENS,
    web_search: bool = False,
) -> tuple[str | None, str | None, dict]:
    """
    Returns (response_text, error_message, usage).
    On success: (text, None, {"prompt_tokens": N, "completion_tokens": N, "total_tokens": N})
    On failure: (None, human-readable error string, {})
    """
    anthropic = _is_anthropic_model(model)
    url = ANTHROPIC_URL if anthropic else LANGDOCK_URL
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }
    if anthropic:
        payload = {
            "model":      model,
            "messages":   messages,
            "max_tokens": max_tokens,
        }
    else:
        payload = {
            "model":                 model,
            "messages":              messages,
            "max_completion_tokens": max_tokens,
        }
    # web_search is handled via prompt instruction — Langdock's completions API
    # only supports type:"function" tools, not built-in search tool types.

    for attempt in range(4):
        try:
            t0 = time.time()
            r = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=REQUEST_TIMEOUT,
            )
            elapsed = time.time() - t0

            log.info(
                "HTTP %s | attempt %d | model=%s | tokens=%d | %.2fs",
                r.status_code, attempt + 1, model, max_tokens, elapsed,
            )

            r.raise_for_status()
            resp_data = r.json()

            if anthropic:
                content_blocks = resp_data.get("content", [])
                text   = content_blocks[0].get("text") if content_blocks else None
                raw_u  = resp_data.get("usage", {})
                usage  = {
                    "prompt_tokens":     raw_u.get("input_tokens", 0),
                    "completion_tokens": raw_u.get("output_tokens", 0),
                    "total_tokens":      raw_u.get("input_tokens", 0) + raw_u.get("output_tokens", 0),
                }
                finish = resp_data.get("stop_reason", "?")
            else:
                message = resp_data["choices"][0]["message"]
                text    = message.get("content")
                usage   = resp_data.get("usage", {})
                finish  = resp_data["choices"][0].get("finish_reason", "?")

            if not text:
                # Reasoning models (e.g. gpt-5-mini / o-series) return null content
                # when the token budget was exhausted by internal reasoning.
                # Log the full choice so the cause is diagnosable.
                log.warning(
                    "Null/empty content — finish_reason=%s | tokens=%d",
                    finish, max_tokens,
                )
                # Anthropic uses "max_tokens"; OpenAI uses "length"
                if finish in ("max_tokens", "length"):
                    return None, (
                        f"Token-Budget erschöpft (max_tokens={max_tokens}). "
                        "Für Reasoning-Modelle wie gpt-5-mini werden mehr Tokens benötigt."
                    ), usage
                return None, f"Modell lieferte keinen Inhalt (finish_reason={finish}).", usage

            return text, None, usage

        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else "?"
            body   = _strip_html(e.response.text) if e.response is not None else ""
            log.error("HTTP error %s: %s", status, body)

            if status == 429:
                # TPM/RPM rate limit. Waits must cover the ~60s rolling window;
                # jitter breaks up thundering herd when many parallel calls hit 429 together.
                wait = 15 * (2 ** attempt) + random.uniform(0, 5)  # 15-20s, 30-35s, 60-65s, 120-125s
                log.info("Rate limited (429), waiting %.1fs before retry (attempt %d/4)", wait, attempt + 1)
                time.sleep(wait)
                continue

            # Map common status codes to readable messages
            messages_map = {
                401: "API-Key ungültig oder abgelaufen (401).",
                403: "Zugriff verweigert (403). API-Key prüfen.",
                404: f"Endpunkt nicht gefunden (404). Region prüfen: '{LANGDOCK_REGION}'. URL: {url}",
                422: f"Ungültige Anfrage (422) — wahrscheinlich falscher Modell-Name: '{model}'. Antwort: {body}",
            }
            err = messages_map.get(status, f"HTTP-Fehler {status}: {body}")
            return None, err, {}

        except requests.exceptions.ConnectionError as e:
            err = f"Keine Verbindung zur Langdock API. Internet-Verbindung prüfen. ({e})"
            log.error("ConnectionError: %s", e)
            return None, err, {}

        except requests.exceptions.Timeout:
            if attempt < 2:
                log.warning("Timeout, retrying (attempt %d)", attempt + 1)
                time.sleep(2 ** attempt)
                continue
            err = f"Timeout nach {REQUEST_TIMEOUT}s."
            log.error(err)
            return None, err, {}

        except (KeyError, IndexError, json.JSONDecodeError) as e:
            err = f"Unerwartetes Antwort-Format: {e}"
            log.error("Parse error: %s | Response: %s", e,
                      r.text[:200] if 'r' in dir() else "no response")
            return None, err, {}

        except Exception as e:
            err = f"Unbekannter Fehler: {e}"
            log.error("Unexpected error: %s", e)
            return None, err, {}

    return None, "Maximale Anzahl an Versuchen erreicht (4).", {}


# ---------------------------------------------------------------------------
# Connection test
# Makes a minimal API call to verify credentials and model name.
# Shown in Step 1 so problems are caught before a long run starts.
# ---------------------------------------------------------------------------
def test_connection(api_key: str, model: str) -> tuple[bool, str]:
    """
    Minimal call to verify credentials and model name.
    Uses the same payload shape as brand_monitor.py.
    """
    text, err, _ = call_langdock(
        api_key,
        [{"role": "user", "content": "Say 'OK' and nothing else."}],
        model=model,
        max_tokens=MAX_TOKENS,  # use same value as main calls, not a small number
    )
    if text:
        return True, f"Verbindung erfolgreich. Modell antwortet: '{text.strip()[:80]}'"
    return False, err or "Keine Antwort erhalten."


# ---------------------------------------------------------------------------
# Question generation (Step 1 API call)
# Sends the topic and asks the model to generate N questions.
# Returns a list of question strings, one per line.
# ---------------------------------------------------------------------------
def generate_questions(api_key: str, topic: str, n: int, model: str) -> tuple[list[str], str | None]:
    prompt = (
        f"Generiere genau {n} verschiedene, neutrale Fragen zum Thema: \"{topic}\".\n\n"
        "Die Fragen sollen:\n"
        "- Typische Nutzerfragen sein, die jemand einem KI-Assistenten stellen würde\n"
        "- Verschiedene Aspekte des Themas abdecken (Empfehlungen, Vergleiche, Eigenschaften, Use Cases)\n"
        "- So formuliert sein, dass die Antwort natürlicherweise Marken, Produkte oder Anbieter nennen würde\n\n"
        "Antworte NUR mit den Fragen, eine pro Zeile, ohne Nummerierung, ohne Einleitung."
    )
    text, err, _ = call_langdock(
        api_key,
        [{"role": "user", "content": prompt}],
        model=model,
        max_tokens=QUESTION_MAX_TOKENS,  # reasoning models consume the budget for internal thinking + output
    )
    if not text:
        return [], err
    questions = [q.strip() for q in text.strip().splitlines() if q.strip()]
    return questions, None


# ---------------------------------------------------------------------------
# Main question run (Step 3 API calls)
# Asks a single question in the context of the topic.
# Returns (answer_text, error_message).
# ---------------------------------------------------------------------------
def ask_question(
    api_key: str, topic: str, question: str, model: str,
    web_search: bool = False, max_tokens: int = MAX_TOKENS,
    short_answer: bool = False,
) -> tuple[str | None, str | None, dict]:
    web_note = (
        "Nutze dein aktuellstes Wissen und beziehe dich wenn möglich auf aktuelle Entwicklungen und Informationen.\n"
        if web_search else ""
    )
    if short_answer:
        content = (
            f"Du bist ein Assistent mit Expertise zum Thema: {topic}.\n"
            f"{web_note}"
            "Antworte ausschließlich mit einer stichpunktartigen Auflistung. "
            "Jede Zeile: Markenname — ein Satz Begründung. "
            "Keine Einleitung, kein Fazit, kein Fließtext.\n\n"
            f"Frage: {question}"
        )
    else:
        content = (
            f"Du bist ein hilfreicher, neutraler Assistent mit Expertise zum Thema: {topic}.\n"
            f"{web_note}"
            f"Beantworte die folgende Frage sachlich und ausführlich. "
            f"Nenne konkrete Marken, Produkte oder Anbieter wo relevant.\n\n"
            f"Frage: {question}"
        )
    return call_langdock(
        api_key,
        [{"role": "user", "content": content}],
        model=model,
        max_tokens=max_tokens,
        web_search=web_search,
    )


# ---------------------------------------------------------------------------
# HTML stripper — removes tags and collapses whitespace for clean log output.
# Used so 502 / 503 gateway error bodies don't spam logs with raw HTML.
# ---------------------------------------------------------------------------
def _strip_html(text: str, max_len: int = 200) -> str:
    clean = re.sub(r"<[^>]+>", " ", text)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean[:max_len]


# ---------------------------------------------------------------------------
# Robust JSON-array extractor
# Tries three strategies in order:
#   1. Direct parse (model returned clean JSON)
#   2. Strip markdown code fences (```json ... ```)
#   3. Find the first [...] block that parses as a list
# Logs a warning and returns [] on total failure.
# ---------------------------------------------------------------------------
def _parse_json_array(text: str) -> list:
    # 1. Direct parse
    stripped = text.strip()
    if stripped.startswith("["):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

    # 2. Markdown code fence
    fence = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", text)
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            pass

    # 3. Find first [...] that is valid JSON — scan character by character
    #    to find balanced brackets instead of using a greedy regex
    start = text.find("[")
    while start != -1:
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        result = json.loads(candidate)
                        if isinstance(result, list):
                            return result
                    except json.JSONDecodeError:
                        pass
                    break
        start = text.find("[", start + 1)

    log.warning("Brand JSON parse failed. Response: %s", text[:300])
    return []


# ---------------------------------------------------------------------------
# Unified brand analysis (replaces manual + auto extraction)
# Sends full answer + question context to the model.
# Returns richer data: sentiment, confidence, reason, aspect, excerpt.
# Works in both modes:
#   brands=[]    → auto: model identifies all brands
#   brands=[...] → manual: model checks only those brands
# ---------------------------------------------------------------------------
def analyze_answer_for_brands(
    api_key: str,
    answer: str,
    question: str,
    brands: list[str],
    model: str,
) -> tuple[list[dict], dict]:
    if brands:
        brand_list = ", ".join(f'"{b}"' for b in brands)
        prompt = (
            f"Prüfe den folgenden Text ausschließlich auf Erwähnungen dieser Marken: {brand_list}.\n"
            "WICHTIG: Gib NUR Ergebnisse für Marken aus dieser Liste zurück — ignoriere alle anderen Marken, "
            "Produkte oder Anbieter, die im Text vorkommen.\n\n"
            f"Frage: {question}\n\n"
            f"Antwort:\n{answer}\n\n"
            "Für jede Marke aus der Liste, die tatsächlich im Text erwähnt wird, gib ein JSON-Objekt zurück:\n"
            '- "brand": Markenname exakt wie in der Liste oben angegeben\n'
            '- "sentiment": "positive", "neutral" oder "negative"\n'
            '- "confidence": "high", "medium" oder "low"\n'
            '- "reason": Ein Satz auf Deutsch, der das Sentiment begründet\n'
            '- "aspect": Hauptaspekt, z.B. "Qualität", "Preis", "Empfehlung", "Bekanntheit", "Funktionen"\n'
            '- "excerpt": Relevanter Satz aus dem Text (max 200 Zeichen)\n\n'
            "Nur valides JSON-Array, kein Markdown. Falls keine der gelisteten Marken erwähnt wird: []"
        )
    else:
        prompt = (
            "Extrahiere alle genannten Marken, Produkte und Anbieter aus dem folgenden Text "
            "und analysiere, wie das Modell über jede spricht.\n\n"
            f"Frage: {question}\n\n"
            f"Antwort:\n{answer}\n\n"
            "Antworte mit einem JSON-Array. Jedes Element:\n"
            '- "brand": Name der Marke\n'
            '- "sentiment": "positive", "neutral" oder "negative"\n'
            '- "confidence": "high", "medium" oder "low"\n'
            '- "reason": Ein Satz auf Deutsch, der das Sentiment begründet\n'
            '- "aspect": Hauptaspekt, z.B. "Qualität", "Preis", "Empfehlung", "Bekanntheit", "Funktionen"\n'
            '- "excerpt": Relevanter Satz aus dem Text (max 200 Zeichen)\n\n'
            "Nur valides JSON-Array, kein Markdown. Falls keine Marken: []"
        )

    text, err, usage = call_langdock(
        api_key,
        [{"role": "user", "content": prompt}],
        model=model,
        max_tokens=ANALYSIS_MAX_TOKENS,
    )
    if not text:
        log.warning("Brand analysis failed: %s", err)
        return [], usage
    return _parse_json_array(text), usage


# ---------------------------------------------------------------------------
# Analysis aggregation
# Flattens nested results into a pandas DataFrame.
# One row per (question, run, brand mention).
# ---------------------------------------------------------------------------
def build_analysis(results: list[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        for b in r.get("brands_found", []):
            rows.append({
                "question":   r["question"],
                "run":        r["run"],
                "brand":      b.get("brand", ""),
                "sentiment":  b.get("sentiment", "neutral"),
                "confidence": b.get("confidence", ""),
                "reason":     b.get("reason", ""),
                "aspect":     b.get("aspect", ""),
                "excerpt":    b.get("excerpt", b.get("context", "")),
                "mentions":   1,
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# CSV export
# Saves in the same column format as brand_monitor.py so the file is
# compatible with analyze_csv.py for downstream analysis.
# ---------------------------------------------------------------------------
def save_csv(results: list[dict], model: str) -> str:
    Path("results").mkdir(exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"results/brand_visibility_{ts}.csv"
    fieldnames = [
        "run_date", "model", "prompt", "repetition",
        "brands_found", "mention_count", "raw_response",
    ]
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            brands_str = "; ".join(b["brand"] for b in r.get("brands_found", []))
            writer.writerow({
                "run_date":      datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                "model":         model,
                "prompt":        r["question"],
                "repetition":    r["run"],
                "brands_found":  brands_str,
                "mention_count": len(r.get("brands_found", [])),
                "raw_response":  r.get("answer", "")[:500].replace("\n", " "),
            })
    return filename


# ===========================================================================
# UI — Step 1: Setup
# Collects all configuration and allows testing the connection before running.
# ===========================================================================
def render_step1():
    st.title("📊 LLM Brand Visibility")
    st.caption("Messe, wie LLMs Marken in beliebigen Themen wahrnehmen und empfehlen.")
    st.divider()

    col_form, col_info = st.columns([3, 2])

    with col_form:
        st.subheader("Konfiguration")

        api_key = st.text_input(
            "Langdock API Key",
            type="password",
            placeholder="sk--...",
            help="Wird nicht gespeichert. Nur für diese Session.",
        )

        model_option = st.selectbox(
            "Modell",
            options=_MODEL_OPTIONS,
            index=0,  # gpt-5-mini
            help="Verfügbare Modelle deines Langdock-Workspace. 'Benutzerdefiniert...' für andere.",
        )
        if model_option == "Benutzerdefiniert...":
            model = st.text_input(
                "Modell-Name (benutzerdefiniert)",
                placeholder="z.B. gpt-4o-search-preview",
            )
        else:
            model = _model_id_from_option(model_option)

        web_search = st.checkbox(
            "Aktualität betonen",
            value=False,
            help="Fügt dem Prompt eine Anweisung hinzu, aktuelles Wissen zu nutzen. Kein echter Web-Zugriff — die Langdock Completions API unterstützt keine nativen Search-Tools.",
        )

        topic = st.text_area(
            "Thema / Kontext",
            placeholder=(
                "z.B. 'High-Performance Sportwagen im DACH-Markt'\n"
                "oder 'CRM-Software für mittelständische Unternehmen'"
            ),
            height=100,
        )

        n_questions = st.slider(
            "Anzahl Fragen generieren", min_value=5, max_value=200, value=20, step=5
        )

        st.markdown("**Brand-Erkennung**")
        brand_mode = st.radio(
            "Modus",
            ["Manuell — Brands vorgeben", "Automatisch — aus Antworten extrahieren"],
            help="Manuell: schneller, kein Extra-Call. Automatisch: ein zusätzlicher API-Call pro Antwort.",
        )

        brands_input = ""
        if brand_mode.startswith("Manuell"):
            brands_input = st.text_input(
                "Brands (kommagetrennt)",
                placeholder="Nike, Adidas, ASICS",
            )

        # Connection test — runs a minimal API call to catch auth/model errors early
        st.markdown("---")
        if st.button("🔌 Verbindung testen", disabled=not (api_key and model)):
            with st.spinner("Teste Verbindung..."):
                ok, msg = test_connection(api_key, model)
            if ok:
                st.success(msg)
            else:
                st.error(f"Fehler: {msg}")
                st.info(
                    "Häufige Ursachen:\n"
                    "- API-Key falsch oder abgelaufen\n"
                    f"- Modell-Name '{model}' nicht verfügbar (Workspace-Einstellungen prüfen)\n"
                    f"- Falsche Region (aktuell: '{LANGDOCK_REGION}') — mit `LANGDOCK_REGION=us streamlit run app.py` wechseln"
                )

    with col_info:
        st.subheader("Hinweise")
        auto_preview = brand_mode.startswith("Auto")
        st.markdown(f"""
**Verbindung testen** bevor du startest — spart Zeit bei Fehlern.

**Thema:** Je spezifischer, desto relevanter die generierten Fragen.

**Modell:** Aktuell `{model or '—'}`. Welche Modelle verfügbar sind hängt von deinem Langdock-Workspace ab — bei Fehlern "Verbindung testen" nutzen.

**Region:** Aktuell `{LANGDOCK_REGION}`.
Ändern mit:
```
LANGDOCK_REGION=us streamlit run app.py
```

**Brand-Erkennung:**
- *Manuell*: Nur angegebene Brands werden getrackt.
- *Automatisch*: Modell erkennt alle Marken im Text. Ein Extra-Call pro Antwort.

**Kosten (ca.):**
- **{n_questions} Fragen** × R Runs = **{n_questions} × R Calls**
{"- Auto-Detect verdoppelt die Anzahl Calls" if auto_preview else ""}
- R (Runs pro Frage) konfigurierst du im nächsten Schritt
- Analyse lässt sich jederzeit stoppen — bisherige Ergebnisse bleiben erhalten
        """)

    st.divider()

    ready = bool(api_key and topic and model)
    if brand_mode.startswith("Manuell") and not brands_input.strip():
        ready = False
        st.warning("Bitte mindestens eine Brand eingeben, oder auf automatische Erkennung wechseln.")

    if st.button("Weiter: Fragen generieren →", type="primary", disabled=not ready):
        brands = (
            [b.strip() for b in brands_input.split(",") if b.strip()]
            if brand_mode.startswith("Manuell")
            else []
        )
        st.session_state.config = {
            "api_key":     api_key,
            "model":       model,
            "topic":       topic,
            "n_questions": n_questions,
            "brand_mode":  "manual" if brand_mode.startswith("Manuell") else "auto",
            "brands":      brands,
            "web_search":  web_search,
        }
        with st.spinner("Fragen werden generiert..."):
            questions, err = generate_questions(api_key, topic, n_questions, model)
        if questions:
            st.session_state.questions = questions
            st.session_state.step = 2
            st.rerun()
        else:
            st.error(f"Fragen konnten nicht generiert werden: {err}")
            st.info("Verbindung testen (Button oben) um die genaue Ursache zu sehen.")


# ===========================================================================
# UI — Step 2: Review questions
# One question per line in a text area — free to edit, delete, or add.
# ===========================================================================
def render_step2():
    st.title("📊 LLM Brand Visibility")
    st.progress(0.33, "Schritt 2 von 3 — Fragen prüfen")
    st.divider()

    st.subheader("Fragen prüfen und anpassen")
    st.caption(
        "Jede Zeile ist eine Frage. Bearbeiten, löschen oder neue ergänzen. "
        "Leere Zeilen werden ignoriert — die tatsächliche Anzahl siehst du unten."
    )

    questions_text = st.text_area(
        "Fragen (eine pro Zeile)",
        value="\n".join(st.session_state.questions),
        height=450,
    )

    n_lines = len([q for q in questions_text.splitlines() if q.strip()])
    orig    = len(st.session_state.questions)
    delta   = n_lines - orig
    delta_str = f" ({'+' if delta >= 0 else ''}{delta} gegenüber generiert)" if delta != 0 else ""
    st.caption(f"**{n_lines} Fragen**{delta_str} — nur diese werden analysiert.")

    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("← Zurück"):
            st.session_state.step = 1
            st.rerun()
    with col2:
        if st.button("Weiter: Runs konfigurieren →", type="primary"):
            cleaned = [q.strip() for q in questions_text.splitlines() if q.strip()]
            if not cleaned:
                st.error("Mindestens eine Frage erforderlich.")
            else:
                st.session_state.questions = cleaned
                st.session_state.step = 3
                st.rerun()


# ===========================================================================
# UI — Step 3: Configure and start runs
# Sets run count and delay, then kicks off the analysis loop.
# ===========================================================================
def render_step3():
    st.title("📊 LLM Brand Visibility")
    st.progress(0.66, "Schritt 3 von 3 — Runs starten")
    st.divider()

    cfg  = st.session_state.config
    n_q  = len(st.session_state.questions)
    auto = cfg["brand_mode"] == "auto"

    st.subheader("Runs konfigurieren")

    col_form, col_summary = st.columns([3, 2])

    with col_form:
        runs = st.slider("Wie oft soll jede Frage ausgeführt werden?", 1, 100, 3)

        parallel_calls = st.slider(
            "Parallele API-Calls",
            min_value=1, max_value=10, value=5,
            help="Bei 8000 Tokens/Call und 60k TPM-Limit passen ca. 7 Calls/min. 3–5 Workers ist ein guter Startpunkt.",
        )

        short_answer = st.toggle(
            "Kurzantwort-Modus",
            value=False,
            help="LLM gibt nur Marken + einen Begründungssatz zurück. Viel weniger Tokens, schneller, günstiger.",
        )

        max_tokens_val = st.slider(
            "Max. Tokens pro Antwort",
            min_value=500, max_value=16000, value=MAX_TOKENS, step=500,
            help=(
                "Bei Reasoning-Modellen (gpt-5-mini) fließen interne Denkschritte ins Budget ein — "
                "mindestens 8000 einplanen. gpt-5-mini-eu: 60.000 Tokens/Minute Limit."
            ),
        )

        # Live TPM estimate — assumes ~30s avg response time for reasoning models
        assumed_resp_s = 30
        calls_per_min  = parallel_calls * (60 / assumed_resp_s)
        tpm_estimate   = int(calls_per_min * max_tokens_val)
        tpm_pct        = tpm_estimate / 60000 * 100
        tpm_color      = "🟢" if tpm_pct < 70 else ("🟡" if tpm_pct < 100 else "🔴")
        st.caption(
            f"{tpm_color} Geschätzte Token-Last: **{tpm_estimate:,} Tokens/min** "
            f"({tpm_pct:.0f}% des 60k-Limits bei gpt-5-mini) — "
            f"Annahme: {assumed_resp_s}s Ø Antwortzeit, {parallel_calls} parallel"
        )

        delay = st.slider(
            "Pause zwischen API-Calls (Sekunden)",
            min_value=0.0, max_value=5.0, value=0.5, step=0.5,
            help="Pause nach jedem Call (sequenziell) bzw. nach jedem abgeschlossenen Batch (parallel).",
            disabled=parallel_calls > 1,
        )

        if auto:
            st.info(
                f"Auto-Detect aktiv: pro Antwort ein zusätzlicher API-Call. "
                f"Gesamt: ca. **{n_q * runs * 2} Calls** ({n_q} Fragen × {runs} Runs × 2)."
            )
        else:
            st.success(
                f"Brands vorgegeben: {', '.join(cfg['brands'])}. "
                f"Gesamt: ca. **{n_q * runs} Calls** ({n_q} Fragen × {runs} Runs)."
            )

        if cfg.get("web_search"):
            st.info("Aktualität betont — der Prompt enthält eine Anweisung, aktuelles Wissen zu priorisieren.")

        st.caption("Du kannst die Analyse jederzeit stoppen — bisherige Ergebnisse werden trotzdem angezeigt.")

    with col_summary:
        st.markdown("**Zusammenfassung**")
        st.metric("Fragen", n_q)
        st.metric("Runs pro Frage", runs)
        st.metric("Calls gesamt (ca.)", n_q * runs * (2 if auto else 1))
        st.metric("Modell", cfg["model"])

    st.divider()

    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("← Zurück"):
            st.session_state.step = 2
            st.rerun()
    with col2:
        if st.button("🚀 Analyse starten", type="primary"):
            st.session_state.config["runs"]           = runs
            st.session_state.config["delay"]          = delay
            st.session_state.config["parallel_calls"] = parallel_calls
            st.session_state.config["max_tokens"]     = max_tokens_val
            st.session_state.config["short_answer"]   = short_answer
            st.session_state.stop_requested           = False
            _run_analysis()


def _render_live_metrics(box, completed: int, total: int, elapsed: float, call_times: list[float]):
    """Renders a live 5-column timing dashboard into an st.empty() container."""
    rate_s    = completed / elapsed if elapsed > 0 else 0
    eta       = (total - completed) / rate_s if rate_s > 0 and completed < total else 0
    avg_t     = sum(call_times) / len(call_times) if call_times else 0
    min_t     = min(call_times) if call_times else 0
    max_t     = max(call_times) if call_times else 0
    with box.container():
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Fortschritt",       f"{completed} / {total}",
                  help="Abgeschlossene API-Calls / Calls insgesamt in dieser Phase.")
        c2.metric("Verstrichene Zeit", f"{elapsed:.0f}s",
                  help="Gesamtdauer seit Start dieser Phase.")
        c3.metric("Verbleibend (ETA)", f"~{eta:.0f}s" if completed < total else "✓ fertig",
                  help="Geschätzte Restdauer basierend auf der bisherigen Durchschnittsgeschwindigkeit.")
        c4.metric("Geschwindigkeit",   f"{rate_s * 60:.0f} / min",
                  help="Abgeschlossene Calls pro Minute (gleitend über diese Phase).")
        c5.metric("Ø Antwortzeit",     f"{avg_t:.1f}s", f"min {min_t:.1f}s  max {max_t:.1f}s",
                  help="Durchschnittliche Zeit pro API-Call. Delta zeigt schnellsten und langsamsten Call.")


def _run_analysis():
    cfg        = st.session_state.config
    questions  = st.session_state.questions
    api_key    = cfg["api_key"]
    model      = cfg["model"]
    topic      = cfg["topic"]
    runs       = cfg["runs"]
    delay      = cfg["delay"]
    parallel     = cfg.get("parallel_calls", 1)
    max_tokens   = cfg.get("max_tokens", MAX_TOKENS)
    web_search   = cfg.get("web_search", False)
    short_answer = cfg.get("short_answer", False)

    total       = len(questions) * runs
    raw_answers = []
    errors      = []
    p1_timings: list[dict] = []

    # ------------------------------------------------------------------
    # Phase 1 — Collect answers (parallel)
    # ------------------------------------------------------------------
    st.subheader("Phase 1 — Antworten sammeln")
    if st.button("⏹ Stoppen", type="secondary", key="stop_phase1"):
        st.session_state.stop_requested = True

    bar1        = st.progress(0.0)
    metrics_box = st.empty()
    status_line = st.empty()
    error_box   = st.empty()

    t_phase1   = time.time()
    call_times: list[float] = []

    all_tasks = [
        (i, question, run_num)
        for i, question in enumerate(questions)
        for run_num in range(runs)
    ]

    with ThreadPoolExecutor(max_workers=parallel) as executor:
        future_map = {
            executor.submit(ask_question, api_key, topic, q, model, web_search, max_tokens, short_answer): (i, run_num, q, time.time())
            for i, q, run_num in all_tasks
        }

        completed = 0
        for future in as_completed(future_map):
            i, run_num, question, t_sub = future_map[future]
            completed += 1

            if st.session_state.get("stop_requested", False):
                status_line.warning(
                    f"⏹ Gestoppt nach {len(raw_answers)} von {total} Antworten. "
                    "Analyse wird mit bisherigen Antworten fortgesetzt."
                )
                break

            answer, err, usage = future.result()
            call_elapsed = time.time() - t_sub
            call_times.append(call_elapsed)
            p1_timings.append({
                "question":  question,
                "run":       run_num + 1,
                "elapsed_s": round(call_elapsed, 3),
                "ok":        answer is not None,
            })

            bar1.progress(completed / total)
            _render_live_metrics(metrics_box, completed, total, time.time() - t_phase1, call_times)

            if not answer:
                errors.append(f"Frage {i+1}, Run {run_num+1}: {err}")
                error_box.error(f"⚠️ Frage {i+1}, Run {run_num+1} fehlgeschlagen: {err}")
                log.warning("No answer — Frage %d, Run %d: %s", i+1, run_num+1, err)
            else:
                error_box.empty()
                raw_answers.append({
                    "question":        question,
                    "run":             run_num + 1,
                    "answer":          answer,
                    "tokens_in":       usage.get("prompt_tokens", 0),
                    "tokens_out":      usage.get("completion_tokens", 0),
                })
                log.info("Phase1 OK — Frage %d Run %d | len=%d | tok_in=%d tok_out=%d | %.2fs",
                         i+1, run_num+1, len(answer),
                         usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0), call_elapsed)

            q_short = question[:70] + ("…" if len(question) > 70 else "")
            status_line.caption(f"Zuletzt: Frage {i+1} · Run {run_num+1} · {call_elapsed:.1f}s · {q_short}")

            if parallel == 1:
                time.sleep(delay)

    phase1_elapsed = time.time() - t_phase1
    bar1.progress(1.0)
    _render_live_metrics(metrics_box, completed, total, phase1_elapsed, call_times)
    status_line.caption(
        f"Phase 1 abgeschlossen — {len(raw_answers)}/{total} Antworten in {phase1_elapsed:.1f}s"
    )

    st.session_state.raw_answers    = raw_answers
    st.session_state.stop_requested = False
    st.session_state.timing = {
        "phase1_s": round(phase1_elapsed, 2),
        "p1_calls": p1_timings,
        "phase2_s": 0.0,
        "p2_calls": [],
    }

    # ------------------------------------------------------------------
    # Phase 2 — Brand + sentiment analysis
    # ------------------------------------------------------------------
    _run_brand_analysis(raw_answers, errors)


def _run_brand_analysis(raw_answers: list[dict], prior_errors: list | None = None):
    """
    Phase 2: brand extraction + sentiment, grouped by unique question.
    Each question's runs are split into chunks of BRAND_CHUNK_SIZE so the model
    receives a manageable input rather than all 15 runs concatenated.
    Brand results from each chunk are merged (deduplicated by name) per question.
    """
    BRAND_CHUNK_SIZE = 3

    cfg      = st.session_state.config
    api_key  = cfg["api_key"]
    model    = cfg["model"]
    brands   = cfg.get("brands", [])
    parallel = cfg.get("parallel_calls", 1)
    errors   = list(prior_errors or [])

    # Group raw answers by question, preserving insertion order
    by_question: dict[str, list[dict]] = {}
    for raw in raw_answers:
        by_question.setdefault(raw["question"], []).append(raw)

    unique_questions = list(by_question.keys())

    def _combine(runs: list[dict]) -> str:
        if len(runs) == 1:
            return runs[0]["answer"]
        return "\n\n---\n\n".join(
            f"Antwort {i+1}:\n{r['answer']}" for i, r in enumerate(runs)
        )

    # Map returned brand names to canonical user-specified casing (case-insensitive).
    # Also filters out unlisted brands in manual mode as a second safety net.
    brand_lookup: dict[str, str] = {b.strip().lower(): b for b in brands} if brands else {}

    def _normalize_brands(brand_list: list[dict]) -> list[dict]:
        out = []
        for b in brand_list:
            raw_name = b.get("brand", "").strip()
            if not raw_name:
                continue
            if brand_lookup:
                canonical = brand_lookup.get(raw_name.lower())
                if canonical is None:
                    continue  # not in user's list — drop it
                b = {**b, "brand": canonical}
            out.append(b)
        return out

    def _merge_chunks(brand_lists: list[list[dict]]) -> list[dict]:
        seen: dict[str, dict] = {}
        for chunk in brand_lists:
            for b in _normalize_brands(chunk):
                key = b.get("brand", "").strip().lower()
                if key and key not in seen:
                    seen[key] = b
        return list(seen.values())

    # Build one task per chunk (not one per question)
    task_list = []
    for q_idx, q in enumerate(unique_questions):
        runs = by_question[q]
        for chunk_start in range(0, len(runs), BRAND_CHUNK_SIZE):
            chunk = runs[chunk_start : chunk_start + BRAND_CHUNK_SIZE]
            task_list.append((q_idx, q, chunk, _combine(chunk)))

    total = len(task_list)
    n_questions = len(unique_questions)

    st.subheader("Phase 2 — Brand & Sentiment Analyse")
    st.caption(
        f"{n_questions} Fragen analysieren · {total} Analyse-Calls "
        f"(je {BRAND_CHUNK_SIZE} Antworten pro Call)"
    )
    bar2        = st.progress(0.0)
    metrics_box = st.empty()
    status_line = st.empty()

    t_phase2    = time.time()
    call_times2: list[float] = []
    p2_timings:  list[dict]  = []
    # Accumulate chunk results and token usage per question
    chunk_results: dict[str, list[list[dict]]] = {q: [] for q in unique_questions}
    chunk_tokens:  dict[str, int]              = {q: 0  for q in unique_questions}
    completed   = 0

    with ThreadPoolExecutor(max_workers=parallel) as executor:
        future_map = {
            executor.submit(analyze_answer_for_brands, api_key, combined, q, brands, model): (q_idx, q, chunk, time.time())
            for q_idx, q, chunk, combined in task_list
        }

        for future in as_completed(future_map):
            q_idx, question, chunk, t_sub = future_map[future]
            completed += 1

            brands_found, usage = future.result()
            call_elapsed = time.time() - t_sub
            call_times2.append(call_elapsed)
            p2_timings.append({"question": question, "elapsed_s": round(call_elapsed, 3)})
            chunk_results[question].append(brands_found)
            chunk_tokens[question] += usage.get("completion_tokens", 0)

            bar2.progress(completed / total)
            _render_live_metrics(metrics_box, completed, total, time.time() - t_phase2, call_times2)

            q_short = question[:70] + ("…" if len(question) > 70 else "")
            status_line.caption(
                f"Zuletzt: Q{q_idx+1} · {len(brands_found)} Brand(s) in diesem Chunk · {call_elapsed:.1f}s · {q_short}"
            )
            log.info(
                "Phase2 chunk — Q %d/%d | chunk_brands: %s | %.2fs",
                q_idx+1, n_questions, [b["brand"] for b in brands_found], call_elapsed,
            )

    # Merge chunks per question and reconstruct results in original order
    results = []
    for question in unique_questions:
        merged = _merge_chunks(chunk_results[question])
        log.info(
            "Phase2 OK — Q %d/%d | runs=%d | brands: %s",
            unique_questions.index(question) + 1, n_questions,
            len(by_question[question]), [b["brand"] for b in merged],
        )
        analysis_tokens = chunk_tokens[question]
        for raw in by_question[question]:
            results.append({
                "question":        raw["question"],
                "run":             raw["run"],
                "answer":          raw["answer"],
                "tokens_in":       raw.get("tokens_in", 0),
                "tokens_out":      raw.get("tokens_out", 0),
                "tokens_analysis": analysis_tokens,
                "brands_found":    merged,
            })

    phase2_elapsed = time.time() - t_phase2
    bar2.progress(1.0)
    _render_live_metrics(metrics_box, completed, total, phase2_elapsed, call_times2)
    status_line.caption(
        f"Phase 2 abgeschlossen — {n_questions} Fragen ({total} Calls) in {phase2_elapsed:.1f}s"
        + (f", {len(errors)} Fehler" if errors else "")
    )

    # Persist Phase 2 timing into the session state dict written by Phase 1
    timing = st.session_state.get("timing", {})
    timing["phase2_s"] = round(phase2_elapsed, 2)
    timing["p2_calls"] = p2_timings
    st.session_state.timing = timing

    if errors:
        with st.expander(f"⚠️ {len(errors)} Fehler"):
            for e in errors:
                st.markdown(f"- {e}")

    st.session_state.results = results
    st.session_state.step    = 4
    time.sleep(1)
    st.rerun()


# ===========================================================================
# UI — Step 4: Results
# KPI metrics, charts, context details, raw data table, export options.
# ===========================================================================
def render_step4():
    st.title("📊 LLM Brand Visibility — Ergebnisse")
    st.divider()

    results = st.session_state.results
    cfg     = st.session_state.config

    # --- Configuration summary (from Step 1 + Step 3) ----------------------
    with st.expander("Analyse-Konfiguration", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f"**Thema**  \n{cfg.get('topic', '—')}")
        c2.markdown(f"**Modell**  \n`{cfg.get('model', '—')}`")
        runs_val     = cfg.get("runs", "—")
        parallel_val = cfg.get("parallel_calls", 1)
        n_q          = len(st.session_state.get("questions", []))
        c3.markdown(f"**Fragen / Runs**  \n{n_q} Fragen × {runs_val} Runs")
        c4.markdown(
            f"**Parallel / Delay / Tokens**  \n"
            f"{parallel_val} parallel | {cfg.get('delay', 0.5)}s Pause | max {cfg.get('max_tokens', MAX_TOKENS)} Tokens"
        )

        brand_mode = cfg.get("brand_mode", "auto")
        if brand_mode == "manual":
            brands_list = cfg.get("brands", [])
            st.markdown(
                f"**Brands (manuell):** " +
                ", ".join(f"`{b}`" for b in brands_list) if brands_list else "—"
            )
        else:
            st.markdown("**Brand-Erkennung:** Automatisch (aus Antworten extrahiert)")

        if cfg.get("web_search"):
            st.markdown("**Aktualität betont:** Ja")

    st.divider()

    if not results:
        st.warning("Keine auswertbaren Ergebnisse. Alle API-Calls sind fehlgeschlagen.")
        if st.button("← Zurück zur Konfiguration"):
            st.session_state.step = 1
            st.rerun()
        return

    df = build_analysis(results)

    # --- KPI row: one metric per brand --------------------------------------
    if not df.empty:
        brands     = sorted(df["brand"].unique().tolist())
        total_runs = len(results)
        cols       = st.columns(max(len(brands), 1))

        for col, brand in zip(cols, brands):
            mentions    = int(df[df["brand"] == brand]["mentions"].sum())
            sov         = round(mentions / total_runs * 100, 1)
            sent_counts = df[df["brand"] == brand].groupby("sentiment")["mentions"].sum()
            top_sent    = sent_counts.idxmax() if not sent_counts.empty else "n/a"
            col.metric(brand, f"{sov}% Share of Voice", f"Sentiment: {top_sent}")

    st.divider()

    # Re-analyze button — reruns Phase 2 on stored raw answers without re-querying
    raw_answers = st.session_state.get("raw_answers", [])
    if raw_answers:
        with st.expander("🔄 Sentiment neu analysieren (ohne neue API-Calls für Antworten)"):
            st.caption(
                f"{len(raw_answers)} Antworten gespeichert. "
                "Brand-Extraktion und Sentiment werden neu berechnet — Antworten werden nicht erneut abgefragt."
            )
            if st.button("Analyse neu starten", type="primary", key="reanalyze"):
                _run_brand_analysis(raw_answers)

    st.divider()

    tab_sov, tab_sent, tab_answers, tab_raw, tab_timing = st.tabs(
        ["Share of Voice", "Sentiment", "Alle Antworten", "Rohdaten", "Laufzeit"]
    )

    # --- Tab 1: Share of Voice bar chart ------------------------------------
    with tab_sov:
        if df.empty:
            st.info(
                "Keine Brands erkannt.\n\n"
                "Bei manuellem Modus: Prüfen ob die Brand-Namen genau so im Text vorkommen "
                "(z.B. 'Nike' vs 'NIKE'). Die Rohdaten-Tab zeigt die vollständigen Antworten."
            )
        else:
            st.subheader("Wie oft wird jede Brand genannt?")
            sov_df = (
                df.groupby("brand")["mentions"]
                .sum()
                .reset_index()
                .assign(anteil=lambda x: x["mentions"] / x["mentions"].sum() * 100)
                .sort_values("anteil", ascending=True)
            )
            fig = px.bar(
                sov_df,
                x="anteil",
                y="brand",
                orientation="h",
                text=sov_df["anteil"].map(lambda x: f"{x:.1f}%"),
                labels={"anteil": "Share of Voice (%)", "brand": ""},
                color="brand",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(showlegend=False, xaxis_range=[0, 110])
            st.plotly_chart(fig, use_container_width=True)

            # Heatmap: brand × question, using Q1/Q2/… labels
            st.subheader("Sentiment-Heatmap: Brand × Frage")
            st.caption("Grün = positiv (+1), Grau = neutral (0), Rot = negativ (−1). Weiß = nicht erwähnt.")
            import plotly.graph_objects as go

            sentiment_score = {"positive": 1, "neutral": 0, "negative": -1}
            df_heat = df.copy()
            df_heat["score"] = df_heat["sentiment"].map(sentiment_score).fillna(0)

            # Map full question text → Q1, Q2, … (order by first appearance in results)
            q_order = list(dict.fromkeys(r["question"] for r in results))
            q_label = {q: f"Q{i+1}" for i, q in enumerate(q_order)}
            df_heat["q_label"] = df_heat["question"].map(q_label)

            pivot = (
                df_heat.groupby(["brand", "q_label"])["score"]
                .mean()
                .reset_index()
                .pivot(index="brand", columns="q_label", values="score")
            )
            # Sort columns numerically (Q1, Q2, … Q10, Q11, …)
            pivot = pivot.reindex(
                columns=sorted(pivot.columns, key=lambda c: int(c[1:]))
            )

            fig_heat = go.Figure(data=go.Heatmap(
                z=pivot.values,
                x=list(pivot.columns),
                y=list(pivot.index),
                colorscale=[
                    [0.0, "#e74c3c"],
                    [0.5, "#ecf0f1"],
                    [1.0, "#2ecc71"],
                ],
                zmin=-1, zmax=1,
                text=[[
                    f"{v:+.2f}" if not pd.isna(v) else "—"
                    for v in row
                ] for row in pivot.values],
                texttemplate="%{text}",
                hoverongaps=False,
                customdata=[[
                    q_order[int(col[1:]) - 1] if col[1:].isdigit() else col
                    for col in pivot.columns
                ] for _ in pivot.index],
                hovertemplate="<b>%{y}</b> × %{x}<br>Score: %{z:.2f}<br>%{customdata}<extra></extra>",
            ))
            fig_heat.update_layout(
                xaxis={"tickfont": {"size": 11}},
                margin={"l": 120, "b": 40},
                height=max(200, 55 * len(pivot.index) + 80),
            )
            st.plotly_chart(fig_heat, use_container_width=True)

            # Question reference table
            with st.expander("Fragenverzeichnis (Q1 … Q" + str(len(q_order)) + ")"):
                for q, label in q_label.items():
                    st.markdown(f"**{label}** — {q}")

    # --- Tab 2: Sentiment breakdown + brand mention summary -----------------
    with tab_sent:
        if df.empty:
            st.info("Keine Brands erkannt.")
        else:
            # Sentiment % breakdown per brand as metrics
            st.subheader("Sentiment-Verteilung je Brand")
            sentiment_cols = st.columns(len(sorted(df["brand"].unique())))
            for col, brand in zip(sentiment_cols, sorted(df["brand"].unique())):
                brand_df     = df[df["brand"] == brand]
                total        = int(brand_df["mentions"].sum())
                counts       = brand_df.groupby("sentiment")["mentions"].sum()
                pos_pct  = round(counts.get("positive", 0) / total * 100)
                neu_pct  = round(counts.get("neutral",  0) / total * 100)
                neg_pct  = round(counts.get("negative", 0) / total * 100)
                with col:
                    st.markdown(f"**{brand}**")
                    st.markdown(
                        f"🟢 {pos_pct}% positiv  \n"
                        f"⚪ {neu_pct}% neutral  \n"
                        f"🔴 {neg_pct}% negativ  \n"
                        f"*{total} Nennungen gesamt*"
                    )

            st.divider()
            st.subheader("Sentiment-Vergleich (gestapelt)")
            sent_df = df.groupby(["brand", "sentiment"])["mentions"].sum().reset_index()
            # Add % column for hover
            sent_df["pct"] = sent_df.groupby("brand")["mentions"].transform(
                lambda x: (x / x.sum() * 100).round(1)
            )
            fig = px.bar(
                sent_df,
                x="brand",
                y="mentions",
                color="sentiment",
                barmode="stack",
                text=sent_df["pct"].map(lambda x: f"{x:.0f}%"),
                color_discrete_map={
                    "positive": "#2ecc71",
                    "neutral":  "#95a5a6",
                    "negative": "#e74c3c",
                },
                labels={"mentions": "Anzahl Nennungen", "brand": ""},
            )
            fig.update_traces(textposition="inside", textfont_size=12)
            st.plotly_chart(fig, use_container_width=True)

            st.divider()
            st.subheader("Was sagen die Modelle über jede Brand?")
            st.caption("Alle Sätze, in denen die Brand erwähnt wurde, gesammelt pro Sentiment.")
            for brand in sorted(df["brand"].unique()):
                brand_df = df[df["brand"] == brand]
                total    = int(brand_df["mentions"].sum())
                with st.expander(f"{brand} — {total} Nennungen"):
                    for sentiment, label, icon in [
                        ("positive", "Positiv", "🟢"),
                        ("neutral",  "Neutral", "⚪"),
                        ("negative", "Negativ", "🔴"),
                    ]:
                        rows_sent = brand_df[brand_df["sentiment"] == sentiment]
                        if rows_sent.empty:
                            continue
                        st.markdown(f"**{icon} {label}**")
                        for _, row in rows_sent.head(15).iterrows():
                            reason  = row.get("reason", "")
                            excerpt = row.get("excerpt", "")
                            aspect  = row.get("aspect", "")
                            conf    = row.get("confidence", "")
                            parts = []
                            if reason:
                                parts.append(reason)
                            if excerpt and excerpt != reason:
                                parts.append(f'*„{excerpt}“*')
                            detail = "  \n".join(parts) if parts else "—"
                            badge = f"`{aspect}`  " if aspect else ""
                            conf_str = f"  <small>({conf})</small>" if conf else ""
                            st.markdown(f"{badge}{detail}{conf_str}")

    # --- Tab 3: All answers in readable format -------------------------------
    with tab_answers:
        # Build question → Q-number mapping (insertion order = question order)
        q_order_ans = list(dict.fromkeys(r["question"] for r in results))
        q_num_ans   = {q: i + 1 for i, q in enumerate(q_order_ans)}
        n_runs_seen = max((r["run"] for r in results), default=1)

        st.subheader("Alle Antworten")
        st.caption(
            f"{len(q_order_ans)} Fragen × {n_runs_seen} Run(s) = {len(results)} Antworten"
        )

        filter_q = st.selectbox(
            "Frage filtern",
            options=["Alle Fragen"] + [f"Q{q_num_ans[q]} — {q[:60]}" for q in q_order_ans],
            key="answers_filter",
        )
        # Resolve filter back to full question text
        filter_q_text = (
            None if filter_q == "Alle Fragen"
            else q_order_ans[int(filter_q.split(" — ")[0][1:]) - 1]
        )

        for r in results:
            if filter_q_text and r["question"] != filter_q_text:
                continue
            answer       = r.get("answer") or ""
            brands_found = r.get("brands_found", [])
            qn           = q_num_ans[r["question"]]
            label        = f"Q{qn} · Run {r['run']} · {r['question'][:65]}{'…' if len(r['question']) > 65 else ''}"
            with st.expander(label):
                st.markdown(f"**Vollständige Frage:** {r['question']}")
                st.divider()
                if answer:
                    st.markdown(answer)
                else:
                    st.error("Keine Antwort erhalten (API-Fehler).")
                if brands_found:
                    st.divider()
                    st.markdown("**Erkannte Brands:**")
                    brand_cols = st.columns(min(len(brands_found), 4))
                    for bc, b in zip(brand_cols, brands_found):
                        sentiment_icon = {"positive": "🟢", "neutral": "⚪", "negative": "🔴"}.get(
                            b.get("sentiment", "neutral"), "⚪"
                        )
                        bc.markdown(f"{sentiment_icon} **{b['brand']}**")

    # --- Tab 4: Raw data table -----------------------------------------------
    with tab_raw:
        st.subheader("Rohdaten")
        raw_rows = []
        for r in results:
            answer = r.get("answer") or ""
            raw_rows.append({
                "Frage":            r["question"][:80] + ("..." if len(r["question"]) > 80 else ""),
                "Run":              r["run"],
                "Antwort":          (answer[:200] + "...") if len(answer) > 200 else answer,
                "Brands":           ", ".join(b["brand"] for b in r.get("brands_found", [])),
                "Tok. Input":       r.get("tokens_in", 0),
                "Tok. Antwort":     r.get("tokens_out", 0),
                "Tok. Analyse":     r.get("tokens_analysis", 0),
                "API OK":           "✅" if answer else "❌",
            })
        st.dataframe(
            pd.DataFrame(raw_rows),
            column_config={
                "Tok. Input":   st.column_config.NumberColumn(
                    "Tok. Input",
                    help="Prompt-Tokens: Größe des Eingabe-Textes (Systemprompt + Frage), den das Modell erhält.",
                ),
                "Tok. Antwort": st.column_config.NumberColumn(
                    "Tok. Antwort",
                    help="Completion-Tokens: Tokens der generierten Antwort in Phase 1. Bei Reasoning-Modellen schließt das interne Denkschritte ein.",
                ),
                "Tok. Analyse": st.column_config.NumberColumn(
                    "Tok. Analyse",
                    help="Completion-Tokens der Markenanalyse (Phase 2), summiert über alle Chunks dieser Frage.",
                ),
            },
            use_container_width=True,
        )

        # Full raw answers for the first 3 runs — useful for debugging brand matching
        with st.expander("🔍 Debug: Erste 3 vollständige Antworten"):
            for r in results[:3]:
                st.markdown(f"**Frage:** {r['question']}")
                st.markdown("**Vollständige Antwort:**")
                st.text(r.get("answer") or "— LEER —")
                st.markdown(f"**Erkannte Brands:** {r.get('brands_found', [])}")
                st.divider()

    # --- Tab 5: Laufzeit -------------------------------------------------------
    with tab_timing:
        timing = st.session_state.get("timing", {})
        p1_s     = timing.get("phase1_s", 0.0)
        p2_s     = timing.get("phase2_s", 0.0)
        p1_calls = timing.get("p1_calls", [])
        p2_calls = timing.get("p2_calls", [])

        if not p1_calls and not p2_calls:
            st.info("Keine Laufzeitdaten verfügbar (Analyse wurde vor diesem Update gestartet).")
        else:
            # --- Summary metrics ---
            all_elapsed = [x["elapsed_s"] for x in p1_calls + p2_calls]
            avg_all = sum(all_elapsed) / len(all_elapsed) if all_elapsed else 0
            min_all = min(all_elapsed) if all_elapsed else 0
            max_all = max(all_elapsed) if all_elapsed else 0

            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Phase 1 — Antworten",  f"{p1_s:.1f}s",  f"{len(p1_calls)} Calls",
                       help="Gesamtdauer, in der das Modell alle Fragen beantwortet hat (parallele Calls).")
            mc2.metric("Phase 2 — Analyse",    f"{p2_s:.1f}s",  f"{len(p2_calls)} Calls",
                       help="Gesamtdauer der Marken- und Sentiment-Analyse. Je mehr Runs und Fragen, desto mehr Chunks.")
            mc3.metric("Gesamt",               f"{p1_s + p2_s:.1f}s",
                       help="Phase 1 + Phase 2 zusammen. Wartezeiten bei Rate-Limits (429) sind enthalten.")
            mc4.metric("Ø Antwortzeit",        f"{avg_all:.1f}s", f"min {min_all:.1f}s  max {max_all:.1f}s",
                       help="Durchschnitt über alle Calls beider Phasen. Delta: schnellster und langsamster Call.")

            st.divider()

            # Build question label map (Q1, Q2, …) from result order
            q_order_t = list(dict.fromkeys(r["question"] for r in results))
            q_label_t = {q: f"Q{i+1}" for i, q in enumerate(q_order_t)}

            col_p1, col_p2 = st.columns(2)

            # Phase 1 — response time histogram
            with col_p1:
                st.subheader("Phase 1 — Antwortzeit-Verteilung")
                if p1_calls:
                    df_p1 = pd.DataFrame(p1_calls)
                    fig_hist = px.histogram(
                        df_p1,
                        x="elapsed_s",
                        nbins=min(30, max(5, len(p1_calls) // 2)),
                        labels={"elapsed_s": "Antwortzeit (s)", "count": "Anzahl Calls"},
                        color_discrete_sequence=["#3498db"],
                    )
                    fig_hist.update_layout(bargap=0.05, showlegend=False)
                    st.plotly_chart(fig_hist, use_container_width=True)
                    ok_pct = round(sum(1 for x in p1_calls if x["ok"]) / len(p1_calls) * 100, 1)
                    st.caption(f"{ok_pct}% der Calls erfolgreich")

            # Phase 2 — per-question bar chart
            with col_p2:
                st.subheader("Phase 2 — Analysezeit je Frage")
                if p2_calls:
                    df_p2 = pd.DataFrame(p2_calls)
                    df_p2["label"] = df_p2["question"].map(q_label_t)
                    df_p2 = df_p2.sort_values("elapsed_s", ascending=False)
                    fig_p2 = px.bar(
                        df_p2,
                        x="elapsed_s",
                        y="label",
                        orientation="h",
                        labels={"elapsed_s": "Zeit (s)", "label": ""},
                        color="elapsed_s",
                        color_continuous_scale="Blues",
                    )
                    fig_p2.update_layout(showlegend=False, coloraxis_showscale=False,
                                        height=max(200, 28 * len(df_p2) + 60))
                    st.plotly_chart(fig_p2, use_container_width=True)

            # Phase 1 detail table
            if p1_calls:
                with st.expander("Phase 1 — Detailtabelle"):
                    df_detail = pd.DataFrame(p1_calls)
                    df_detail["Frage"] = df_detail["question"].map(
                        lambda q: f"{q_label_t.get(q, '?')} — {q[:60]}{'…' if len(q) > 60 else ''}"
                    )
                    df_detail = df_detail.rename(columns={"run": "Run", "elapsed_s": "Zeit (s)", "ok": "Erfolg"})
                    df_detail["Erfolg"] = df_detail["Erfolg"].map({True: "✅", False: "❌"})
                    st.dataframe(
                        df_detail[["Frage", "Run", "Zeit (s)", "Erfolg"]].sort_values("Zeit (s)", ascending=False),
                        use_container_width=True,
                    )

    st.divider()

    # --- Export options -----------------------------------------------------
    st.subheader("Export")
    col1, col2, col3 = st.columns(3)

    with col1:
        if not df.empty:
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Analyse als CSV",
                csv_bytes,
                "brand_visibility_analysis.csv",
                "text/csv",
            )

    with col2:
        raw_json = json.dumps(results, ensure_ascii=False, indent=2)
        st.download_button(
            "📥 Rohdaten als JSON",
            raw_json.encode("utf-8"),
            "brand_visibility_raw.json",
            "application/json",
        )

    with col3:
        # Saves in brand_monitor.py format — compatible with analyze_csv.py
        if st.button("💾 Lokal speichern (results/)"):
            filepath = save_csv(results, cfg.get("model", ""))
            st.success(f"Gespeichert: `{filepath}`")

    st.divider()
    if st.button("🔄 Neue Analyse starten"):
        for key in ["step", "questions", "results", "config"]:
            del st.session_state[key]
        st.rerun()


# ===========================================================================
# Router — renders the correct step based on session state
# ===========================================================================
step = st.session_state.step

if step == 1:
    render_step1()
elif step == 2:
    render_step2()
elif step == 3:
    render_step3()
elif step == 4:
    render_step4()
