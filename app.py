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
        "phase1_errors":  [],   # failed calls collected during phase 1
        "results":        [],   # phase 2 output — with brands + sentiment
        "unlisted_brands": [],  # manual mode: brands the model found that aren't on the user's list
        "analysis_summary": "",  # phase 2 executive summary (from the single Opus call)
        "stop_requested": False,
        "timing":         {},   # phase wall times + per-call elapsed data
        "lang":           "de", # UI + model response language: "de" or "en"
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

def reset_process():
    """Clears everything but the language preference and restarts at Step 1."""
    for key in ["step", "config", "questions", "raw_answers", "phase1_errors", "results", "unlisted_brands", "analysis_summary", "timing"]:
        if key in st.session_state:
            del st.session_state[key]
    init_state()
    st.rerun()

# ---------------------------------------------------------------------------
# i18n
# tr(de, en) returns whichever string matches the language currently selected
# in the top-of-screen switcher. Kept inline (no key dictionary) so each call
# site stays self-documenting.
# An explicit `lang` override is required from worker threads (ThreadPoolExecutor) —
# st.session_state is not accessible off the main script thread and triggers
# Streamlit's "missing ScriptRunContext" warning.
# ---------------------------------------------------------------------------
def tr(de: str, en: str, lang: str | None = None) -> str:
    if lang is None:
        lang = st.session_state.get("lang", "de")
    return de if lang == "de" else en

def render_language_switch():
    col_spacer, col_lang = st.columns([6, 1])
    with col_lang:
        current = st.session_state.get("lang", "de")
        choice = st.selectbox(
            "🌐",
            ["Deutsch", "English"],
            index=0 if current == "de" else 1,
            key="lang_selector",
            label_visibility="collapsed",
        )
        st.session_state.lang = "de" if choice == "Deutsch" else "en"

render_language_switch()

# ---------------------------------------------------------------------------
# Langdock API configuration
# ---------------------------------------------------------------------------
LANGDOCK_REGION     = os.environ.get("LANGDOCK_REGION", "eu")
LANGDOCK_URL        = f"https://api.langdock.com/openai/{LANGDOCK_REGION}/v1/chat/completions"
ANTHROPIC_URL       = f"https://api.langdock.com/anthropic/{LANGDOCK_REGION}/v1/messages"
GOOGLE_URL_TEMPLATE = "https://api.langdock.com/google/" + LANGDOCK_REGION + "/v1beta/models/{model}:generateContent"
# Agent Completions API — model-agnostic, no region in the URL. Only endpoint that
# supports real built-in web search (capabilities.webSearch) for any model string.
AGENT_URL           = "https://api.langdock.com/agent/v1/chat/completions"
# The Agent API has its own model catalog with its own IDs (e.g. "claude-opus-4-7@default",
# "gpt-5-mini-eu") that don't match the provider-passthrough model IDs above — must be
# fetched live rather than hardcoded, see list_agent_models() below.
AGENT_MODELS_URL    = "https://api.langdock.com/agent/v1/models"
REQUEST_TIMEOUT     = 180   # gpt-5-mini (reasoning) regularly takes 30-56s but spikes under parallel load; 120s still caused avoidable timeouts
AGENT_STREAM_TIMEOUT = 240  # web-search agent calls: non-streaming requests hit a hard 524 at 100s server-side, so we always stream; this is just the client-side read timeout
MAX_TOKENS          = 8000  # answer calls — default; reasoning models need this much just to clear their thinking budget (see Step 3 slider help). User can raise up to 16000 in the UI
QUESTION_MAX_TOKENS = 16000 # question generation: reasoning model uses token budget for internal thinking too

# Phase 2 (analysis) always runs on one strong model in a single call — regardless of
# which model collected the answers — so the brand/sentiment judgement is consistent
# across runs and doesn't inherit weaker models' extraction errors.
ANALYSIS_MODEL              = "claude-opus-4-8"
DATASET_ANALYSIS_MAX_TOKENS = 16000  # per-batch analysis output cap; batching keeps the array from being truncated
ANALYSIS_ANSWER_CHARS       = 2000   # per-answer truncation in the analysis prompt — bounds total input tokens

# Sampling temperature.
# Collection uses a non-zero temperature so that repeating the SAME question N times
# actually produces varied answers — otherwise "runs per question" adds no statistical
# information (a deterministic model returns the identical answer N times). Question
# generation runs a bit hotter for diversity. Analysis runs at 0 so brand/sentiment
# judgements are as reproducible as possible across re-analysis.
COLLECTION_TEMPERATURE = 0.7
QUESTION_TEMPERATURE   = 0.8
ANALYSIS_TEMPERATURE   = 0.0

# The whole-dataset analysis is split into batches. A single call capped at
# DATASET_ANALYSIS_MAX_TOKENS silently truncates its JSON array on large runs
# (finish_reason=max_tokens → unparseable → total loss of ALL analysis). Batching
# bounds each call's output and makes one bad batch cost only that batch. A batch is
# limited by answer count AND estimated input tokens, whichever is hit first.
ANALYSIS_BATCH_MAX_ANSWERS      = 40
ANALYSIS_BATCH_MAX_INPUT_TOKENS = 40000

# Long answers are truncated head+tail (not head-only) before analysis, so brands
# listed near the END of an answer — items 8–15 of a "top tools" list — aren't dropped.
ANALYSIS_ANSWER_HEAD_FRAC = 0.7

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
    "Google": [
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
    ],
    "Meta (via Langdock)": [
        "langdock-llama-3.3-70b-2",
    ],
}

_MODEL_OPTIONS = (
    [f"{provider} — {m}" for provider, models in AVAILABLE_MODELS.items() for m in models]
    + [tr("Benutzerdefiniert...", "Custom...")]
)

def _model_id_from_option(option: str) -> str:
    return option.split(" — ", 1)[1] if " — " in option else option

def _is_anthropic_model(model: str) -> bool:
    return model.startswith("claude-")

def _is_google_model(model: str) -> bool:
    return model.startswith("gemini-")


@st.cache_data(ttl=300, show_spinner=False)
def list_agent_models(api_key: str) -> tuple[list[str], str | None]:
    """
    Fetches the live model catalog for the Agent Completions API (GET /agent/v1/models).
    Cached per api_key for 5 minutes — this endpoint is only relevant when web search
    is enabled, and its model IDs differ from the provider-passthrough ones above.
    """
    try:
        r = requests.get(
            AGENT_MODELS_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=15,
        )
        r.raise_for_status()
        ids = sorted({m["id"] for m in r.json().get("data", []) if m.get("id")})
        return ids, None
    except requests.exceptions.RequestException as e:
        return [], str(e)


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
    lang: str = "de",
    temperature: float | None = None,
) -> tuple[str | None, str | None, dict]:
    """
    Returns (response_text, error_message, usage).
    On success: (text, None, {"prompt_tokens": N, "completion_tokens": N, "total_tokens": N})
    On failure: (None, human-readable error string, {})

    `lang` must be passed explicitly (not read from st.session_state) because this
    function runs inside ThreadPoolExecutor worker threads, where Streamlit's
    session_state is not accessible.

    `temperature` is forwarded to the provider-native endpoints when set. It is NOT
    applied on the web-search (Agent API) path — that endpoint's request shape doesn't
    document a temperature field, and answer variability there already comes from live
    search results rather than sampling, so run-to-run runs still differ.
    """
    if web_search:
        # Real web search is only available via the model-agnostic Agent Completions
        # API (capabilities.webSearch) — the provider-native endpoints below don't
        # support a built-in search tool. Routed separately since it's a different
        # request/response shape (Vercel AI SDK UIMessage, SSE streaming).
        return call_langdock_agent(api_key, messages, model, lang=lang)

    anthropic = _is_anthropic_model(model)
    google    = _is_google_model(model)
    if anthropic:
        url = ANTHROPIC_URL
    elif google:
        url = GOOGLE_URL_TEMPLATE.format(model=model)
    else:
        url = LANGDOCK_URL
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
        if temperature is not None:
            payload["temperature"] = temperature
    elif google:
        payload = {
            "contents": [
                {
                    "role":  "model" if m["role"] == "assistant" else "user",
                    "parts": [{"text": m["content"]}],
                }
                for m in messages
            ],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
            },
        }
        if temperature is not None:
            payload["generationConfig"]["temperature"] = temperature
    else:
        payload = {
            "model":                 model,
            "messages":              messages,
            "max_completion_tokens": max_tokens,
        }
        if temperature is not None:
            payload["temperature"] = temperature
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
            elif google:
                candidates = resp_data.get("candidates", [])
                parts      = candidates[0].get("content", {}).get("parts", []) if candidates else []
                text       = parts[0].get("text") if parts else None
                raw_u      = resp_data.get("usageMetadata", {})
                usage      = {
                    "prompt_tokens":     raw_u.get("promptTokenCount", 0),
                    "completion_tokens": raw_u.get("candidatesTokenCount", 0),
                    "total_tokens":      raw_u.get("totalTokenCount", 0),
                }
                finish = candidates[0].get("finishReason", "?") if candidates else "?"
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
                # Anthropic uses "max_tokens"; OpenAI uses "length"; Google uses "MAX_TOKENS"
                if finish in ("max_tokens", "length", "MAX_TOKENS"):
                    return None, tr(
                        f"Token-Budget erschöpft (max_tokens={max_tokens}). "
                        "Für Reasoning-Modelle wie gpt-5-mini werden mehr Tokens benötigt.",
                        f"Token budget exhausted (max_tokens={max_tokens}). "
                        "Reasoning models like gpt-5-mini need a larger budget.",
                        lang=lang,
                    ), usage
                return None, tr(
                    f"Modell lieferte keinen Inhalt (finish_reason={finish}).",
                    f"Model returned no content (finish_reason={finish}).",
                    lang=lang,
                ), usage

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
                401: tr("API-Key ungültig oder abgelaufen (401).", "API key invalid or expired (401).", lang=lang),
                403: tr("Zugriff verweigert (403). API-Key prüfen.", "Access denied (403). Check the API key.", lang=lang),
                404: tr(
                    f"Endpunkt nicht gefunden (404). Region prüfen: '{LANGDOCK_REGION}'. URL: {url}",
                    f"Endpoint not found (404). Check the region: '{LANGDOCK_REGION}'. URL: {url}",
                    lang=lang,
                ),
                422: tr(
                    f"Ungültige Anfrage (422) — wahrscheinlich falscher Modell-Name: '{model}'. Antwort: {body}",
                    f"Invalid request (422) — likely a wrong model name: '{model}'. Response: {body}",
                    lang=lang,
                ),
            }
            err = messages_map.get(status, tr(f"HTTP-Fehler {status}: {body}", f"HTTP error {status}: {body}", lang=lang))
            return None, err, {}

        except requests.exceptions.ConnectionError as e:
            err = tr(
                f"Keine Verbindung zur Langdock API. Internet-Verbindung prüfen. ({e})",
                f"Could not connect to the Langdock API. Check your internet connection. ({e})",
                lang=lang,
            )
            log.error("ConnectionError: %s", e)
            return None, err, {}

        except requests.exceptions.Timeout:
            if attempt < 3:
                log.warning("Timeout, retrying (attempt %d)", attempt + 1)
                time.sleep(2 ** attempt)
                continue
            err = tr(f"Timeout nach {REQUEST_TIMEOUT}s.", f"Timed out after {REQUEST_TIMEOUT}s.", lang=lang)
            log.error(err)
            return None, err, {}

        except (KeyError, IndexError, json.JSONDecodeError) as e:
            err = tr(f"Unerwartetes Antwort-Format: {e}", f"Unexpected response format: {e}", lang=lang)
            log.error("Parse error: %s | Response: %s", e,
                      r.text[:200] if 'r' in dir() else "no response")
            return None, err, {}

        except Exception as e:
            err = tr(f"Unbekannter Fehler: {e}", f"Unknown error: {e}", lang=lang)
            log.error("Unexpected error: %s", e)
            return None, err, {}

    return None, tr("Maximale Anzahl an Versuchen erreicht (4).", "Maximum number of retries reached (4).", lang=lang), {}


# ---------------------------------------------------------------------------
# Agent Completions API — real web search
# Routes through https://api.langdock.com/agent/v1/chat/completions using a
# temporary agent with capabilities.webSearch=true. This is the only Langdock
# endpoint that gives every provider's models (OpenAI/Anthropic/Google/...) a
# real, built-in web search tool via one flag — the provider-native endpoints
# used by call_langdock() above don't support a built-in search tool.
#
# Uses SSE streaming (Vercel AI SDK "UI Message Stream" protocol): Langdock
# kills non-streaming Agent API requests with an HTTP 524 after 100s, and
# web-search calls (search + read results + compose an answer) regularly run
# longer than that.
#
# Token usage is not documented for this endpoint's response, so usage always
# comes back as zeros for these calls (tokens_in/tokens_out show 0 in the UI).
# ---------------------------------------------------------------------------
def call_langdock_agent(
    api_key: str,
    messages: list[dict],
    model: str,
    lang: str = "de",
) -> tuple[str | None, str | None, dict]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }
    ui_messages = [
        {
            "id":    f"msg_{i}",
            "role":  m["role"],
            "parts": [{"type": "text", "text": m["content"]}],
        }
        for i, m in enumerate(messages)
    ]
    payload = {
        "agent": {
            "name": "Brand Visibility Assistant",
            # capabilities.webSearch only makes the tool available — the model still
            # decides on its own whether to call it. Without an explicit nudge, models
            # (especially smaller ones like Haiku) tend to fall back on their trained
            # "I don't have real-time access" response instead of trying the tool,
            # particularly for questions they're confident are "in the future".
            "instructions": (
                "You have a live web search tool with current results. For any question about "
                "products, brands, providers, rankings, prices, or recommendations, search the web "
                "first and base your answer on what you find — even if you feel you already know the "
                "answer, since your training data is outdated. Do not state or imply you lack "
                "real-time access; you have it. Only skip searching for purely timeless facts "
                "(definitions, basic concepts)."
            ),
            "model":        model,
            "capabilities": {"webSearch": True},
        },
        "messages": ui_messages,
        "stream":   True,
    }

    for attempt in range(4):
        try:
            t0 = time.time()
            with requests.post(
                AGENT_URL,
                json=payload,
                headers=headers,
                timeout=AGENT_STREAM_TIMEOUT,
                stream=True,
            ) as r:
                # Check the status code before touching the body — reading
                # e.response.text *after* the `with` block has closed the
                # connection (as happens if we let raise_for_status() raise
                # and catch it outside) can return an empty or truncated body.
                if r.status_code >= 400:
                    status = r.status_code
                    body   = _strip_html(r.text)
                    log.error("Agent HTTP error %s: %s", status, body)

                    if status == 429:
                        wait = 15 * (2 ** attempt) + random.uniform(0, 5)
                        log.info("Agent API rate limited (429), waiting %.1fs before retry (attempt %d/4)", wait, attempt + 1)
                        time.sleep(wait)
                        continue

                    messages_map = {
                        400: tr(
                            f"Ungültige Anfrage (400) — Modell '{model}' evtl. nicht über die Agent-API verfügbar. Antwort: {body}",
                            f"Invalid request (400) — model '{model}' may not be available via the Agent API. Response: {body}",
                            lang=lang,
                        ),
                        401: tr("API-Key ungültig oder abgelaufen (401).", "API key invalid or expired (401).", lang=lang),
                        403: tr("Zugriff verweigert (403). API-Key prüfen.", "Access denied (403). Check the API key.", lang=lang),
                    }
                    err = messages_map.get(status, tr(f"HTTP-Fehler {status}: {body}", f"HTTP error {status}: {body}", lang=lang))
                    return None, err, {}

                full_text   = ""
                error_text  = None
                web_search  = False   # a search tool actually fired
                tools_used  = set()   # every tool the model invoked (by name)
                sources: list[dict] = []  # cited pages (authoritative proof of a search)
                seen_urls   = set()
                types_seen  = set()   # every stream event type — logged for diagnosis

                def _harvest_urls(obj):
                    # Recursively pull {"url": ..., "title"/"name": ...} pairs out of an
                    # arbitrary tool/source payload — the Agent stream nests search
                    # results differently across models, so we don't assume a shape.
                    if isinstance(obj, dict):
                        url = obj.get("url")
                        if isinstance(url, str) and url.startswith("http") and url not in seen_urls:
                            seen_urls.add(url)
                            sources.append({"url": url, "title": obj.get("title") or obj.get("name") or ""})
                        for v in obj.values():
                            _harvest_urls(v)
                    elif isinstance(obj, list):
                        for v in obj:
                            _harvest_urls(v)

                for line in r.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[len("data:"):].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        event = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    etype = event.get("type", "")
                    types_seen.add(etype)
                    if etype == "text-delta":
                        full_text += event.get("delta", "")
                    elif etype == "error":
                        error_text = event.get("errorText", "Agent stream error")
                    elif etype == "finish":
                        break
                    elif etype.startswith("tool-"):
                        tool_name = etype[len("tool-"):]
                        tools_used.add(tool_name)
                        if "search" in tool_name.lower():
                            web_search = True
                        _harvest_urls(event.get("output"))
                        _harvest_urls(event.get("input"))
                    elif etype.startswith("source"):
                        # source-url / source-document events — hard citation evidence.
                        web_search = True
                        _harvest_urls(event)

                # Any harvested source URL is itself proof a search ran — regardless of
                # which event carried it or whether the tool name contained "search".
                if sources:
                    web_search = True

                # Models embed inline citation markers (【toolu_…】 / 【call_…】) in the
                # answer text itself. Their presence is proof a tool ran even when the
                # stream's tool/source events use a shape we didn't catch above. Strip
                # them so the stored/displayed/analyzed answer is clean.
                full_text, n_citations = _strip_citation_markers(full_text)
                if n_citations:
                    web_search = True

                elapsed = time.time() - t0
                log.info(
                    "HTTP %s | agent attempt %d | model=%s | web_search_used=%s | tools=%s | sources=%d | citations=%d | types=%s | %.2fs",
                    r.status_code, attempt + 1, model, web_search, sorted(tools_used),
                    len(sources), n_citations, sorted(types_seen), elapsed,
                )

            meta = {
                "web_search_used": web_search,
                "tools_used":      sorted(tools_used),
                "sources":         sources,
                "citation_count":  n_citations,
            }

            if error_text:
                log.warning("Agent stream error: %s", error_text)
                return None, tr(f"Agent-Fehler: {error_text}", f"Agent error: {error_text}", lang=lang), {}

            if not full_text:
                log.warning("Agent stream returned no text (model=%s)", model)
                return None, tr(
                    "Kein Inhalt vom Agent erhalten — evtl. wird dieses Modell nicht von der Langdock Agent-API "
                    "unterstützt (siehe /agent/v1/models).",
                    "No content received from the agent — this model may not be supported by the Langdock "
                    "Agent API (see /agent/v1/models).",
                    lang=lang,
                ), {}

            return full_text, None, meta

        except requests.exceptions.ConnectionError as e:
            err = tr(
                f"Keine Verbindung zur Langdock Agent-API. Internet-Verbindung prüfen. ({e})",
                f"Could not connect to the Langdock Agent API. Check your internet connection. ({e})",
                lang=lang,
            )
            log.error("Agent ConnectionError: %s", e)
            return None, err, {}

        except requests.exceptions.Timeout:
            if attempt < 3:
                log.warning("Agent request timeout, retrying (attempt %d)", attempt + 1)
                time.sleep(2 ** attempt)
                continue
            err = tr(f"Timeout nach {AGENT_STREAM_TIMEOUT}s.", f"Timed out after {AGENT_STREAM_TIMEOUT}s.", lang=lang)
            log.error(err)
            return None, err, {}

        except Exception as e:
            err = tr(f"Unbekannter Fehler (Agent-API): {e}", f"Unknown error (Agent API): {e}", lang=lang)
            log.error("Agent unexpected error: %s", e)
            return None, err, {}

    return None, tr("Maximale Anzahl an Versuchen erreicht (4).", "Maximum number of retries reached (4).", lang=lang), {}


# ---------------------------------------------------------------------------
# Connection test
# Makes a minimal API call to verify credentials and model name.
# Shown in Step 1 so problems are caught before a long run starts.
# ---------------------------------------------------------------------------
def test_connection(api_key: str, model: str, web_search: bool = False) -> tuple[bool, str]:
    """
    Minimal call to verify credentials and model name.
    Uses the same payload shape as brand_monitor.py.
    When `web_search` is set, tests the Agent Completions API path instead —
    the endpoint actually used once web search is enabled for the run.
    """
    text, err, _ = call_langdock(
        api_key,
        [{"role": "user", "content": "Say 'OK' and nothing else."}],
        model=model,
        max_tokens=MAX_TOKENS,  # use same value as main calls, not a small number
        web_search=web_search,
        lang=st.session_state.get("lang", "de"),
    )
    if text:
        return True, tr(
            f"Verbindung erfolgreich. Modell antwortet: '{text.strip()[:80]}'",
            f"Connection successful. Model responded: '{text.strip()[:80]}'",
        )
    return False, err or tr("Keine Antwort erhalten.", "No response received.")


# ---------------------------------------------------------------------------
# Question generation (Step 1 API call)
# Sends the topic and asks the model to generate N questions.
# Returns a list of question strings, one per line.
# ---------------------------------------------------------------------------
# Strips list scaffolding models add despite being told not to: "1. ", "1) ",
# "- ", "* ", "• ", and surrounding quotes.
_Q_PREFIX_RE = re.compile(r'^\s*(?:\d+[.)]\s*|[-*•]\s*)+')

def _clean_question_line(line: str) -> str:
    q = _Q_PREFIX_RE.sub("", line).strip()
    if len(q) >= 2 and q[0] in "\"'“„" and q[-1] in "\"'”“":
        q = q[1:-1].strip()
    return q

def _dedupe_questions(questions: list[str]) -> list[str]:
    """Case-insensitive, whitespace-normalized dedupe that preserves first-seen order."""
    out, seen = [], set()
    for q in questions:
        key = re.sub(r"\s+", " ", q).strip().lower().rstrip("?.!")
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(q)
    return out

def generate_questions(api_key: str, topic: str, n: int, model: str) -> tuple[list[str], str | None]:
    lang = st.session_state.get("lang", "de")
    if lang == "de":
        prompt = (
            f"Generiere genau {n} verschiedene, neutrale Fragen zum Thema: \"{topic}\".\n\n"
            "Die Fragen sollen:\n"
            "- Typische Nutzerfragen sein, die jemand einem KI-Assistenten stellen würde\n"
            "- Verschiedene Aspekte des Themas abdecken (Empfehlungen, Vergleiche, Eigenschaften, Use Cases)\n"
            "- Sich inhaltlich klar voneinander unterscheiden (keine Umformulierungen derselben Frage)\n"
            "- So formuliert sein, dass die Antwort natürlicherweise Marken, Produkte oder Anbieter nennen würde\n\n"
            "Antworte NUR mit einem validen JSON-Array von Strings (ohne Markdown, ohne Einleitung), "
            f'z.B. ["Frage 1?", "Frage 2?"]. Genau {n} Elemente. Fragen auf Deutsch.'
        )
    else:
        prompt = (
            f"Generate exactly {n} different, neutral questions about the topic: \"{topic}\".\n\n"
            "The questions should:\n"
            "- Be typical user questions someone would ask an AI assistant\n"
            "- Cover different aspects of the topic (recommendations, comparisons, features, use cases)\n"
            "- Be clearly distinct from one another (no rephrasings of the same question)\n"
            "- Be phrased so that the answer would naturally mention brands, products, or providers\n\n"
            "Respond ONLY with a valid JSON array of strings (no markdown, no introduction), "
            f'e.g. ["Question 1?", "Question 2?"]. Exactly {n} elements. Questions in English.'
        )
    text, err, _ = call_langdock(
        api_key,
        [{"role": "user", "content": prompt}],
        model=model,
        max_tokens=QUESTION_MAX_TOKENS,  # reasoning models consume the budget for internal thinking + output
        lang=lang,
        temperature=QUESTION_TEMPERATURE,
    )
    if not text:
        return [], err

    # Prefer the requested JSON array; fall back to line-splitting if the model
    # ignored the format. Either way, strip list scaffolding and dedupe.
    parsed = _parse_json_array(text)
    if parsed:
        raw = [str(q) for q in parsed if isinstance(q, (str, int, float))]
    else:
        raw = text.strip().splitlines()

    questions = _dedupe_questions(
        [q for q in (_clean_question_line(line) for line in raw) if q]
    )

    if not questions:
        return [], tr(
            "Es konnten keine Fragen aus der Antwort extrahiert werden.",
            "No questions could be extracted from the response.",
            lang=lang,
        )
    if len(questions) > n:
        questions = questions[:n]  # model over-generated; trim to the requested count
    if len(questions) < n:
        # Fewer than requested after dedupe — usually near-duplicates were collapsed.
        # Not an error: Step 2 shows the real count and lets the user add more.
        log.info("generate_questions: requested %d, got %d after cleaning/dedupe", n, len(questions))
    return questions, None


# ---------------------------------------------------------------------------
# Main question run (Step 3 API calls)
# Asks a single question in the context of the topic.
# Returns (answer_text, error_message).
# ---------------------------------------------------------------------------
def ask_question(
    api_key: str, question: str, model: str, lang: str,
    web_search: bool = False, max_tokens: int = MAX_TOKENS,
    short_answer: bool = False,
) -> tuple[str | None, str | None, dict]:
    # `lang` is passed in explicitly (captured on the main thread before submitting
    # to ThreadPoolExecutor) — st.session_state is not accessible from worker threads.
    # No "use current knowledge" prompt hint needed here — when web_search is True,
    # call_langdock() routes this to the Agent API, which gives the model a real
    # web search tool instead of just asking it to pretend it has fresh knowledge.
    if lang == "de":
        if short_answer:
            content = (
                "Du bist ein hilfreicher Assistent.\n"
                "Antworte ausschließlich mit einer stichpunktartigen Auflistung. "
                "Jede Zeile: Markenname — ein Satz Begründung. "
                "Keine Einleitung, kein Fazit, kein Fließtext.\n\n"
                f"Frage: {question}"
            )
        else:
            content = (
                "Du bist ein hilfreicher Assistent.\n"
                "Beantworte die folgende Frage sachlich und ausführlich.\n\n"
                f"Frage: {question}"
            )
    else:
        if short_answer:
            content = (
                "You are a helpful assistant.\n"
                "Respond exclusively with a bullet-point list. "
                "Each line: brand name — one sentence of reasoning. "
                "No introduction, no conclusion, no prose.\n\n"
                f"Question: {question}"
            )
        else:
            content = (
                "You are a helpful assistant.\n"
                "Answer the following question factually and in detail.\n\n"
                f"Question: {question}"
            )
    return call_langdock(
        api_key,
        [{"role": "user", "content": content}],
        model=model,
        max_tokens=max_tokens,
        web_search=web_search,
        lang=lang,
        # Non-zero so repeated runs of the same question vary — otherwise the
        # multiple-runs statistic is meaningless. No effect on the web-search path.
        temperature=COLLECTION_TEMPERATURE,
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
# Web-search citation markers. When the Agent API's web search runs, models embed
# inline citation references into the answer text pointing at the tool-call that
# produced each fact. The prefix differs by provider:
#   Anthropic → 【toolu_vrtx_014m5maSG2ZTGFn9...-5】
#   OpenAI    → 【call_ft31fjFd1YmhnAeYTDMVo6uI-2】
# These pollute the displayed answer AND the brand-analysis input, so we strip
# them — but their presence is also hard proof a tool (search) actually ran, so
# we report how many were found. Matches full-width 【…】 and ASCII […] brackets
# around a `<prefix>_<id>` token for any known tool-call id prefix.
# ---------------------------------------------------------------------------
_CITATION_RE = re.compile(r"[【\[]\s*(?:toolu|call|fc|tool|resp|ws)_[A-Za-z0-9_\-]+\s*[】\]]")

def _strip_citation_markers(text: str) -> tuple[str, int]:
    """Returns (cleaned_text, marker_count)."""
    if not text:
        return text, 0
    count = len(_CITATION_RE.findall(text))
    if not count:
        return text, 0
    clean = _CITATION_RE.sub("", text)
    clean = re.sub(r"[ \t]+([.,;:!?])", r"\1", clean)  # tidy " ." → "."
    clean = re.sub(r"[ \t]{2,}", " ", clean)            # collapse double spaces
    return clean.strip(), count


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
# Robust JSON-object extractor — same three strategies as _parse_json_array but
# for a top-level {...}. Returns {} on total failure.
# ---------------------------------------------------------------------------
def _parse_json_object(text: str) -> dict:
    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

    fence = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    while start != -1:
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        result = json.loads(candidate)
                        if isinstance(result, dict):
                            return result
                    except json.JSONDecodeError:
                        pass
                    break
        start = text.find("{", start + 1)

    log.warning("Analysis JSON object parse failed. Response: %s", text[:300])
    return {}


# ---------------------------------------------------------------------------
# Whole-dataset brand analysis. The collected dataset is split into batches
# (see _make_analysis_batches) and each batch is sent to a single strong model
# (ANALYSIS_MODEL). Batching keeps a large run's output JSON from being silently
# truncated — a single mega-call that hits finish_reason=max_tokens produces
# unparseable JSON and loses ALL analysis, whereas a bad batch loses only itself.
# Each answer in a batch is numbered locally; the model returns a flat array whose
# "index" ties each brand mention back to its answer, which we regroup (offsetting
# the local index by the batch's start position) per answer.
#
# Works in both modes:
#   brands=[]    → auto: model identifies all brands
#   brands=[...] → manual: model still detects all brands (so unlisted competitors
#                  can be surfaced), using the given names as canonical spellings;
#                  the client filters to the listed brands (see _run_brand_analysis).
# ---------------------------------------------------------------------------
def _truncate_for_analysis(text: str) -> str:
    """Head+tail truncation so brands listed near the END of a long answer survive."""
    text = text or ""
    if len(text) <= ANALYSIS_ANSWER_CHARS:
        return text
    head_len = int(ANALYSIS_ANSWER_CHARS * ANALYSIS_ANSWER_HEAD_FRAC)
    tail_len = ANALYSIS_ANSWER_CHARS - head_len
    return text[:head_len].rstrip() + " […] " + text[-tail_len:].lstrip()


def _make_analysis_batches(answers: list[dict]) -> list[tuple[int, list[dict]]]:
    """
    Splits answers into (start_index, sublist) batches bounded by both an answer
    count and an estimated input-token budget, whichever is reached first.
    """
    batches: list[tuple[int, list[dict]]] = []
    i, n = 0, len(answers)
    while i < n:
        j, tok = i, 0
        while j < n and (j - i) < ANALYSIS_BATCH_MAX_ANSWERS:
            ans_chars = min(len(answers[j].get("answer") or ""), ANALYSIS_ANSWER_CHARS)
            tok += ans_chars // 4 + len(answers[j].get("question") or "") // 4 + 40
            if tok > ANALYSIS_BATCH_MAX_INPUT_TOKENS and j > i:
                break
            j += 1
        batches.append((i, answers[i:j]))
        i = j
    return batches


def _build_analysis_prompt(batch: list[dict], brands: list[str], lang: str) -> str:
    """Builds the analysis prompt for one batch; answers are numbered [0..len-1] locally."""
    blocks = []
    for i, a in enumerate(batch):
        ans = _truncate_for_analysis(a.get("answer") or "")
        blocks.append(f"[{i}] Frage: {a['question']}\nAntwort: {ans}")
    corpus = "\n\n".join(blocks)

    if lang == "de":
        if brands:
            brand_list = ", ".join(f'"{b}"' for b in brands)
            scope = (
                "Erkenne alle genannten Marken, Produkte und Anbieter. "
                f"Besonders wichtig sind diese Marken: {brand_list}. "
                "Verwende für sie exakt die angegebene Schreibweise. "
                "Nenne aber auch alle anderen tatsächlich erwähnten Marken."
            )
        else:
            scope = (
                "Erkenne alle genannten Marken, Produkte und Anbieter. "
                "Fasse Schreibvarianten, Einheiten und Rebrandings derselben Marke zu EINEM Namen zusammen "
                "(z.B. 'Havas Health' und 'Havas Life' → 'Havas Health / Havas Life')."
            )
        prompt = (
            "Du bist ein Analyst für Markensichtbarkeit in LLM-Antworten. "
            "Unten stehen nummerierte Antworten eines Sprachmodells auf verschiedene Fragen.\n\n"
            f"{scope}\n\n"
            "Antworte NUR mit einem validen JSON-Objekt (ohne Markdown) mit genau zwei Feldern:\n\n"
            '"mentions": Ein Array. Für JEDE Antwort und JEDE darin tatsächlich erwähnte Marke ein Objekt mit:\n'
            '  - "index": die Nummer der Antwort in eckigen Klammern (Ganzzahl)\n'
            '  - "brand": Markenname (normalisiert)\n'
            '  - "sentiment": "positive", "neutral" oder "negative"\n'
            '  - "confidence": "high", "medium" oder "low"\n'
            '  - "reason": Ein Satz auf Deutsch, der das Sentiment begründet\n'
            '  - "aspect": Hauptaspekt, z.B. "Qualität", "Preis", "Empfehlung", "Bekanntheit", "Funktionen"\n'
            '  - "excerpt": Relevanter Satz aus der Antwort (max 200 Zeichen)\n'
            '  - "rank": Position dieser Marke in der Antwort (1 = zuerst genannte Marke), Ganzzahl\n'
            "  Wird eine Marke innerhalb derselben Antwort mehrfach genannt, gib sie für diese Antwort nur EINMAL aus.\n\n"
            '"summary": Eine kurze, sachliche Zusammenfassung auf Deutsch (3–5 Sätze) über den gesamten Datensatz: '
            "welche Marken dominieren (Sichtbarkeit/Nennungen), welche auffällig selten oder gar nicht vorkommen, "
            "und die vorherrschende Tonalität. Keine Aufzählung, Fließtext.\n\n"
            "Wenn keine Marken erkannt werden: {\"mentions\": [], \"summary\": \"...\"}\n\n"
            "=== ANTWORTEN ===\n"
            f"{corpus}"
        )
    else:
        if brands:
            brand_list = ", ".join(f'"{b}"' for b in brands)
            scope = (
                "Detect all brands, products, and providers mentioned. "
                f"These brands are of particular interest: {brand_list}. "
                "Use exactly the given spelling for them. "
                "But also report every other brand actually mentioned."
            )
        else:
            scope = (
                "Detect all brands, products, and providers mentioned. "
                "Merge spelling variants, units, and rebrandings of the same brand into ONE name "
                "(e.g. 'Havas Health' and 'Havas Life' → 'Havas Health / Havas Life')."
            )
        prompt = (
            "You are an analyst for brand visibility in LLM answers. "
            "Below are numbered answers a language model gave to various questions.\n\n"
            f"{scope}\n\n"
            "Respond ONLY with a valid JSON object (no markdown) with exactly two fields:\n\n"
            '"mentions": An array. For EACH answer and EACH brand actually mentioned in it, an object with:\n'
            '  - "index": the answer number shown in square brackets (integer)\n'
            '  - "brand": brand name (normalized)\n'
            '  - "sentiment": "positive", "neutral", or "negative"\n'
            '  - "confidence": "high", "medium", or "low"\n'
            '  - "reason": one sentence in English justifying the sentiment\n'
            '  - "aspect": main aspect, e.g. "quality", "price", "recommendation", "reputation", "features"\n'
            '  - "excerpt": relevant sentence from the answer (max 200 characters)\n'
            '  - "rank": position of this brand within the answer (1 = first brand mentioned), integer\n'
            "  If a brand is mentioned several times within the same answer, include it only ONCE for that answer.\n\n"
            '"summary": A short, factual summary in English (3–5 sentences) about the whole dataset: '
            "which brands dominate (visibility/mentions), which are notably rare or absent, and the prevailing "
            "sentiment. Prose, not a list.\n\n"
            "If no brands are detected: {\"mentions\": [], \"summary\": \"...\"}\n\n"
            "=== ANSWERS ===\n"
            f"{corpus}"
        )
    return prompt


_MENTION_FIELDS = ("brand", "sentiment", "confidence", "reason", "aspect", "excerpt", "rank")


def _analyze_batch(
    api_key: str, batch: list[dict], brands: list[str], lang: str,
) -> tuple[list[dict], str, dict, str | None]:
    """Runs one batch. Returns (mentions, summary, usage, error). Mentions keep the
    batch-local "index" as returned by the model — the caller offsets it."""
    prompt = _build_analysis_prompt(batch, brands, lang)
    text, err, usage = call_langdock(
        api_key,
        [{"role": "user", "content": prompt}],
        model=ANALYSIS_MODEL,
        max_tokens=DATASET_ANALYSIS_MAX_TOKENS,
        lang=lang,
        temperature=ANALYSIS_TEMPERATURE,
    )
    if not text:
        log.warning("Dataset analysis batch failed: %s", err)
        return [], "", usage, err

    obj  = _parse_json_object(text)
    # Backward-compatible fallback: if the model returned a bare array instead of the
    # {mentions, summary} object, treat the array as the mentions list.
    flat = obj.get("mentions") if isinstance(obj.get("mentions"), list) else _parse_json_array(text)
    summary = (obj.get("summary") or "").strip() if isinstance(obj, dict) else ""

    # Distinguish "genuinely no brands" from a parse failure / truncated JSON. A clean
    # empty result ([] or {}) is legitimate; a non-empty response that parsed to nothing
    # means the output was unreadable (likely truncated) — surface it.
    parse_err = None
    if not flat and text.strip() not in ("[]", "{}", "") and not summary:
        parse_err = tr(
            "Ein Analyse-Batch konnte nicht als JSON gelesen werden (evtl. abgeschnitten).",
            "An analysis batch could not be parsed as JSON (possibly truncated).",
            lang=lang,
        )
        log.warning("Analysis batch: non-empty response but empty parse. First 300 chars: %s", text[:300])

    return (flat or []), summary, usage, parse_err


def _summarize_dataset(
    api_key: str, by_index: dict[int, list[dict]], n_answers: int, lang: str,
) -> tuple[str, dict, str | None]:
    """Produces one executive summary from aggregated stats. Used only when the dataset
    spans multiple batches (each batch's own summary would see only part of the data)."""
    from collections import Counter
    coverage, sentiment = Counter(), Counter()
    for entries in by_index.values():
        for e in entries:
            b = (e.get("brand") or "").strip()
            if not b:
                continue
            coverage[b] += 1
            sentiment[(e.get("sentiment") or "neutral")] += 1
    if not coverage:
        return "", {}, None

    stats = "\n".join(f"{b}: {c}/{n_answers}" for b, c in coverage.most_common(20))
    sent  = ", ".join(f"{k}: {v}" for k, v in sentiment.items())
    if lang == "de":
        prompt = (
            "Du erhältst aggregierte Statistiken einer Markensichtbarkeits-Analyse über einen ganzen "
            "Datensatz. Schreibe eine kurze, sachliche Zusammenfassung (3–5 Sätze, Fließtext, keine "
            "Aufzählung): welche Marken dominieren, welche selten/gar nicht vorkommen, und die "
            f"vorherrschende Tonalität.\n\nMarken-Abdeckung (Antworten mit Nennung):\n{stats}\n\n"
            f"Sentiment-Verteilung: {sent}"
        )
    else:
        prompt = (
            "You are given aggregated statistics from a brand-visibility analysis over a whole dataset. "
            "Write a short, factual summary (3–5 sentences, prose, not a list): which brands dominate, "
            "which are rare or absent, and the prevailing sentiment.\n\n"
            f"Brand coverage (answers mentioning the brand):\n{stats}\n\nSentiment distribution: {sent}"
        )
    text, err, usage = call_langdock(
        api_key,
        [{"role": "user", "content": prompt}],
        model=ANALYSIS_MODEL,
        max_tokens=1000,
        lang=lang,
        temperature=ANALYSIS_TEMPERATURE,
    )
    return (text or "").strip(), usage, err


def analyze_dataset(
    api_key: str,
    answers: list[dict],
    brands: list[str],
    lang: str,
) -> tuple[dict[int, list[dict]], str, dict, str | None]:
    """
    Returns (by_index, summary, usage, error).
      by_index: {answer_index: [ {brand, sentiment, confidence, reason, aspect, excerpt, rank}, ... ]}
      summary:  a short executive-summary paragraph
      usage:    combined token usage across all analysis calls
      error:    human-readable error string, or None on success
    `answers` is a list of dicts each with "question" and "answer"; the list index is
    the stable key used in the returned mapping.
    """
    batches = _make_analysis_batches(answers)
    by_index: dict[int, list[dict]] = {}
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    errors: list[str] = []
    batch_summaries: list[str] = []

    for start, batch in batches:
        flat, b_summary, usage, err = _analyze_batch(api_key, batch, brands, lang)
        for k in total_usage:
            total_usage[k] += usage.get(k, 0)
        if err:
            errors.append(err)
        if b_summary:
            batch_summaries.append(b_summary)
        for item in flat:
            if not isinstance(item, dict):
                continue
            try:
                idx = int(item.get("index"))
            except (TypeError, ValueError):
                continue
            # Guard against a hallucinated/out-of-range index — without this the
            # mention would land under a key the reconstruction loop never reads and
            # be silently dropped (or misattributed). Validate against THIS batch.
            if not (0 <= idx < len(batch)):
                log.warning("Analysis: index %s out of range (batch len=%d, start=%d) — dropped", idx, len(batch), start)
                continue
            entry = {k: item.get(k, "") for k in _MENTION_FIELDS}
            by_index.setdefault(start + idx, []).append(entry)

    # Summary: a single batch already produced one over all its data; multiple batches
    # each saw only a slice, so synthesize one from the aggregate instead.
    if len(batches) <= 1:
        summary = batch_summaries[0] if batch_summaries else ""
    else:
        summary, s_usage, _ = _summarize_dataset(api_key, by_index, len(answers), lang)
        for k in total_usage:
            total_usage[k] += s_usage.get(k, 0)
        if not summary and batch_summaries:
            summary = batch_summaries[0]

    # De-duplicate identical batch errors so the user sees one message, not one per batch.
    error = None
    if errors:
        uniq = list(dict.fromkeys(errors))
        error = "; ".join(uniq)
        if any("truncat" in e.lower() or "abgeschnitten" in e.lower() for e in uniq):
            error += " " + tr(
                "Betroffene Batches können unvollständig sein.",
                "Affected batches may be incomplete.",
                lang=lang,
            )
    return by_index, summary, total_usage, error


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
                "model":      r.get("model", ""),
                "brand":      b.get("brand", ""),
                "sentiment":  b.get("sentiment", "neutral"),
                "confidence": b.get("confidence", ""),
                "reason":     b.get("reason", ""),
                "aspect":     b.get("aspect", ""),
                "excerpt":    b.get("excerpt", b.get("context", "")),
                "rank":       pd.to_numeric(b.get("rank"), errors="coerce"),
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
                "model":         r.get("model", model),
                "prompt":        r["question"],
                "repetition":    r["run"],
                "brands_found":  brands_str,
                "mention_count": len(r.get("brands_found", [])),
                "raw_response":  r.get("answer", "")[:500].replace("\n", " "),
            })
    return filename


# ---------------------------------------------------------------------------
# Per-step tutorial. Collapsed expander at the top of each page with a short
# "what to do here" description, so the workflow is self-explanatory without
# cluttering the page. Kept collapsed by default (a single click to open).
# ---------------------------------------------------------------------------
_TUTORIALS = {
    1: (
        "❓ Anleitung — Schritt 1: Einrichtung",
        "❓ Tutorial — Step 1: Setup",
        """
1. **API-Key** eingeben (Langdock).
2. **Websuche** ist standardmäßig aktiv — die Modelle antworten dann mit echter Web-Recherche.
3. **Modell** wählen, mit dem die Antworten gesammelt werden.
4. **Fragen-Modus** wählen: automatisch generieren (dann Thema + Anzahl angeben) oder eigene Fragen eintippen.
5. **Brand-Erkennung**: eigene Marken vorgeben oder automatisch aus den Antworten extrahieren lassen.
6. Optional **Verbindung testen**, dann weiter.
""",
        """
1. Enter your **API key** (Langdock).
2. **Web search** is on by default — models then answer with real web research.
3. Pick the **model** used to collect the answers.
4. Choose a **question mode**: auto-generate (then give a topic + count) or type your own questions.
5. **Brand detection**: specify your own brands, or let them be extracted automatically from the answers.
6. Optionally **test the connection**, then continue.
""",
    ),
    2: (
        "❓ Anleitung — Schritt 2: Fragen prüfen",
        "❓ Tutorial — Step 2: Review questions",
        """
Jede Zeile ist eine Frage. Bearbeite, lösche oder ergänze sie frei — nur diese Fragen werden anschließend an das Modell gestellt.
""",
        """
Each line is one question. Edit, delete, or add freely — only these questions will be sent to the model.
""",
    ),
    3: (
        "❓ Anleitung — Schritt 3: Runs konfigurieren",
        "❓ Tutorial — Step 3: Configure runs",
        """
- **Runs pro Frage**: wie oft jede Frage wiederholt wird (mehr Runs = stabilere Statistik).
- **Parallele Calls**: die Voreinstellung bleibt sicher unter dem Rate-Limit.
- Danach werden die **Rohdaten** gesammelt; die Marken-/Sentiment-Analyse folgt erst im nächsten Schritt.
""",
        """
- **Runs per question**: how often each question is repeated (more runs = more stable statistics).
- **Parallel calls**: the default stays safely under the rate limit.
- This collects the **raw data**; the brand/sentiment analysis only happens in the next step.
""",
    ),
    4: (
        "❓ Anleitung — Schritt 4: Rohdaten",
        "❓ Tutorial — Step 4: Raw data",
        """
Prüfe und exportiere die gesammelten Antworten. Du kannst dann die **Analyse starten** (ein Durchlauf mit Claude Opus 4.8), **Einstellungen ändern** (z.B. anderes Modell, gleiche Fragen) oder **neu beginnen**.
""",
        """
Inspect and export the collected answers. You can then **start the analysis** (a single pass with Claude Opus 4.8), **change settings** (e.g. a different model, same questions), or **start over**.
""",
    ),
    5: (
        "❓ Anleitung — Schritt 5: Ergebnisse",
        "❓ Tutorial — Step 5: Results",
        """
Sichtbarkeit (Share of Voice), Sentiment und Rohdaten je Marke. Über die Tabs wechselst du zwischen den Ansichten; unten kannst du alles als CSV/JSON exportieren.
""",
        """
Visibility (share of voice), sentiment, and raw data per brand. Use the tabs to switch views; export everything as CSV/JSON at the bottom.
""",
    ),
}

def render_tutorial(step: int):
    entry = _TUTORIALS.get(step)
    if not entry:
        return
    title_de, title_en, body_de, body_en = entry
    with st.expander(tr(title_de, title_en), expanded=False):
        st.markdown(tr(body_de, body_en))


# ===========================================================================
# UI — Step 1: Setup
# Collects all configuration and allows testing the connection before running.
# ===========================================================================
def render_step1():
    st.title("📊 LLM Brand Visibility")
    st.caption(tr(
        "Messe, wie LLMs Marken in beliebigen Themen wahrnehmen und empfehlen.",
        "Measure how LLMs perceive and recommend brands across any topic.",
    ))
    render_tutorial(1)
    st.divider()

    col_form, col_info = st.columns([3, 2])

    with col_form:
        st.subheader(tr("Konfiguration", "Configuration"))

        api_key = st.text_input(
            "Langdock API Key",
            type="password",
            placeholder="sk--...",
            help=tr("Wird nicht gespeichert. Nur für diese Session.", "Not stored. Only used for this session."),
        )

        web_search = st.checkbox(
            tr("Websuche aktivieren", "Enable web search"),
            value=True,
            help=tr(
                "Echte Websuche über die Langdock Agent-API (funktioniert mit jedem Modell). "
                "Läuft über einen separaten Endpunkt ohne Token-Reporting und kann pro Call länger dauern. "
                "Die Modell-Liste unten wechselt dann auf den Agent-API-Katalog (andere Modell-IDs).",
                "Real web search via the Langdock Agent API (works with any model). "
                "Runs through a separate endpoint with no token reporting, and can take longer per call. "
                "The model list below switches to the Agent API's catalog (different model IDs) when enabled.",
            ),
        )

        custom_label = tr("Benutzerdefiniert...", "Custom...")
        if web_search:
            # The Agent API has its own model catalog with its own IDs — fetch it
            # live instead of using the hardcoded provider model list above.
            if not api_key:
                st.info(tr(
                    "API-Key eingeben, um die für die Websuche verfügbaren Modelle zu laden.",
                    "Enter an API key to load the models available for web search.",
                ))
                agent_models, agent_models_err = [], None
            else:
                agent_models, agent_models_err = list_agent_models(api_key)
            if agent_models_err:
                st.warning(tr(
                    f"Modell-Liste (Agent-API) konnte nicht geladen werden: {agent_models_err}",
                    f"Could not load the model list (Agent API): {agent_models_err}",
                ))
            if agent_models:
                model = st.selectbox(
                    tr("Modell (Agent-API)", "Model (Agent API)"),
                    options=agent_models,
                    help=tr(
                        "Nur Modelle, die über die Langdock Agent-API verfügbar sind — IDs unterscheiden sich "
                        "von den Standard-Modell-IDs (z.B. 'claude-opus-4-7@default' statt 'claude-opus-4-7').",
                        "Only models available via the Langdock Agent API — IDs differ from the standard "
                        "model IDs (e.g. 'claude-opus-4-7@default' instead of 'claude-opus-4-7').",
                    ),
                )
            else:
                model = st.text_input(
                    tr("Modell-ID (Agent-API, manuell)", "Model ID (Agent API, manual)"),
                    placeholder="z.B. claude-opus-4-7@default",
                )
        else:
            model_option = st.selectbox(
                tr("Modell", "Model"),
                options=_MODEL_OPTIONS,
                index=0,  # gpt-5-mini
                help=tr(
                    "Verfügbare Modelle deines Langdock-Workspace. 'Benutzerdefiniert...' für andere.",
                    "Models available in your Langdock workspace. 'Custom...' for others.",
                ),
            )
            if model_option == custom_label:
                model = st.text_input(
                    tr("Modell-Name (benutzerdefiniert)", "Model name (custom)"),
                    placeholder=tr("z.B. gpt-4o-search-preview", "e.g. gpt-4o-search-preview"),
                )
            else:
                model = _model_id_from_option(model_option)

        st.markdown(f"**{tr('Fragen', 'Questions')}**")
        opt_gen      = tr("Automatisch generieren", "Auto-generate")
        opt_manual_q = tr("Eigene Fragen eingeben", "Enter your own questions")
        question_mode = st.radio(
            tr("Fragen-Modus", "Question mode"),
            [opt_gen, opt_manual_q],
            help=tr(
                "Automatisch: das Modell generiert Fragen zu einem Thema (ein Extra-Call). "
                "Eigene Fragen: direkt eingeben, kein Thema und kein Generierungs-Call nötig.",
                "Auto-generate: the model creates questions about a topic (one extra call). "
                "Your own: enter them directly, no topic and no generation call needed.",
            ),
        )

        # Topic + count are only relevant for auto-generation. In manual mode the user
        # supplies the questions directly, so we don't ask for a topic at all.
        topic = ""
        manual_questions_input = ""
        if question_mode == opt_gen:
            topic = st.text_area(
                tr("Thema / Kontext", "Topic / Context"),
                placeholder=tr(
                    "z.B. 'High-Performance Sportwagen im DACH-Markt'\n"
                    "oder 'CRM-Software für mittelständische Unternehmen'",
                    "e.g. 'High-performance sports cars in the DACH market'\n"
                    "or 'CRM software for mid-sized companies'",
                ),
                height=100,
            )
            n_questions = st.slider(
                tr("Anzahl Fragen generieren", "Number of questions to generate"),
                min_value=5, max_value=200, value=20, step=5,
            )
        else:
            n_questions = 0
            manual_questions_input = st.text_area(
                tr("Eigene Fragen (eine pro Zeile)", "Your own questions (one per line)"),
                height=150,
                placeholder=tr(
                    "Was sind die besten CRM-Tools für kleine Unternehmen?\n"
                    "Welche Anbieter empfiehlst du für Cloud-Hosting?",
                    "What are the best CRM tools for small businesses?\n"
                    "Which providers would you recommend for cloud hosting?",
                ),
            )

        st.markdown(f"**{tr('Brand-Erkennung', 'Brand detection')}**")
        opt_manual = tr("Manuell — Brands vorgeben", "Manual — specify brands")
        opt_auto   = tr("Automatisch — aus Antworten extrahieren", "Automatic — extract from answers")
        brand_mode = st.radio(
            tr("Modus", "Mode"),
            [opt_manual, opt_auto],
            help=tr(
                "Manuell: schneller, kein Extra-Call. Automatisch: ein zusätzlicher API-Call pro Antwort.",
                "Manual: faster, no extra call. Automatic: one additional API call per answer.",
            ),
        )

        brands_input = ""
        if brand_mode == opt_manual:
            brands_input = st.text_input(
                tr("Brands (kommagetrennt)", "Brands (comma-separated)"),
                placeholder="Nike, Adidas, ASICS",
            )

        # Connection test — runs a minimal API call to catch auth/model errors early
        st.markdown("---")
        if st.button("🔌 " + tr("Verbindung testen", "Test connection"), disabled=not (api_key and model)):
            with st.spinner(tr("Teste Verbindung...", "Testing connection...")):
                ok, msg = test_connection(api_key, model, web_search)
            if ok:
                st.success(msg)
            else:
                st.error(tr(f"Fehler: {msg}", f"Error: {msg}"))
                st.info(
                    tr(
                        "Häufige Ursachen:\n"
                        "- API-Key falsch oder abgelaufen\n"
                        f"- Modell-Name '{model}' nicht verfügbar (Workspace-Einstellungen prüfen)\n"
                        f"- Falsche Region (aktuell: '{LANGDOCK_REGION}') — mit `LANGDOCK_REGION=us streamlit run app.py` wechseln",
                        "Common causes:\n"
                        "- API key wrong or expired\n"
                        f"- Model name '{model}' not available (check workspace settings)\n"
                        f"- Wrong region (currently: '{LANGDOCK_REGION}') — switch with `LANGDOCK_REGION=us streamlit run app.py`",
                    )
                )

    with col_info:
        st.subheader(tr("Hinweise", "Notes"))
        cost_extra_line = tr(
            "- Analyse: **1 zusätzlicher Call** (Claude Opus 4.8, gesamter Datensatz)",
            "- Analysis: **1 extra call** (Claude Opus 4.8, whole dataset)",
        )
        n_q_for_cost = (
            n_questions
            if question_mode == opt_gen
            else len([q for q in manual_questions_input.splitlines() if q.strip()])
        )
        st.markdown(tr(
            f"""
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
- **{n_q_for_cost} Fragen** × R Runs = **{n_q_for_cost} × R Calls**
{cost_extra_line}
- R (Runs pro Frage) konfigurierst du im nächsten Schritt
- Analyse lässt sich jederzeit stoppen — bisherige Ergebnisse bleiben erhalten
        """,
            f"""
**Test the connection** before you start — saves time on errors.

**Topic:** The more specific, the more relevant the generated questions.

**Model:** Currently `{model or '—'}`. Which models are available depends on your Langdock workspace — use "Test connection" if you hit errors.

**Region:** Currently `{LANGDOCK_REGION}`.
Change with:
```
LANGDOCK_REGION=us streamlit run app.py
```

**Brand detection:**
- *Manual*: Only the specified brands are tracked.
- *Automatic*: The model detects all brands in the text. One extra call per answer.

**Cost (approx.):**
- **{n_q_for_cost} questions** × R runs = **{n_q_for_cost} × R calls**
{cost_extra_line}
- R (runs per question) is configured in the next step
- The analysis can be stopped at any time — results so far are kept
        """,
        ))

    st.divider()

    ready = bool(api_key and model)
    if question_mode == opt_gen and not topic.strip():
        ready = False
        st.warning(tr(
            "Bitte ein Thema eingeben, oder auf 'Eigene Fragen eingeben' wechseln.",
            "Please enter a topic, or switch to 'Enter your own questions'.",
        ))
    if question_mode == opt_manual_q and not manual_questions_input.strip():
        ready = False
        st.warning(tr(
            "Bitte mindestens eine Frage eingeben, oder auf automatische Generierung wechseln.",
            "Please enter at least one question, or switch to auto-generate.",
        ))
    if brand_mode == opt_manual and not brands_input.strip():
        ready = False
        st.warning(tr(
            "Bitte mindestens eine Brand eingeben, oder auf automatische Erkennung wechseln.",
            "Please enter at least one brand, or switch to automatic detection.",
        ))

    btn_label = (
        tr("Weiter: Fragen generieren →", "Next: Generate questions →")
        if question_mode == opt_gen
        else tr("Weiter: Fragen prüfen →", "Next: Review questions →")
    )

    if st.button(btn_label, type="primary", disabled=not ready):
        brands = (
            [b.strip() for b in brands_input.split(",") if b.strip()]
            if brand_mode == opt_manual
            else []
        )
        st.session_state.config = {
            "api_key":     api_key,
            "model":       model,
            "topic":       topic,
            "n_questions": n_questions,
            "brand_mode":  "manual" if brand_mode == opt_manual else "auto",
            "brands":      brands,
            "web_search":  web_search,
        }
        if question_mode == opt_gen:
            with st.spinner(tr("Fragen werden generiert...", "Generating questions...")):
                questions, err = generate_questions(api_key, topic, n_questions, model)
            if questions:
                st.session_state.questions = questions
                st.session_state.step = 2
                st.rerun()
            else:
                st.error(tr(f"Fragen konnten nicht generiert werden: {err}", f"Could not generate questions: {err}"))
                st.info(tr(
                    "Verbindung testen (Button oben) um die genaue Ursache zu sehen.",
                    "Use 'Test connection' (button above) to see the exact cause.",
                ))
        else:
            st.session_state.questions = [
                q.strip() for q in manual_questions_input.splitlines() if q.strip()
            ]
            st.session_state.step = 2
            st.rerun()


# ===========================================================================
# UI — Step 2: Review questions
# One question per line in a text area — free to edit, delete, or add.
# ===========================================================================
def render_step2():
    st.title("📊 LLM Brand Visibility")
    st.progress(0.25, tr("Schritt 2 von 4 — Fragen prüfen", "Step 2 of 4 — Review questions"))
    render_tutorial(2)
    st.divider()

    st.subheader(tr("Fragen prüfen und anpassen", "Review and adjust questions"))
    st.caption(tr(
        "Jede Zeile ist eine Frage. Bearbeiten, löschen oder neue ergänzen. "
        "Leere Zeilen werden ignoriert — die tatsächliche Anzahl siehst du unten.",
        "Each line is one question. Edit, delete, or add new ones. "
        "Empty lines are ignored — the actual count is shown below.",
    ))

    questions_text = st.text_area(
        tr("Fragen (eine pro Zeile)", "Questions (one per line)"),
        value="\n".join(st.session_state.questions),
        height=450,
    )

    n_lines = len([q for q in questions_text.splitlines() if q.strip()])
    orig    = len(st.session_state.questions)
    delta   = n_lines - orig
    delta_str = (
        tr(f" ({'+' if delta >= 0 else ''}{delta} gegenüber generiert)",
           f" ({'+' if delta >= 0 else ''}{delta} vs. generated)")
        if delta != 0 else ""
    )
    st.caption(tr(
        f"**{n_lines} Fragen**{delta_str} — nur diese werden analysiert.",
        f"**{n_lines} questions**{delta_str} — only these will be analyzed.",
    ))

    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button(tr("← Zurück", "← Back")):
            st.session_state.step = 1
            st.rerun()
    with col2:
        if st.button(tr("Weiter: Runs konfigurieren →", "Next: Configure runs →"), type="primary"):
            cleaned = [q.strip() for q in questions_text.splitlines() if q.strip()]
            if not cleaned:
                st.error(tr("Mindestens eine Frage erforderlich.", "At least one question is required."))
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
    st.progress(0.5, tr("Schritt 3 von 4 — Runs konfigurieren", "Step 3 of 4 — Configure runs"))
    render_tutorial(3)
    st.divider()

    cfg  = st.session_state.config
    n_q  = len(st.session_state.questions)
    auto = cfg["brand_mode"] == "auto"

    st.subheader(tr("Runs konfigurieren", "Configure runs"))

    col_form, col_summary = st.columns([3, 2])

    with col_form:
        web_search = st.checkbox(
            tr("Websuche aktivieren", "Enable web search"),
            value=cfg.get("web_search", False),
            help=tr(
                "Echte Websuche über die Langdock Agent-API. Die Modell-Liste unten wechselt dann auf "
                "den Agent-API-Katalog (andere Modell-IDs).",
                "Real web search via the Langdock Agent API. The model list below switches to the Agent "
                "API's catalog (different model IDs) when enabled.",
            ),
        )

        st.markdown(f"**{tr('Modell', 'Model')}**")
        custom_label  = tr("Benutzerdefiniert...", "Custom...")
        current_model = cfg.get("model", "")
        api_key       = cfg.get("api_key", "")

        if web_search:
            if not api_key:
                st.info(tr(
                    "Kein API-Key in der Konfiguration gefunden — Modell-Liste kann nicht geladen werden.",
                    "No API key found in the configuration — cannot load the model list.",
                ))
                agent_models, agent_models_err = [], None
            else:
                agent_models, agent_models_err = list_agent_models(api_key)
            if agent_models_err:
                st.warning(tr(
                    f"Modell-Liste (Agent-API) konnte nicht geladen werden: {agent_models_err}",
                    f"Could not load the model list (Agent API): {agent_models_err}",
                ))
            if agent_models:
                model = st.selectbox(
                    tr("Modell (Agent-API)", "Model (Agent API)"),
                    options=agent_models,
                    index=agent_models.index(current_model) if current_model in agent_models else 0,
                )
            else:
                model = st.text_input(
                    tr("Modell-ID (Agent-API, manuell)", "Model ID (Agent API, manual)"),
                    value=current_model,
                    placeholder="z.B. claude-opus-4-7@default",
                )
        else:
            _full_options = [f"{p} — {m}" for p, models in AVAILABLE_MODELS.items() for m in models]
            _default_option = next((o for o in _full_options if o.endswith(f"— {current_model}")), custom_label)
            model_option = st.selectbox(
                tr("Modell für die Datensammlung", "Model for data collection"),
                options=_MODEL_OPTIONS,
                index=_MODEL_OPTIONS.index(_default_option) if _default_option in _MODEL_OPTIONS else len(_MODEL_OPTIONS) - 1,
                help=tr(
                    "Kann vom Modell aus Schritt 1 abweichen — nützlich, um dieselben Fragen mit einem "
                    "anderen Modell erneut zu sammeln.",
                    "Can differ from the model chosen in Step 1 — useful for re-collecting the same "
                    "questions with a different model.",
                ),
            )
            if model_option == custom_label:
                model = st.text_input(
                    tr("Modell-Name (benutzerdefiniert)", "Model name (custom)"),
                    value=current_model if _default_option == custom_label else "",
                )
            else:
                model = _model_id_from_option(model_option)

        runs = st.slider(tr("Wie oft soll jede Frage ausgeführt werden?", "How many times should each question run?"), 1, 100, 2)

        parallel_calls = st.slider(
            tr("Parallele API-Calls", "Parallel API calls"),
            min_value=1, max_value=10, value=2,
            help=tr(
                "Standard 2 bleibt sicher unter dem 60k-TPM-Limit (~32k geschätzt). Höher = schneller, aber "
                "größeres Risiko für 429-Rate-Limits.",
                "The default of 2 stays safely under the 60k TPM limit (~32k estimated). Higher = faster, but "
                "a greater risk of 429 rate limits.",
            ),
        )

        short_answer = st.toggle(
            tr("Kurzantwort-Modus", "Short-answer mode"),
            value=False,
            help=tr(
                "LLM gibt nur Marken + einen Begründungssatz zurück. Viel weniger Tokens, schneller, günstiger.",
                "The LLM returns only brands + one sentence of reasoning. Far fewer tokens, faster, cheaper.",
            ),
        )

        if web_search:
            # The Agent API doesn't accept a max_tokens parameter, so the slider/TPM
            # gauge below don't apply — hide them to avoid implying a control that
            # has no effect on this endpoint.
            max_tokens_val = MAX_TOKENS
            st.caption(tr(
                "ℹ️ Token-Limit pro Antwort ist bei aktiver Websuche (Agent-API) nicht einstellbar.",
                "ℹ️ Per-answer token limit is not adjustable while web search (Agent API) is active.",
            ))
        else:
            max_tokens_val = st.slider(
                tr("Max. Tokens pro Antwort", "Max tokens per answer"),
                min_value=500, max_value=16000, value=MAX_TOKENS, step=500,
                help=tr(
                    "Bei Reasoning-Modellen (gpt-5-mini) fließen interne Denkschritte ins Budget ein — "
                    "mindestens 8000 einplanen. gpt-5-mini-eu: 60.000 Tokens/Minute Limit.",
                    "For reasoning models (gpt-5-mini), internal thinking steps count against the budget — "
                    "plan for at least 8000. gpt-5-mini-eu: 60,000 tokens/minute limit.",
                ),
            )

            # Live TPM estimate — assumes ~30s avg response time for reasoning models
            assumed_resp_s = 30
            calls_per_min  = parallel_calls * (60 / assumed_resp_s)
            tpm_estimate   = int(calls_per_min * max_tokens_val)
            tpm_pct        = tpm_estimate / 60000 * 100
            tpm_color      = "🟢" if tpm_pct < 70 else ("🟡" if tpm_pct < 100 else "🔴")
            st.caption(tr(
                f"{tpm_color} Geschätzte Token-Last: **{tpm_estimate:,} Tokens/min** "
                f"({tpm_pct:.0f}% des 60k-Limits bei gpt-5-mini) — "
                f"Annahme: {assumed_resp_s}s Ø Antwortzeit, {parallel_calls} parallel",
                f"{tpm_color} Estimated token load: **{tpm_estimate:,} tokens/min** "
                f"({tpm_pct:.0f}% of the 60k limit for gpt-5-mini) — "
                f"assuming {assumed_resp_s}s avg. response time, {parallel_calls} parallel",
            ))

        delay = st.slider(
            tr("Pause zwischen API-Calls (Sekunden)", "Pause between API calls (seconds)"),
            min_value=0.0, max_value=5.0, value=0.5, step=0.5,
            help=tr(
                "Pause nach jedem Call (sequenziell) bzw. nach jedem abgeschlossenen Batch (parallel).",
                "Pause after each call (sequential) or after each completed batch (parallel).",
            ),
            disabled=parallel_calls > 1,
        )

        # Call estimate. Phase 1 = one collection call per question×run. Phase 2 is a
        # SINGLE whole-dataset analysis call (Claude Opus 4.8), regardless of brand mode.
        collection_calls = n_q * runs
        st.success(tr(
            f"Sammlung: **{collection_calls} Calls** ({n_q} Fragen × {runs} Runs). "
            f"Analyse: **1 Call** ({ANALYSIS_MODEL}, gesamter Datensatz).",
            f"Collection: **{collection_calls} calls** ({n_q} questions × {runs} runs). "
            f"Analysis: **1 call** ({ANALYSIS_MODEL}, whole dataset).",
        ))
        if not auto and cfg.get("brands"):
            st.caption(tr(
                f"Vorgegebene Brands: {', '.join(cfg['brands'])}",
                f"Specified brands: {', '.join(cfg['brands'])}",
            ))

        if web_search:
            st.info(tr(
                "Websuche aktiv — läuft über die Langdock Agent-API (separater Endpunkt, kein Token-Reporting, "
                "kann länger dauern).",
                "Web search active — runs via the Langdock Agent API (separate endpoint, no token reporting, "
                "can take longer).",
            ))

        st.caption(tr(
            "Du kannst die Analyse jederzeit stoppen — bisherige Ergebnisse werden trotzdem angezeigt.",
            "You can stop the analysis at any time — results so far will still be shown.",
        ))

    with col_summary:
        st.markdown(f"**{tr('Zusammenfassung', 'Summary')}**")
        st.metric(tr("Fragen", "Questions"), n_q)
        st.metric(tr("Runs pro Frage", "Runs per question"), runs)
        # Phase 1 collection calls + one whole-dataset analysis call.
        st.metric(tr("Sammel-Calls", "Collection calls"), n_q * runs)
        st.metric(tr("Analyse-Calls", "Analysis calls"), 1)
        st.metric(tr("Modell", "Model"), model)

    st.divider()

    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button(tr("← Zurück", "← Back")):
            st.session_state.step = 2
            st.rerun()
    with col2:
        if st.button("🚀 " + tr("Daten sammeln", "Collect data"), type="primary", disabled=not model):
            st.session_state.config["model"]          = model
            st.session_state.config["web_search"]     = web_search
            st.session_state.config["runs"]           = runs
            st.session_state.config["delay"]          = delay
            st.session_state.config["parallel_calls"] = parallel_calls
            st.session_state.config["max_tokens"]     = max_tokens_val
            st.session_state.config["short_answer"]   = short_answer
            st.session_state.stop_requested           = False
            _run_phase1()


def _render_live_metrics(box, completed: int, total: int, elapsed: float, call_times: list[float]):
    """Renders a live 5-column timing dashboard into an st.empty() container."""
    rate_s    = completed / elapsed if elapsed > 0 else 0
    eta       = (total - completed) / rate_s if rate_s > 0 and completed < total else 0
    avg_t     = sum(call_times) / len(call_times) if call_times else 0
    min_t     = min(call_times) if call_times else 0
    max_t     = max(call_times) if call_times else 0
    with box.container():
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric(tr("Fortschritt", "Progress"), f"{completed} / {total}",
                  help=tr("Abgeschlossene API-Calls / Calls insgesamt in dieser Phase.",
                          "Completed API calls / total calls in this phase."))
        c2.metric(tr("Verstrichene Zeit", "Elapsed time"), f"{elapsed:.0f}s",
                  help=tr("Gesamtdauer seit Start dieser Phase.", "Total duration since this phase started."))
        c3.metric(tr("Verbleibend (ETA)", "Remaining (ETA)"),
                  f"~{eta:.0f}s" if completed < total else tr("✓ fertig", "✓ done"),
                  help=tr("Geschätzte Restdauer basierend auf der bisherigen Durchschnittsgeschwindigkeit.",
                          "Estimated remaining time based on the average speed so far."))
        c4.metric(tr("Geschwindigkeit", "Speed"), f"{rate_s * 60:.0f} / min",
                  help=tr("Abgeschlossene Calls pro Minute (gleitend über diese Phase).",
                          "Completed calls per minute (rolling over this phase)."))
        c5.metric(tr("Ø Antwortzeit", "Avg. response time"), f"{avg_t:.1f}s",
                  f"min {min_t:.1f}s  max {max_t:.1f}s",
                  help=tr("Durchschnittliche Zeit pro API-Call. Delta zeigt schnellsten und langsamsten Call.",
                          "Average time per API call. Delta shows the fastest and slowest call."))


def _run_phase1():
    cfg        = st.session_state.config
    questions  = st.session_state.questions
    api_key    = cfg["api_key"]
    model      = cfg["model"]
    runs       = cfg["runs"]
    delay      = cfg["delay"]
    parallel     = cfg.get("parallel_calls", 1)
    max_tokens   = cfg.get("max_tokens", MAX_TOKENS)
    web_search   = cfg.get("web_search", False)
    short_answer = cfg.get("short_answer", False)
    lang         = st.session_state.get("lang", "de")  # captured here — worker threads can't read session_state

    total        = len(questions) * runs
    raw_answers  = []
    errors       = []
    failed_tasks: list[tuple[int, str, int]] = []  # (question_index, question, run_num) to retry once
    p1_timings: list[dict] = []

    def _record_answer(question: str, run_num: int, answer: str, usage: dict):
        raw_answers.append({
            "question":        question,
            "run":             run_num + 1,
            "answer":          answer,
            "model":           model,
            "tokens_in":       usage.get("prompt_tokens", 0),
            "tokens_out":      usage.get("completion_tokens", 0),
            # web-search evidence (only populated when web_search routed via the Agent API)
            "web_search_used": usage.get("web_search_used", False),
            "sources":         usage.get("sources", []),
            "citation_count":  usage.get("citation_count", 0),
        })

    # ------------------------------------------------------------------
    # Phase 1 — Collect answers (parallel)
    # ------------------------------------------------------------------
    st.subheader(tr("Phase 1 — Antworten sammeln", "Phase 1 — Collecting answers"))
    if st.button("⏹ " + tr("Stoppen", "Stop"), type="secondary", key="stop_phase1"):
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
            executor.submit(ask_question, api_key, q, model, lang, web_search, max_tokens, short_answer): (i, run_num, q, time.time())
            for i, q, run_num in all_tasks
        }

        completed = 0
        for future in as_completed(future_map):
            i, run_num, question, t_sub = future_map[future]
            completed += 1

            if st.session_state.get("stop_requested", False):
                status_line.warning(tr(
                    f"⏹ Gestoppt nach {len(raw_answers)} von {total} Antworten. "
                    "Analyse wird mit bisherigen Antworten fortgesetzt.",
                    f"⏹ Stopped after {len(raw_answers)} of {total} answers. "
                    "Analysis continues with the answers collected so far.",
                ))
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
                # Collect for a single retry pass at the end of the phase rather than
                # giving up now — an exhausted-retry failure here otherwise leaves this
                # question with fewer runs than the others, biasing share-of-voice.
                failed_tasks.append((i, question, run_num))
                error_box.warning(tr(
                    f"⚠️ Frage {i+1}, Run {run_num+1} fehlgeschlagen: {err} — wird am Ende erneut versucht.",
                    f"⚠️ Question {i+1}, Run {run_num+1} failed: {err} — will retry at the end.",
                ))
                log.warning("No answer — Frage %d, Run %d: %s", i+1, run_num+1, err)
            else:
                error_box.empty()
                _record_answer(question, run_num, answer, usage)
                log.info("Phase1 OK — Frage %d Run %d | len=%d | tok_in=%d tok_out=%d | web_search=%s | sources=%d | %.2fs",
                         i+1, run_num+1, len(answer),
                         usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0),
                         usage.get("web_search_used", False), len(usage.get("sources", [])), call_elapsed)

            q_short = question[:70] + ("…" if len(question) > 70 else "")
            status_line.caption(tr(
                f"Zuletzt: Frage {i+1} · Run {run_num+1} · {call_elapsed:.1f}s · {q_short}",
                f"Latest: Question {i+1} · Run {run_num+1} · {call_elapsed:.1f}s · {q_short}",
            ))

            if parallel == 1:
                time.sleep(delay)

    # ------------------------------------------------------------------
    # Retry pass — one more attempt for calls that failed above. Only runs if the
    # user didn't stop. Anything still failing after this is recorded as an error.
    # ------------------------------------------------------------------
    if failed_tasks and not st.session_state.get("stop_requested", False):
        status_line.caption(tr(
            f"Wiederhole {len(failed_tasks)} fehlgeschlagene Calls …",
            f"Retrying {len(failed_tasks)} failed calls …",
        ))
        with ThreadPoolExecutor(max_workers=parallel) as retry_executor:
            retry_map = {
                retry_executor.submit(ask_question, api_key, q, model, lang, web_search, max_tokens, short_answer):
                    (i, run_num, q, time.time())
                for i, q, run_num in failed_tasks
            }
            for future in as_completed(retry_map):
                i, run_num, question, t_sub = retry_map[future]
                if st.session_state.get("stop_requested", False):
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
                if not answer:
                    errors.append(tr(
                        f"Frage {i+1}, Run {run_num+1}: {err}",
                        f"Question {i+1}, Run {run_num+1}: {err}",
                    ))
                    log.warning("Retry failed — Frage %d, Run %d: %s", i+1, run_num+1, err)
                else:
                    _record_answer(question, run_num, answer, usage)
                    log.info("Phase1 retry OK — Frage %d Run %d | len=%d | %.2fs", i+1, run_num+1, len(answer), call_elapsed)
        _render_live_metrics(metrics_box, len(call_times), len(call_times), time.time() - t_phase1, call_times)

    phase1_elapsed = time.time() - t_phase1
    bar1.progress(1.0)
    _render_live_metrics(metrics_box, completed, total, phase1_elapsed, call_times)
    status_line.caption(tr(
        f"Phase 1 abgeschlossen — {len(raw_answers)}/{total} Antworten in {phase1_elapsed:.1f}s",
        f"Phase 1 complete — {len(raw_answers)}/{total} answers in {phase1_elapsed:.1f}s",
    ))

    st.session_state.raw_answers    = raw_answers
    st.session_state.phase1_errors  = errors
    st.session_state.stop_requested = False
    st.session_state.timing = {
        "phase1_s": round(phase1_elapsed, 2),
        "p1_calls": p1_timings,
        "phase2_s": 0.0,
        "p2_calls": [],
    }

    st.session_state.step = 4
    time.sleep(1)
    st.rerun()


def _normalize_brand_key(name: str) -> str:
    """Matching key: lowercase, strip trademark glyphs, drop legal-form suffixes
    (Inc, GmbH, Ltd, …) and any non-alphanumerics. So 'Nike Inc.', 'nike' and 'NIKE®'
    all collapse to 'nike'."""
    key = (name or "").lower()
    key = re.sub(r"[®™©]", "", key)
    key = re.sub(r"\b(inc|corp|corporation|gmbh|ltd|limited|llc|ag|co|company|group|the)\b", " ", key)
    key = re.sub(r"[^a-z0-9]+", "", key)
    return key


def _run_brand_analysis(raw_answers: list[dict], prior_errors: list | None = None):
    """
    Phase 2: brand extraction + sentiment for the whole dataset, batched across one or
    more calls to a single strong model (ANALYSIS_MODEL / Claude Opus 4.8). The model
    returns flat JSON arrays that we regroup per answer.
    """
    cfg      = st.session_state.config
    api_key  = cfg["api_key"]
    model    = cfg["model"]  # collection model — kept per-answer for the results table
    brands   = cfg.get("brands", [])
    lang     = st.session_state.get("lang", "de")
    errors   = list(prior_errors or [])

    n_answers = len(raw_answers)

    # Map returned brand names to the canonical user-specified spelling. Matching is
    # fuzzy (normalized key + substring), so "Nike Inc." still maps to a listed "Nike".
    brand_lookup: dict[str, str] = {}
    for b in brands:
        k = _normalize_brand_key(b)
        if k:
            brand_lookup[k] = b.strip()
    unlisted_seen: dict[str, str] = {}  # brands the model found that aren't in the user's list

    def _normalize_brands(brand_list: list[dict]) -> list[dict]:
        out, seen = [], set()
        for b in brand_list:
            raw_name = (b.get("brand") or "").strip()
            if not raw_name:
                continue
            if brand_lookup:
                rk = _normalize_brand_key(raw_name)
                canonical = brand_lookup.get(rk)
                if canonical is None:
                    # Fuzzy: substring containment either direction, length-guarded to
                    # avoid matching on a stray couple of characters.
                    for ck, cval in brand_lookup.items():
                        if len(ck) >= 3 and (ck in rk or (len(rk) >= 3 and rk in ck)):
                            canonical = cval
                            break
                if canonical is None:
                    # Not one of the user's brands — keep it aside to surface later
                    # instead of silently dropping it.
                    if rk:
                        unlisted_seen[rk] = raw_name
                    continue
                b = {**b, "brand": canonical}
            key = _normalize_brand_key(b["brand"])
            if key in seen:
                continue  # dedupe within a single answer
            seen.add(key)
            out.append(b)
        return out

    st.subheader(tr("Phase 2 — Brand & Sentiment Analyse", "Phase 2 — Brand & sentiment analysis"))
    st.caption(tr(
        f"{n_answers} Antworten in einem Durchlauf mit `{ANALYSIS_MODEL}` analysieren.",
        f"Analyzing {n_answers} answers in a single pass with `{ANALYSIS_MODEL}`.",
    ))

    # Analysis is batched, so a large dataset no longer risks a single truncated call.
    # Just tell the user how many batches (= calls) to expect.
    n_batches = len(_make_analysis_batches(raw_answers))
    if n_batches > 1:
        st.info(tr(
            f"Großer Datensatz ({n_answers} Antworten) — Analyse läuft in {n_batches} Batches "
            "und wird anschließend zusammengeführt.",
            f"Large dataset ({n_answers} answers) — analysis runs in {n_batches} batches "
            "and is then merged.",
        ))

    bar2        = st.progress(0.0)
    status_line = st.empty()

    t_phase2 = time.time()
    status_line.caption(tr(
        f"Analyse läuft mit {ANALYSIS_MODEL} …",
        f"Analysis running with {ANALYSIS_MODEL} …",
    ))

    by_index, summary, usage, analysis_err = analyze_dataset(api_key, raw_answers, brands, lang)
    phase2_elapsed = time.time() - t_phase2
    bar2.progress(1.0)

    if analysis_err:
        errors.append(tr(f"Analyse fehlgeschlagen: {analysis_err}", f"Analysis failed: {analysis_err}"))

    analysis_tokens = usage.get("completion_tokens", 0)

    # Reconstruct results in original answer order, attaching each answer's brands.
    results = []
    for i, raw in enumerate(raw_answers):
        merged = _normalize_brands(by_index.get(i, []))
        results.append({
            "question":        raw["question"],
            "run":             raw["run"],
            "answer":          raw["answer"],
            "model":           raw.get("model", model),
            "tokens_in":       raw.get("tokens_in", 0),
            "tokens_out":      raw.get("tokens_out", 0),
            "tokens_analysis": analysis_tokens,
            "brands_found":    merged,
        })
    # In manual mode, remember brands the model found that aren't on the user's list,
    # so the results view can surface unlisted competitors instead of hiding them.
    st.session_state.unlisted_brands = sorted(unlisted_seen.values(), key=str.lower) if brands else []
    log.info(
        "Phase2 OK — %d answers analyzed via %s (batched) | %.2fs | brands total: %d | unlisted: %d",
        n_answers, ANALYSIS_MODEL, phase2_elapsed,
        sum(len(r["brands_found"]) for r in results), len(unlisted_seen),
    )

    status_line.caption(tr(
        f"Phase 2 abgeschlossen — {n_answers} Antworten in {phase2_elapsed:.1f}s"
        + (f", {len(errors)} Fehler" if errors else ""),
        f"Phase 2 complete — {n_answers} answers in {phase2_elapsed:.1f}s"
        + (f", {len(errors)} errors" if errors else ""),
    ))

    # Persist Phase 2 timing into the session state dict written by Phase 1.
    # p2_calls holds the single dataset call so the Runtime tab still has data.
    timing = st.session_state.get("timing", {})
    timing["phase2_s"] = round(phase2_elapsed, 2)
    timing["p2_calls"] = [{"question": tr("Gesamter Datensatz", "Whole dataset"),
                           "elapsed_s": round(phase2_elapsed, 3)}]
    st.session_state.timing = timing

    if errors:
        with st.expander(tr(f"⚠️ {len(errors)} Fehler", f"⚠️ {len(errors)} errors")):
            for e in errors:
                st.markdown(f"- {e}")

    st.session_state.results          = results
    st.session_state.analysis_summary = summary
    st.session_state.step             = 5
    time.sleep(1)
    st.rerun()


# ===========================================================================
# UI — Step 4: Raw data checkpoint
# Shown right after Phase 1 (data collection) and before Phase 2 (brand/
# sentiment analysis). Lets the user inspect and export the raw answers
# before spending extra API calls on analysis, or restart from scratch.
# ===========================================================================
def render_step4():
    st.title("📊 LLM Brand Visibility")
    st.progress(0.75, tr("Schritt 4 von 4 — Rohdaten prüfen", "Step 4 of 4 — Review raw data"))
    render_tutorial(4)
    st.divider()

    raw_answers = st.session_state.get("raw_answers", [])
    errors      = st.session_state.get("phase1_errors", [])
    cfg         = st.session_state.get("config", {})

    st.subheader(tr("Gesammelte Antworten", "Collected answers"))
    st.caption(tr(
        f"{len(raw_answers)} Antworten erfolgreich gesammelt"
        + (f", {len(errors)} fehlgeschlagen" if errors else "")
        + ". Noch keine Marken-/Sentiment-Analyse durchgeführt — dafür fallen weitere API-Calls an.",
        f"{len(raw_answers)} answers collected successfully"
        + (f", {len(errors)} failed" if errors else "")
        + ". No brand/sentiment analysis has run yet — that will use additional API calls.",
    ))

    # --- Web-search verification -----------------------------------------
    # When web search was enabled, show how many answers actually invoked the
    # search tool. Evidence = a search tool event, cited source URLs, and/or
    # inline 【toolu_…】 citation markers — not whether the answer merely *sounds*
    # current. Source URLs aren't always exposed by the API, so we also count
    # citations as proof and report both.
    if cfg.get("web_search") and raw_answers:
        n_total     = len(raw_answers)
        n_searched  = sum(1 for r in raw_answers if r.get("web_search_used"))
        n_sources   = sum(len(r.get("sources", [])) for r in raw_answers)
        n_citations = sum(r.get("citation_count", 0) for r in raw_answers)
        evidence = tr(
            f"{n_sources} zitierte Quellen, {n_citations} Zitat-Marker",
            f"{n_sources} cited sources, {n_citations} citation markers",
        )
        if n_searched == n_total:
            st.success(tr(
                f"🔍 Websuche bestätigt: alle {n_total} Antworten haben das Such-Tool genutzt ({evidence}).",
                f"🔍 Web search confirmed: all {n_total} answers used the search tool ({evidence}).",
            ))
        elif n_searched > 0:
            st.warning(tr(
                f"🔍 Websuche teilweise genutzt: {n_searched} von {n_total} Antworten haben das Such-Tool "
                f"aufgerufen ({evidence}). Die übrigen wurden aus dem Modellwissen beantwortet.",
                f"🔍 Web search partially used: {n_searched} of {n_total} answers called the search tool "
                f"({evidence}). The rest were answered from model knowledge.",
            ))
        else:
            st.error(tr(
                "🔍 Keine Antwort hat das Such-Tool nachweislich genutzt (keine Tool-Calls, Quellen oder "
                "Zitat-Marker). Das Modell hat vermutlich aus seinem Trainingswissen geantwortet.",
                "🔍 No answer verifiably used the search tool (no tool calls, sources, or citation markers). "
                "The model most likely answered from its training knowledge.",
            ))

    if errors:
        with st.expander(tr(f"⚠️ {len(errors)} Fehler", f"⚠️ {len(errors)} errors")):
            for e in errors:
                st.markdown(f"- {e}")

    if not raw_answers:
        st.warning(tr(
            "Keine Antworten gesammelt. Alle API-Calls sind fehlgeschlagen.",
            "No answers collected. All API calls failed.",
        ))
        if st.button(tr("🔄 Neuen Prozess starten", "🔄 Restart process")):
            reset_process()
        return

    st.divider()

    # --- Raw data table -------------------------------------------------
    show_search = bool(cfg.get("web_search"))
    col_frage   = tr("Frage", "Question")
    col_modell  = tr("Modell", "Model")
    col_suche   = tr("🔍 Suche", "🔍 Search")
    col_antwort = tr("Antwort", "Answer")
    col_tok_in  = tr("Tok. Input", "Tok. input")
    col_tok_out = tr("Tok. Antwort", "Tok. answer")
    raw_rows = []
    for r in raw_answers:
        row = {
            col_frage:   r["question"][:80] + ("..." if len(r["question"]) > 80 else ""),
            "Run":       r["run"],
            col_modell:  r.get("model", ""),
        }
        if show_search:
            # Prefer a source count; fall back to the citation-marker count as evidence.
            n_evidence = len(r.get("sources", [])) or r.get("citation_count", 0)
            row[col_suche] = ("✅ " + (f"{n_evidence}" if n_evidence else "")) if r.get("web_search_used") else "—"
        row[col_antwort] = (r["answer"][:200] + "...") if len(r["answer"]) > 200 else r["answer"]
        row[col_tok_in]  = r.get("tokens_in", 0)
        row[col_tok_out] = r.get("tokens_out", 0)
        raw_rows.append(row)
    st.dataframe(pd.DataFrame(raw_rows), width="stretch")
    if show_search:
        st.caption(tr(
            "🔍 Suche: ✅ = Such-Tool nachweislich genutzt (Zahl = Quellen bzw. Zitat-Marker), — = nicht genutzt.",
            "🔍 Search: ✅ = search tool verifiably used (number = sources or citation markers), — = not used.",
        ))

    # --- Full Q&A, one expander per answer -------------------------------
    with st.expander(tr("Alle Antworten (vollständig)", "All answers (full text)")):
        q_order = list(dict.fromkeys(r["question"] for r in raw_answers))
        q_num   = {q: i + 1 for i, q in enumerate(q_order)}
        for r in raw_answers:
            label = f"Q{q_num[r['question']]} · Run {r['run']} · {r['question'][:65]}{'…' if len(r['question']) > 65 else ''}"
            with st.expander(label):
                st.markdown(f"**{tr('Vollständige Frage', 'Full question')}:** {r['question']}")
                st.divider()
                st.markdown(r["answer"])
                sources = r.get("sources", [])
                if sources:
                    st.divider()
                    st.markdown(f"**{tr('🔍 Verwendete Quellen', '🔍 Sources used')}** ({len(sources)}):")
                    for s in sources:
                        title = s.get("title") or s.get("url")
                        st.markdown(f"- [{title}]({s['url']})")
                elif show_search and r.get("web_search_used"):
                    # Search ran (tool event or 【toolu_…】 citations) but the API
                    # didn't expose the underlying source URLs for this answer.
                    st.divider()
                    st.caption(tr(
                        f"🔍 Websuche genutzt ({r.get('citation_count', 0)} Zitate), aber die API hat keine "
                        "Quellen-URLs mitgeliefert.",
                        f"🔍 Web search used ({r.get('citation_count', 0)} citations), but the API returned no "
                        "source URLs for this answer.",
                    ))
                elif show_search:
                    st.caption(tr(
                        "🔍 Diese Antwort hat das Such-Tool nicht nachweislich genutzt.",
                        "🔍 This answer did not verifiably use the search tool.",
                    ))

    st.divider()

    # --- Export raw data --------------------------------------------------
    st.subheader("Export")
    col1, col2 = st.columns(2)
    with col1:
        raw_csv = pd.DataFrame(raw_answers).to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 " + tr("Rohdaten als CSV", "Raw data as CSV"),
            raw_csv,
            "brand_visibility_raw_answers.csv",
            "text/csv",
        )
    with col2:
        raw_json = json.dumps(raw_answers, ensure_ascii=False, indent=2)
        st.download_button(
            "📥 " + tr("Rohdaten als JSON", "Raw data as JSON"),
            raw_json.encode("utf-8"),
            "brand_visibility_raw_answers.json",
            "application/json",
        )

    st.divider()

    col_back, col_a, col_b = st.columns(3)
    with col_back:
        if st.button("← " + tr("Einstellungen ändern", "Change settings")):
            st.session_state.step = 3
            st.rerun()
    with col_a:
        if st.button("🚀 " + tr("Weiter: Analyse starten", "Continue: Start analysis"), type="primary"):
            _run_brand_analysis(raw_answers, errors)
    with col_b:
        if st.button("🔄 " + tr("Neuen Prozess starten", "Restart process")):
            reset_process()
    st.caption(tr(
        "„Einstellungen ändern“ nutzt dieselben Fragen, lässt dich aber z.B. Modell oder Runs anpassen. "
        "Beim erneuten Sammeln werden die aktuellen Rohdaten ersetzt.",
        "\"Change settings\" reuses the same questions but lets you adjust e.g. the model or number of runs. "
        "Collecting again replaces the current raw data.",
    ))


# ===========================================================================
# UI — Step 5: Results
# KPI metrics, charts, context details, raw data table, export options.
# ===========================================================================
def render_step5():
    st.title(tr("📊 LLM Brand Visibility — Ergebnisse", "📊 LLM Brand Visibility — Results"))
    render_tutorial(5)
    st.divider()

    results = st.session_state.results
    cfg     = st.session_state.config

    # --- Configuration summary (from Step 1 + Step 3) ----------------------
    with st.expander(tr("Analyse-Konfiguration", "Analysis configuration"), expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        topic_display = (cfg.get("topic") or "").strip() or "—"
        c1.markdown(f"**{tr('Thema', 'Topic')}**  \n{topic_display}")
        c2.markdown(
            f"**{tr('Modell', 'Model')}**  \n"
            f"`{cfg.get('model', '—')}`  \n"
            + tr(f"Analyse: `{ANALYSIS_MODEL}`", f"Analysis: `{ANALYSIS_MODEL}`")
        )
        runs_val     = cfg.get("runs", "—")
        parallel_val = cfg.get("parallel_calls", 1)
        n_q          = len(st.session_state.get("questions", []))
        c3.markdown(tr(
            f"**Fragen / Runs**  \n{n_q} Fragen × {runs_val} Runs",
            f"**Questions / Runs**  \n{n_q} questions × {runs_val} runs",
        ))
        # max_tokens is meaningless when web search (Agent API) was used — omit it then.
        tokens_part = "" if cfg.get("web_search") else tr(
            f" | max {cfg.get('max_tokens', MAX_TOKENS)} Tokens",
            f" | max {cfg.get('max_tokens', MAX_TOKENS)} tokens",
        )
        c4.markdown(tr(
            f"**Parallel / Delay / Tokens**  \n"
            f"{parallel_val} parallel | {cfg.get('delay', 0.5)}s Pause{tokens_part}",
            f"**Parallel / Delay / Tokens**  \n"
            f"{parallel_val} parallel | {cfg.get('delay', 0.5)}s pause{tokens_part}",
        ))

        brand_mode = cfg.get("brand_mode", "auto")
        if brand_mode == "manual":
            brands_list = cfg.get("brands", [])
            st.markdown(
                tr("**Brands (manuell):** ", "**Brands (manual):** ") +
                (", ".join(f"`{b}`" for b in brands_list) if brands_list else "—")
            )
            unlisted = st.session_state.get("unlisted_brands", [])
            if unlisted:
                shown = ", ".join(f"`{b}`" for b in unlisted[:25])
                more  = tr(f" … (+{len(unlisted) - 25} weitere)", f" … (+{len(unlisted) - 25} more)") if len(unlisted) > 25 else ""
                st.markdown(tr(
                    "**Ebenfalls genannt (nicht in deiner Liste):** ",
                    "**Also mentioned (not in your list):** ",
                ) + shown + more)
                st.caption(tr(
                    "Diese Marken kamen in den Antworten vor, werden aber nicht in den Charts getrackt.",
                    "These brands appeared in the answers but are not tracked in the charts.",
                ))
        else:
            st.markdown(tr(
                "**Brand-Erkennung:** Automatisch (aus Antworten extrahiert)",
                "**Brand detection:** Automatic (extracted from answers)",
            ))

        if cfg.get("web_search"):
            st.markdown(tr("**Websuche:** Aktiv (Agent-API)", "**Web search:** Active (Agent API)"))

    st.divider()

    if not results:
        st.warning(tr(
            "Keine auswertbaren Ergebnisse. Alle API-Calls sind fehlgeschlagen.",
            "No usable results. All API calls failed.",
        ))
        if st.button(tr("← Zurück zur Konfiguration", "← Back to configuration")):
            st.session_state.step = 1
            st.rerun()
        return

    df = build_analysis(results)

    # --- Confidence filter -------------------------------------------------
    # Every chart/table below is driven off `df`, so filtering here narrows the
    # whole view. Applied before any KPI is computed. Mentions with an unknown
    # confidence are always kept (older runs, or a model that omitted the field).
    if not df.empty and "confidence" in df.columns:
        present = [c for c in ("high", "medium", "low") if c in set(df["confidence"])]
        if present:
            selected_conf = st.multiselect(
                tr("Konfidenz-Filter", "Confidence filter"),
                options=present,
                default=present,
                help=tr(
                    "Nur Marken-Nennungen mit der gewählten Analyse-Konfidenz anzeigen. "
                    "Nennungen ohne Konfidenz-Angabe bleiben immer sichtbar.",
                    "Show only brand mentions with the selected analysis confidence. "
                    "Mentions without a confidence value always stay visible.",
                ),
            )
            allowed = set(selected_conf) | {"", None}
            df = df[df["confidence"].isin(allowed)]
            if selected_conf and len(selected_conf) < len(present):
                st.caption(tr(
                    f"Gefiltert auf Konfidenz: {', '.join(selected_conf)}.",
                    f"Filtered to confidence: {', '.join(selected_conf)}.",
                ))

    # --- Overview ----------------------------------------------------------
    # High-level KPIs (responses / questions / brands / leader), the model's
    # short textual analysis, and a mention-comparison chart.
    st.subheader(tr("Überblick", "Overview"))

    total_responses = len(results)
    total_questions = len({r["question"] for r in results})
    n_brands        = int(df["brand"].nunique()) if not df.empty else 0

    # Answer coverage per brand = number of answers that mention the brand (brands are
    # deduped within an answer during analysis, so summing "mentions" gives coverage).
    coverage = (
        df.groupby("brand")["mentions"].sum().sort_values(ascending=False)
        if not df.empty else pd.Series(dtype="int64")
    )
    top_brand   = coverage.index[0] if not coverage.empty else "—"
    top_cover   = int(coverage.iloc[0]) if not coverage.empty else 0
    top_pct     = round(top_cover / total_responses * 100, 1) if total_responses else 0.0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(tr("Antworten", "Responses"), total_responses,
              help=tr("Erfolgreich gesammelte Antworten (Fragen × Runs).",
                      "Successfully collected answers (questions × runs)."))
    k2.metric(tr("Fragen", "Questions"), total_questions,
              help=tr("Eindeutige Fragestellungen im Datensatz.", "Distinct questions in the dataset."))
    k3.metric(tr("Erkannte Marken", "Brands detected"), n_brands)
    k4.metric(tr("Top-Marke", "Top brand"), top_brand if top_brand != "—" else "—",
              f"{top_cover}/{total_responses} ({top_pct}%)" if top_cover else None,
              help=tr("Marke mit der höchsten Antwortabdeckung.", "Brand with the highest answer coverage."))

    # --- Brief textual analysis (from the single Opus analysis call) --------
    summary = st.session_state.get("analysis_summary", "")
    if summary:
        st.markdown(f"**{tr('Kurzanalyse', 'Brief analysis')}**")
        st.info(summary)

    # --- Mention comparison ------------------------------------------------
    if not coverage.empty:
        st.markdown(f"**{tr('Nennungen im Vergleich', 'Mentions compared')}**")
        top_n   = coverage.head(15).sort_values(ascending=True)
        comp_df = pd.DataFrame({
            "brand":    top_n.index,
            "coverage": top_n.values,
            "pct":      (top_n.values / total_responses * 100) if total_responses else top_n.values,
        })
        fig_cmp = px.bar(
            comp_df,
            x="coverage",
            y="brand",
            orientation="h",
            text=[f"{c} ({p:.0f}%)" for c, p in zip(comp_df["coverage"], comp_df["pct"])],
            labels={
                "coverage": tr("Antworten mit mindestens einer Nennung", "Answers with at least one mention"),
                "brand": "",
            },
            color_discrete_sequence=["#3b6ea5"],
        )
        fig_cmp.update_traces(textposition="outside")
        fig_cmp.update_layout(
            showlegend=False,
            xaxis_range=[0, max(total_responses, int(coverage.max())) * 1.15],
            height=max(220, 30 * len(comp_df) + 90),
            margin={"l": 10, "r": 10, "t": 10, "b": 10},
        )
        st.plotly_chart(fig_cmp, width="stretch")
        st.caption(tr(
            f"Antwortabdeckung je Marke (max. {total_responses}). "
            "Prozent = Anteil aller Antworten mit mindestens einer Nennung.",
            f"Answer coverage per brand (max {total_responses}). "
            "Percent = share of all answers with at least one mention.",
        ))

    st.divider()

    # Re-analyze button — reruns Phase 2 on stored raw answers without re-querying
    raw_answers = st.session_state.get("raw_answers", [])
    if raw_answers:
        with st.expander("🔄 " + tr(
            "Sentiment neu analysieren (ohne neue API-Calls für Antworten)",
            "Re-analyze sentiment (without new API calls for answers)",
        )):
            st.caption(tr(
                f"{len(raw_answers)} Antworten gespeichert. "
                "Brand-Extraktion und Sentiment werden neu berechnet — Antworten werden nicht erneut abgefragt.",
                f"{len(raw_answers)} answers stored. "
                "Brand extraction and sentiment will be recomputed — answers are not queried again.",
            ))
            if st.button(tr("Analyse neu starten", "Restart analysis"), type="primary", key="reanalyze"):
                _run_brand_analysis(raw_answers)

    st.divider()

    tab_sov, tab_sent, tab_answers, tab_raw, tab_timing = st.tabs(
        [
            "Share of Voice",
            tr("Sentiment", "Sentiment"),
            tr("Alle Antworten", "All answers"),
            tr("Rohdaten", "Raw data"),
            tr("Laufzeit", "Runtime"),
        ]
    )

    # --- Tab 1: Share of Voice bar chart ------------------------------------
    with tab_sov:
        if df.empty:
            st.info(tr(
                "Keine Brands erkannt.\n\n"
                "Bei manuellem Modus: Prüfen ob die Brand-Namen genau so im Text vorkommen "
                "(z.B. 'Nike' vs 'NIKE'). Die Rohdaten-Tab zeigt die vollständigen Antworten.",
                "No brands detected.\n\n"
                "In manual mode: check whether the brand names appear exactly like that in the text "
                "(e.g. 'Nike' vs 'NIKE'). The Raw data tab shows the full answers.",
            ))
        else:
            st.subheader(tr("Wie oft wird jede Brand genannt?", "How often is each brand mentioned?"))
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
            st.plotly_chart(fig, width="stretch")

            # Prominence: average position of each brand within the answers that mention
            # it (1 = first brand named). Share of voice counts presence; this shows how
            # early a brand tends to appear, which matters as much as how often.
            if "rank" in df.columns and df["rank"].notna().any():
                prom = (
                    df.dropna(subset=["rank"])
                    .groupby("brand")["rank"]
                    .mean()
                    .round(2)
                    .sort_values()
                    .reset_index()
                    .rename(columns={"brand": tr("Marke", "Brand"),
                                     "rank": tr("Ø Position", "Avg. position")})
                )
                if not prom.empty:
                    st.markdown(f"**{tr('Prominenz — Ø Position in der Antwort', 'Prominence — avg. position in the answer')}**")
                    st.caption(tr(
                        "Niedriger = die Marke wird tendenziell früher genannt (1 = zuerst).",
                        "Lower = the brand tends to be named earlier (1 = first).",
                    ))
                    st.dataframe(prom, width="stretch", hide_index=True)

            # Heatmap: brand × question, using Q1/Q2/… labels
            st.subheader(tr("Sentiment-Heatmap: Brand × Frage", "Sentiment heatmap: Brand × Question"))
            st.caption(tr(
                "Grün = positiv (+1), Grau = neutral (0), Rot = negativ (−1). Weiß = nicht erwähnt.",
                "Green = positive (+1), Gray = neutral (0), Red = negative (−1). White = not mentioned.",
            ))
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
            st.plotly_chart(fig_heat, width="stretch")

            # Question reference table
            with st.expander(tr("Fragenverzeichnis", "Question directory") + f" (Q1 … Q{len(q_order)})"):
                for q, label in q_label.items():
                    st.markdown(f"**{label}** — {q}")

    # --- Tab 2: Sentiment breakdown + brand mention summary -----------------
    with tab_sent:
        if df.empty:
            st.info(tr("Keine Brands erkannt.", "No brands detected."))
        else:
            # Sentiment % breakdown per brand as metrics
            st.subheader(tr("Sentiment-Verteilung je Brand", "Sentiment distribution per brand"))
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
                    st.markdown(tr(
                        f"🟢 {pos_pct}% positiv  \n"
                        f"⚪ {neu_pct}% neutral  \n"
                        f"🔴 {neg_pct}% negativ  \n"
                        f"*{total} Nennungen gesamt*",
                        f"🟢 {pos_pct}% positive  \n"
                        f"⚪ {neu_pct}% neutral  \n"
                        f"🔴 {neg_pct}% negative  \n"
                        f"*{total} mentions total*",
                    ))

            st.divider()
            st.subheader(tr("Sentiment-Vergleich (gestapelt)", "Sentiment comparison (stacked)"))
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
                labels={"mentions": tr("Anzahl Nennungen", "Number of mentions"), "brand": ""},
            )
            fig.update_traces(textposition="inside", textfont_size=12)
            st.plotly_chart(fig, width="stretch")

            st.divider()
            st.subheader(tr("Was sagen die Modelle über jede Brand?", "What do the models say about each brand?"))
            st.caption(tr(
                "Alle Sätze, in denen die Brand erwähnt wurde, gesammelt pro Sentiment.",
                "All sentences mentioning the brand, grouped by sentiment.",
            ))
            for brand in sorted(df["brand"].unique()):
                brand_df = df[df["brand"] == brand]
                total    = int(brand_df["mentions"].sum())
                with st.expander(f"{brand} — " + tr(f"{total} Nennungen", f"{total} mentions")):
                    for sentiment, label, icon in [
                        ("positive", tr("Positiv", "Positive"), "🟢"),
                        ("neutral",  tr("Neutral", "Neutral"), "⚪"),
                        ("negative", tr("Negativ", "Negative"), "🔴"),
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
                                quote = f'*„{excerpt}“*' if st.session_state.get("lang", "de") == "de" else f'*"{excerpt}"*'
                                parts.append(quote)
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

        st.subheader(tr("Alle Antworten", "All answers"))
        st.caption(tr(
            f"{len(q_order_ans)} Fragen × {n_runs_seen} Run(s) = {len(results)} Antworten",
            f"{len(q_order_ans)} questions × {n_runs_seen} run(s) = {len(results)} answers",
        ))

        opt_all_questions = tr("Alle Fragen", "All questions")
        filter_q = st.selectbox(
            tr("Frage filtern", "Filter question"),
            options=[opt_all_questions] + [f"Q{q_num_ans[q]} — {q[:60]}" for q in q_order_ans],
            key="answers_filter",
        )
        # Resolve filter back to full question text
        filter_q_text = (
            None if filter_q == opt_all_questions
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
                st.markdown(f"**{tr('Vollständige Frage', 'Full question')}:** {r['question']}")
                st.divider()
                if answer:
                    st.markdown(answer)
                else:
                    st.error(tr("Keine Antwort erhalten (API-Fehler).", "No answer received (API error)."))
                if brands_found:
                    st.divider()
                    st.markdown(f"**{tr('Erkannte Brands', 'Detected brands')}:**")
                    brand_cols = st.columns(min(len(brands_found), 4))
                    for bc, b in zip(brand_cols, brands_found):
                        sentiment_icon = {"positive": "🟢", "neutral": "⚪", "negative": "🔴"}.get(
                            b.get("sentiment", "neutral"), "⚪"
                        )
                        bc.markdown(f"{sentiment_icon} **{b['brand']}**")

    # --- Tab 4: Raw data table -----------------------------------------------
    with tab_raw:
        st.subheader(tr("Rohdaten", "Raw data"))
        col_frage    = tr("Frage", "Question")
        col_modell   = tr("Modell", "Model")
        col_antwort  = tr("Antwort", "Answer")
        col_tok_in   = tr("Tok. Input", "Tok. input")
        col_tok_out  = tr("Tok. Antwort", "Tok. answer")
        col_tok_an   = tr("Tok. Analyse", "Tok. analysis")
        col_api_ok   = tr("API OK", "API OK")
        raw_rows = []
        for r in results:
            answer = r.get("answer") or ""
            raw_rows.append({
                col_frage:    r["question"][:80] + ("..." if len(r["question"]) > 80 else ""),
                "Run":        r["run"],
                col_modell:   r.get("model", ""),
                col_antwort:  (answer[:200] + "...") if len(answer) > 200 else answer,
                "Brands":     ", ".join(b["brand"] for b in r.get("brands_found", [])),
                col_tok_in:   r.get("tokens_in", 0),
                col_tok_out:  r.get("tokens_out", 0),
                col_tok_an:   r.get("tokens_analysis", 0),
                col_api_ok:   "✅" if answer else "❌",
            })
        st.dataframe(
            pd.DataFrame(raw_rows),
            column_config={
                col_tok_in: st.column_config.NumberColumn(
                    col_tok_in,
                    help=tr(
                        "Prompt-Tokens: Größe des Eingabe-Textes (Systemprompt + Frage), den das Modell erhält.",
                        "Prompt tokens: size of the input text (system prompt + question) the model receives.",
                    ),
                ),
                col_tok_out: st.column_config.NumberColumn(
                    col_tok_out,
                    help=tr(
                        "Completion-Tokens: Tokens der generierten Antwort in Phase 1. Bei Reasoning-Modellen schließt das interne Denkschritte ein.",
                        "Completion tokens: tokens of the generated answer in Phase 1. For reasoning models this includes internal thinking steps.",
                    ),
                ),
                col_tok_an: st.column_config.NumberColumn(
                    col_tok_an,
                    help=tr(
                        "Completion-Tokens der Markenanalyse (Phase 2), summiert über alle Chunks dieser Frage.",
                        "Completion tokens of the brand analysis (Phase 2), summed across all chunks for this question.",
                    ),
                ),
            },
            width="stretch",
        )

        # Full raw answers for the first 3 runs — useful for debugging brand matching
        with st.expander("🔍 " + tr("Debug: Erste 3 vollständige Antworten", "Debug: First 3 full answers")):
            for r in results[:3]:
                st.markdown(f"**{tr('Frage', 'Question')}:** {r['question']}")
                st.markdown(f"**{tr('Vollständige Antwort', 'Full answer')}:**")
                st.text(r.get("answer") or tr("— LEER —", "— EMPTY —"))
                st.markdown(f"**{tr('Erkannte Brands', 'Detected brands')}:** {r.get('brands_found', [])}")
                st.divider()

    # --- Tab 5: Runtime -------------------------------------------------------
    with tab_timing:
        timing = st.session_state.get("timing", {})
        p1_s     = timing.get("phase1_s", 0.0)
        p2_s     = timing.get("phase2_s", 0.0)
        p1_calls = timing.get("p1_calls", [])
        p2_calls = timing.get("p2_calls", [])

        if not p1_calls and not p2_calls:
            st.info(tr(
                "Keine Laufzeitdaten verfügbar (Analyse wurde vor diesem Update gestartet).",
                "No runtime data available (analysis was started before this update).",
            ))
        else:
            # --- Summary metrics ---
            all_elapsed = [x["elapsed_s"] for x in p1_calls + p2_calls]
            avg_all = sum(all_elapsed) / len(all_elapsed) if all_elapsed else 0
            min_all = min(all_elapsed) if all_elapsed else 0
            max_all = max(all_elapsed) if all_elapsed else 0

            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric(tr("Phase 1 — Antworten", "Phase 1 — Answers"), f"{p1_s:.1f}s", f"{len(p1_calls)} Calls",
                       help=tr("Gesamtdauer, in der das Modell alle Fragen beantwortet hat (parallele Calls).",
                               "Total time for the model to answer all questions (parallel calls)."))
            mc2.metric(tr("Phase 2 — Analyse", "Phase 2 — Analysis"), f"{p2_s:.1f}s", f"{len(p2_calls)} Calls",
                       help=tr("Gesamtdauer der Marken- und Sentiment-Analyse. Je mehr Runs und Fragen, desto mehr Chunks.",
                               "Total time for the brand and sentiment analysis. More runs and questions means more chunks."))
            mc3.metric(tr("Gesamt", "Total"), f"{p1_s + p2_s:.1f}s",
                       help=tr("Phase 1 + Phase 2 zusammen. Wartezeiten bei Rate-Limits (429) sind enthalten.",
                               "Phase 1 + Phase 2 combined. Wait times from rate limits (429) are included."))
            mc4.metric(tr("Ø Antwortzeit", "Avg. response time"), f"{avg_all:.1f}s", f"min {min_all:.1f}s  max {max_all:.1f}s",
                       help=tr("Durchschnitt über alle Calls beider Phasen. Delta: schnellster und langsamster Call.",
                               "Average across all calls in both phases. Delta: fastest and slowest call."))

            st.divider()

            # Build question label map (Q1, Q2, …) from result order
            q_order_t = list(dict.fromkeys(r["question"] for r in results))
            q_label_t = {q: f"Q{i+1}" for i, q in enumerate(q_order_t)}

            col_p1, col_p2 = st.columns(2)

            # Phase 1 — response time histogram
            with col_p1:
                st.subheader(tr("Phase 1 — Antwortzeit-Verteilung", "Phase 1 — Response time distribution"))
                if p1_calls:
                    df_p1 = pd.DataFrame(p1_calls)
                    fig_hist = px.histogram(
                        df_p1,
                        x="elapsed_s",
                        nbins=min(30, max(5, len(p1_calls) // 2)),
                        labels={"elapsed_s": tr("Antwortzeit (s)", "Response time (s)"),
                                "count": tr("Anzahl Calls", "Number of calls")},
                        color_discrete_sequence=["#3498db"],
                    )
                    fig_hist.update_layout(bargap=0.05, showlegend=False)
                    st.plotly_chart(fig_hist, width="stretch")
                    ok_pct = round(sum(1 for x in p1_calls if x["ok"]) / len(p1_calls) * 100, 1)
                    st.caption(tr(f"{ok_pct}% der Calls erfolgreich", f"{ok_pct}% of calls successful"))

            # Phase 2 — per-question bar chart
            with col_p2:
                st.subheader(tr("Phase 2 — Analysezeit je Frage", "Phase 2 — Analysis time per question"))
                if p2_calls:
                    df_p2 = pd.DataFrame(p2_calls)
                    df_p2["label"] = df_p2["question"].map(q_label_t)
                    df_p2 = df_p2.sort_values("elapsed_s", ascending=False)
                    fig_p2 = px.bar(
                        df_p2,
                        x="elapsed_s",
                        y="label",
                        orientation="h",
                        labels={"elapsed_s": tr("Zeit (s)", "Time (s)"), "label": ""},
                        color="elapsed_s",
                        color_continuous_scale="Blues",
                    )
                    fig_p2.update_layout(showlegend=False, coloraxis_showscale=False,
                                        height=max(200, 28 * len(df_p2) + 60))
                    st.plotly_chart(fig_p2, width="stretch")

            # Phase 1 detail table
            if p1_calls:
                with st.expander(tr("Phase 1 — Detailtabelle", "Phase 1 — Detail table")):
                    col_frage_t = tr("Frage", "Question")
                    col_zeit_t  = tr("Zeit (s)", "Time (s)")
                    col_erfolg  = tr("Erfolg", "Success")
                    df_detail = pd.DataFrame(p1_calls)
                    df_detail[col_frage_t] = df_detail["question"].map(
                        lambda q: f"{q_label_t.get(q, '?')} — {q[:60]}{'…' if len(q) > 60 else ''}"
                    )
                    df_detail = df_detail.rename(columns={"run": "Run", "elapsed_s": col_zeit_t, "ok": col_erfolg})
                    df_detail[col_erfolg] = df_detail[col_erfolg].map({True: "✅", False: "❌"})
                    st.dataframe(
                        df_detail[[col_frage_t, "Run", col_zeit_t, col_erfolg]].sort_values(col_zeit_t, ascending=False),
                        width="stretch",
                    )

    st.divider()

    # --- Export options -----------------------------------------------------
    st.subheader("Export")
    col1, col2, col3 = st.columns(3)

    with col1:
        if not df.empty:
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 " + tr("Analyse als CSV", "Analysis as CSV"),
                csv_bytes,
                "brand_visibility_analysis.csv",
                "text/csv",
            )

    with col2:
        raw_json = json.dumps(results, ensure_ascii=False, indent=2)
        st.download_button(
            "📥 " + tr("Rohdaten als JSON", "Raw data as JSON"),
            raw_json.encode("utf-8"),
            "brand_visibility_raw.json",
            "application/json",
        )

    with col3:
        # Saves in brand_monitor.py format — compatible with analyze_csv.py
        if st.button("💾 " + tr("Lokal speichern (results/)", "Save locally (results/)")):
            filepath = save_csv(results, cfg.get("model", ""))
            st.success(tr(f"Gespeichert: `{filepath}`", f"Saved: `{filepath}`"))

    st.divider()
    if st.button("🔄 " + tr("Neue Analyse starten", "Start new analysis")):
        reset_process()


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
elif step == 5:
    render_step5()
