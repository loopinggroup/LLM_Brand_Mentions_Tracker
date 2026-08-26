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

from langdock_evidence import EvidenceRecorder, key_fingerprint, redact, response_headers

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
# Support evidence
#
# brand_visibility.log is written for US: it summarises. Langdock support needs the
# opposite — the full GET /agent/v1/models response, the exact POST body, and the
# UNMODIFIED response body, all from one run and provably one key. None of that
# survives the normal log (the GET response is parsed and dropped, the POST body is
# never written, error bodies pass through _strip_html() at 600 chars), so the Agent
# API calls additionally write raw transcripts here.
#
# Append-only, one JSON object per request, no API key — only key_fingerprint(). Cheap
# enough to leave on permanently, and it means the next support ticket can be answered
# from real production traffic instead of a fresh reproduction run.
# See support/langdock_repro.py for the standalone version and the report generator.
# ---------------------------------------------------------------------------
evidence = EvidenceRecorder("support_evidence.jsonl")

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
        "phase1_complete": False,  # False + raw_answers present ⇒ a run was interrupted
        "timing":         {},   # phase wall times + per-call elapsed data
        "lang":           "de", # UI + model response language: "de" or "en"
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

def reset_process():
    """Clears everything but the language preference and restarts at Step 1."""
    for key in ["step", "config", "questions", "raw_answers", "phase1_errors", "phase1_complete", "results", "unlisted_brands", "analysis_summary", "timing"]:
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
# The Agent API has its own model catalog with its own IDs (e.g. "claude-opus-4-6-v1",
# "gpt-5-mini-eu") that don't match the provider-passthrough model IDs above — always
# fetched live, never hardcoded, see fetch_agent_models() below.
AGENT_MODELS_URL    = "https://api.langdock.com/agent/v1/models"
# Sentinel sent to the passthrough endpoint to make it reply with its accepted model
# IDs (see _probe_models). Must be a name no workspace can actually have.
_PROBE_MODEL_ID     = "__langdock_model_probe__"
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

# Some newer models reject `temperature` ("`temperature` is deprecated for this
# model", HTTP 400) — claude-opus-4-8, the analysis model, among them. Which ones
# is learned at runtime from the first such 400 and remembered here, so the retry
# happens once per model instead of once per call.
_TEMPERATURE_UNSUPPORTED: set[str] = set()

# Model IDs the Agent API rejected during this run. The picker reloads its catalog
# and clears the selection when one shows up here, per Langdock's guidance that
# deployment IDs change and must not be substituted with a guessed alias.
_invalid_model_id: str | None = None

# "Model provided (x) is not available…" / "Invalid model…" — a 400 about the model
# itself, as opposed to a malformed request.
_MODEL_ERROR_RE = re.compile(r"model provided|invalid model|available models are", re.IGNORECASE)


def invalidate_model(model_id: str) -> None:
    global _invalid_model_id
    _invalid_model_id = model_id
    log.warning("Model '%s' rejected by the Agent API — catalog reloaded, selection cleared", model_id)


def invalid_model_id() -> str | None:
    return _invalid_model_id


def clear_invalid_model() -> None:
    global _invalid_model_id
    _invalid_model_id = None

# Models the API rejected as unavailable during this run. Scoped PER MODEL on
# purpose: a dead deployment ID is a fact about that one model, not about the
# workspace, so in a multi-model run its siblings must keep going. (This used to
# trip the run-level abort below, which made one stale ID look like a total
# outage — every later call returned the abort message whatever model it asked for.)
_dead_models: set[str] = set()


def mark_model_dead(model_id: str) -> None:
    _dead_models.add(model_id)
    log.warning("Model '%s' marked dead for this run — its calls are skipped, other models continue", model_id)


def is_model_dead(model_id: str) -> bool:
    return model_id in _dead_models


def reset_dead_models() -> None:
    _dead_models.clear()


def _dead_model_error(model: str, lang: str) -> str:
    return tr(
        f"Modell '{model}' wurde vom Katalog abgelehnt und ist nicht mehr verfügbar — "
        "Modell-Liste neu laden und neu auswählen. (Kein API-Limit.)",
        f"Model '{model}' was rejected by the catalog and is no longer available — "
        "reload the model list and pick again. (Not an API limit.)",
        lang=lang,
    )

# ---------------------------------------------------------------------------
# Run-level abort ("circuit breaker")
#
# Some API errors arrive as 429 but are permanent for the rest of the run —
# a workspace spending limit is the one that actually bit us: every one of the
# 60 calls burned 4 attempts with 15/30/60/120s backoff, so a run that could
# never succeed sat there sleeping for ~45 minutes and looked frozen.
#
# The first call to see such an error records it here; every other call — including
# ones already sleeping in backoff — then gives up immediately, so the phase ends
# in seconds with whatever was collected instead of hanging.
# ---------------------------------------------------------------------------
_run_abort_reason: str | None = None

# 429s that will not fix themselves within a run. Deliberately narrow: the ordinary
# TPM limit ("exceeded the maximum number of tokens per minute") must keep retrying.
_FATAL_LIMIT_RE = re.compile(
    r"spending limit|monthly limit|quota|billing|credit balance|budget|insufficient[_ ]",
    re.IGNORECASE,
)


def _api_message(raw_body: str) -> str:
    """The human-readable part of an API error body, for showing to the user."""
    try:
        msg = json.loads(raw_body).get("message")
        if isinstance(msg, str) and msg:
            return msg
    except (json.JSONDecodeError, AttributeError, TypeError):
        pass
    return _strip_html(raw_body, 200)


def abort_run(reason: str) -> None:
    global _run_abort_reason
    if _run_abort_reason is None:
        _run_abort_reason = reason
        log.error("Run aborted — every further call is skipped: %s", reason)


def reset_run_abort() -> None:
    global _run_abort_reason
    _run_abort_reason = None


def run_abort_reason() -> str | None:
    return _run_abort_reason


def _abort_error(lang: str) -> str:
    # Only ever reached for a genuine limit: abort_run() is now called exclusively
    # from the _FATAL_LIMIT_RE 429 branches. A model/catalog problem takes the
    # per-model path instead (_dead_model_error), so this wording cannot mislabel one.
    return tr(
        f"Abgebrochen — API-Limit erreicht: {_run_abort_reason}",
        f"Aborted — API limit reached: {_run_abort_reason}",
        lang=lang,
    )


def _sleep_unless_aborted(seconds: float) -> bool:
    """
    Backoff sleep that stays responsive: returns False as soon as another thread
    aborts the run, so workers don't keep sleeping through a dead run.
    """
    deadline = time.time() + seconds
    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            return True
        if _run_abort_reason is not None:
            return False
        time.sleep(min(0.5, remaining))

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

# Display bucket for models that match none of the known provider prefixes.
_OTHER_PROVIDER = "Sonstige / Other"   # not tr(): also used as a cached dict key

# Provider order for the passthrough picker, so its default (first entry) is stable
# across probes. Purely a display ordering — the groups themselves are filled from
# whatever the live probes return.
_PROVIDER_ORDER = ("OpenAI", "Anthropic", "Google", "Mistral", "Meta", _OTHER_PROVIDER)

# Selectbox sentinel for the free-text escape hatch. Not a model ID, and chosen so it
# cannot collide with one.
_CUSTOM_MODEL_OPTION = "__custom__"

# Endpoint routing for the PASSTHROUGH path only. These prefixes describe the plain
# IDs the passthrough endpoints publish (see the catalog notes below) — they are not
# valid for Agent-API IDs, which carry vendor/region decoration and must never reach
# this path at all. call_langdock() only consults them once web_search has already
# routed the Agent case away.
def _is_anthropic_model(model: str) -> bool:
    return model.startswith("claude-")

def _is_google_model(model: str) -> bool:
    return model.startswith("gemini-")


# ---------------------------------------------------------------------------
# Live model catalogs
#
# Two catalogs exist and they OVERLAP heavily — an earlier version of this comment
# claimed they were disjoint, which is wrong. Measured 18.08.2026 with single
# max_tokens=1 probe calls:
#
#   • Agent Completions API (web search on) → GET /agent/v1/models. IDs are
#     deployment names and may carry vendor/region decoration
#     ("eu.anthropic.claude-opus-4-7", "gpt-5-mini-eu", "claude-opus-5@default").
#   • Provider passthrough (web search off) → one endpoint per provider, no catalog
#     endpoint. Each names its accepted IDs in the 400 body
#     ("Invalid model, available models are: a, b, c"), which is what
#     _parse_available_models / _probe_models below read.
#
# What the probes actually established:
#   1. Each passthrough endpoint knows ONLY its own provider's models. Probing
#      /openai/…  returns gpt-*/o3/o4-mini/llama and no claude at all; /anthropic/…
#      returns claude-* only; /google/… returns gemini-* only. So the passthrough
#      catalog needs one probe PER endpoint, merged — probing just one leaves the
#      picker missing whole providers (see list_completion_models()).
#   2. An "@default" suffix is accepted by the passthrough and normalized away
#      server-side: "claude-opus-5@default" → 200, response echoes
#      "model":"claude-opus-5". So that spelling is NOT a wrong-catalog marker.
#   3. Vendor/region-PREFIXED forms are rejected everywhere:
#      "eu.anthropic.claude-opus-4-7" 400s against /anthropic/… (its own provider's
#      endpoint) just as it does against /openai/…. This is the form that actually
#      breaks when an Agent ID leaks onto the passthrough path.
#
# The overlap is not a licence to move an ID between catalogs. Each ID is sent
# verbatim to the endpoint of the catalog it came from — routing follows provenance
# (did the user pick with web search on or off?), never string inspection of the ID.
#
# ---------------------------------------------------------------------------
# THE AGENT CATALOG IS NOT CONSISTENT BETWEEN REQUESTS (established 18.08.2026 from
# brand_visibility.log; this cost five probe calls and a full log trawl to pin down,
# so it is written here rather than left to be re-derived).
#
# The Agent API is served by instances whose model lists disagree, so an ID that
# GET /agent/v1/models just returned can be unknown to the instance serving the very
# next request. The evidence:
#
#   • Same ID, same endpoint, both outcomes: "claude-haiku-4-5@20251001" scored
#     95 × HTTP 200 against 7 × HTTP 400 "is not available". A spelling the API
#     genuinely did not accept would score 0 × 200.
#   • Two 400 bodies ONE MILLISECOND apart (10:22:33,637 and ,638) enumerate
#     different catalogs:
#         …637:  claude-opus-4-8@default,  claude-opus-5,          claude-sonnet-5
#         …638:  claude-opus-4-8,          claude-opus-5@default,  claude-sonnet-5@default
#   • Only ANTHROPIC families flip. Across 28 captured listings all 9 Anthropic
#     families appear under two spellings each (bare ↔ @default, -<date> ↔ @<date>,
#     -v1 ↔ @default); all 22 non-Anthropic families are stable, and no non-Anthropic
#     ID ever produced this 400 (gpt-5.6-terra 100/0, gpt-5.6-sol 46/0, gemini 5/0).
#
# Consequences, both implemented:
#   1. A model-related 400 on the Agent path is retried with the SAME ID rather than
#      being treated as a dead model (call_langdock_agent). Retrying reaches a
#      different instance; the model is only retired once every attempt has failed.
#   2. A selection missing from a freshly fetched catalog is assumed to be a spelling
#      flip and re-pointed at the catalog's current entry, not reported as a removal
#      (catalog_equivalent). Warning on every flip made the warning meaningless for
#      the removals it actually exists to report.
#
# What must NOT be inferred from this: that a "close enough" ID may be substituted.
# Both mechanisms above only ever send a string the live catalog itself supplied.
# ---------------------------------------------------------------------------
_AVAILABLE_MODELS_RE = re.compile(r"available models are:\s*([^\"}]+)", re.IGNORECASE)


def _parse_available_models(body: str) -> list[str]:
    """Extracts the model IDs Langdock lists in its 'invalid model' 400 response."""
    m = _AVAILABLE_MODELS_RE.search(body or "")
    if not m:
        return []
    ids = []
    for part in m.group(1).split(","):
        mid = part.strip().strip(".\"'} ")
        # The list is truncated by the API on long catalogs — the final entry can be
        # a fragment ("claude-"), which must not end up in a picker as a real ID.
        if mid and not mid.endswith("-"):
            ids.append(mid)
    return ids


def _normalize_model_id(model: str) -> str:
    """
    Reduces the ID variants of one and the same model to a common form, so a stale
    or wrong-catalog ID can be matched against what an endpoint accepts:
        claude-opus-4-7@default        → claude-opus-4-7
        eu.anthropic.claude-opus-4-7   → claude-opus-4-7
        gpt-5-mini-eu                  → gpt-5-mini
    """
    m = model.strip().lower()
    m = m.split("@", 1)[0]                                  # deployment suffix
    m = re.sub(r"^(eu|us|apac|global)\.", "", m)            # region prefix
    m = re.sub(r"^(anthropic|openai|google|meta|amazon)\.", "", m)  # vendor prefix
    m = re.sub(r"-(eu|us|apac|global)$", "", m)             # region suffix
    m = re.sub(r"-\d{8}$", "", m)                           # dash-joined dated deployment
    m = re.sub(r"-v\d+(:\d+)?$", "", m)                     # bedrock/deployment version suffix
    return m


def _probe_models(url: str, api_key: str, payload: dict) -> list[str]:
    """
    Asks an endpoint for its accepted model IDs by sending a deliberately unknown
    model and reading them out of the 400. Costs no tokens (nothing is generated)
    and is the only way to get the passthrough endpoints' current catalog — they
    have no documented GET /models of their own.
    """
    try:
        r = requests.post(
            url,
            json=payload,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            timeout=20,
        )
        if r.status_code < 400:
            return []
        return _parse_available_models(r.text)
    except requests.exceptions.RequestException as e:
        log.warning("Model probe failed for %s: %s", url, e)
        return []


# ---------------------------------------------------------------------------
# Agent API model catalog — GET /agent/v1/models
#
# Per Langdock support (14.08.2026): this endpoint returns exactly the models the
# workspace can currently use via the Agent API, `data[].id` must be sent to
# /agent/v1/chat/completions *unchanged*, and the IDs are deployment names that
# can change — there is no guaranteed alias. So: no static catalog, no rewriting
# of IDs, no guessing of alternative spellings. Reload instead, and on a
# model-related 400 drop the selection and let the user pick again.
#
# The short TTL keeps this "fetched when the page is used" rather than a stored
# catalog; the reload button next to the picker clears it on demand.
# ---------------------------------------------------------------------------
AGENT_MODELS_TTL = 60

# `seq` of the most recent recorded GET /agent/v1/models. Every recorded POST carries it
# so support can tie a rejected model ID back to the exact catalog response that supplied
# it — which is precisely the pairing they asked for.
_last_catalog_seq: int | None = None


@st.cache_data(ttl=AGENT_MODELS_TTL, show_spinner=False)
def fetch_agent_models(api_key: str) -> tuple[list[dict], str | None]:
    """
    Returns ([{id, region, supportsExtendedThinking}, ...], error).
    IDs are returned verbatim — they are what must be sent as the model value.
    """
    global _last_catalog_seq
    if not api_key:
        return [], None
    try:
        r = requests.get(
            AGENT_MODELS_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=15,
        )
        # Recorded BEFORE raise_for_status so a failed catalog fetch is captured too.
        _last_catalog_seq = evidence.record(
            "GET /agent/v1/models",
            url=AGENT_MODELS_URL,
            key_fingerprint=key_fingerprint(api_key),
            http_status=r.status_code,
            response_headers=response_headers(r),
            response_body_raw=redact(r.text, api_key),
        )["seq"]
        r.raise_for_status()
        models = [
            {
                "id":                       m["id"],
                "region":                   m.get("region", ""),
                "supportsExtendedThinking": bool(m.get("supportsExtendedThinking")),
            }
            for m in r.json().get("data", [])
            if isinstance(m, dict) and m.get("id")
        ]
        models.sort(key=lambda m: m["id"])
        if not models:
            return [], "GET /agent/v1/models returned an empty catalog"
        return models, None
    except requests.exceptions.RequestException as e:
        return [], str(e)
    except (ValueError, KeyError, TypeError) as e:
        return [], f"Unexpected /agent/v1/models response: {e}"


def reload_agent_models() -> None:
    """Drops the cached catalog so the next read re-fetches it."""
    fetch_agent_models.clear()


def _model_provider_label(model_id: str) -> str:
    """
    Provider bucket for a catalog ID. The lookup runs on the NORMALIZED id, because
    Agent-API IDs carry vendor and region decoration that the plain prefixes miss
    ("eu.anthropic.claude-opus-4-7" is Anthropic, not "Sonstige"). This is display
    only — the id that gets sent is never touched.
    """
    m = _normalize_model_id(model_id)
    if m.startswith("claude-"):
        return "Anthropic"
    if m.startswith("gemini-"):
        return "Google"
    if m.startswith(("gpt-", "o1", "o3", "o4")):
        return "OpenAI"
    if m.startswith("mistral"):
        return "Mistral"
    if "llama" in m:
        return "Meta"
    return _OTHER_PROVIDER


def agent_model_label(m: dict) -> str:
    """
    Readable label for the picker: provider and the raw id, nothing else. Purely
    cosmetic — the selectbox value stays the raw `id`, which is what gets sent.

    Region and supportsExtendedThinking are deliberately NOT shown. Both are still
    read into the catalog dicts (fetch_agent_models) — supportsExtendedThinking gates
    the Extended-Thinking checkbox — but extended thinking is switched on later by the
    user, so it is a property of the run, not something the model overview should
    pre-announce.
    """
    return f"{_model_provider_label(m['id'])}  ·  {m['id']}"


def catalog_equivalent(model_id: str, ids: list[str]) -> str | None:
    """
    The live catalog's CURRENT spelling of `model_id`, or None if the catalog holds
    nothing equivalent.

    Same reasoning as the Agent-path retry: the Agent API's instances disagree about
    how to spell Anthropic models ("claude-haiku-4-5@20251001" one moment,
    "claude-haiku-4-5-20251001" the next), so a selection "missing" from a freshly
    fetched catalog is usually a spelling flip, not a removed model. Warning on that
    trained the user to ignore a warning that also fires for real removals.

    The returned value is always an element of `ids` — a live catalog ID, verbatim.
    Nothing is invented here: normalization only decides WHICH catalog entry to point
    at, it never produces the string that gets sent.
    """
    if model_id in ids:
        return model_id
    want = _normalize_model_id(model_id)
    for cid in ids:
        if _normalize_model_id(cid) == want:
            return cid
    return None


def render_passthrough_probe_notice(failed: list[str]) -> None:
    """
    Names the passthrough endpoints whose probe returned nothing. There is no stored
    fallback list any more, so a failed probe means that provider's models are simply
    absent from the picker and have to be typed in by hand.
    """
    if not failed:
        return
    names = ", ".join(failed)
    st.caption("⚠️ " + tr(
        f"Modell-Liste für {names} nicht abrufbar — diese Anbieter fehlen in der Auswahl. "
        "IDs können unten manuell eingegeben werden.",
        f"Could not load the model list for {names} — those providers are missing from the "
        "picker. Their IDs can be entered manually below.",
    ))


def render_agent_model_picker(api_key: str, current_model: str = "", key_prefix: str = "s1") -> str:
    """
    Model picker for the web-search (Agent API) path. The options come from a live
    GET /agent/v1/models and the returned value is `data[].id` verbatim — no
    rewriting, no stored catalog, reload button next to it.
    """
    rejected = invalid_model_id()
    if rejected:
        # A call in the last run exhausted its retries on this ID: reload the catalog.
        # Whether the user has to pick again is decided below, once the fresh catalog
        # is in hand — if it still holds this model under any spelling, there is
        # nothing for them to do.
        reload_agent_models()
        clear_invalid_model()

    col_pick, col_reload = st.columns([6, 1])
    with col_reload:
        st.markdown("<div style='height:1.85rem'></div>", unsafe_allow_html=True)  # align with the input
        if st.button("🔄", key=f"{key_prefix}_reload_models",
                     help=tr("Modell-Liste neu laden", "Reload model list")):
            reload_agent_models()
            st.rerun()

    with col_pick:
        if not api_key:
            st.info(tr(
                "API-Key eingeben, um die für die Websuche verfügbaren Modelle zu laden.",
                "Enter an API key to load the models available for web search.",
            ))
            return ""

        models, err = fetch_agent_models(api_key)
        if err:
            st.warning(tr(
                f"Modell-Liste (GET /agent/v1/models) konnte nicht geladen werden: {err}",
                f"Could not load the model list (GET /agent/v1/models): {err}",
            ))
        if not models:
            if rejected:
                st.warning(tr(
                    f"Modell '{rejected}' wurde von der Agent-API abgelehnt; der Katalog konnte "
                    "zum Abgleich nicht geladen werden.",
                    f"Model '{rejected}' was rejected by the Agent API, and the catalog could not "
                    "be loaded to check against.",
                ))
            return st.text_input(
                tr("Modell-ID (Agent-API, manuell)", "Model ID (Agent API, manual)"),
                value=current_model,
                placeholder="z.B. claude-opus-4-6-v1",
            )

        ids    = [m["id"] for m in models]
        labels = {m["id"]: agent_model_label(m) for m in models}
        if current_model:
            # Re-point the selection at the catalog's current spelling. Only a model
            # the catalog no longer knows under ANY spelling is a real removal worth
            # interrupting the user for.
            equivalent = catalog_equivalent(current_model, ids)
            if equivalent is None:
                st.warning(tr(
                    f"Bisherige Auswahl '{current_model}' ist nicht mehr im Katalog — bitte neu auswählen.",
                    f"Previous selection '{current_model}' is no longer in the catalog — please pick again.",
                ))
                current_model = ""
            else:
                current_model = equivalent
        if rejected and catalog_equivalent(rejected, ids) is None:
            st.warning(tr(
                f"Modell '{rejected}' wurde von der Agent-API abgelehnt und steht nicht mehr im Katalog. "
                "Liste neu geladen — bitte neu auswählen.",
                f"Model '{rejected}' was rejected by the Agent API and is no longer in the catalog. "
                "List reloaded — please pick again.",
            ))
        model  = st.selectbox(
            tr("Modell (Agent-API)", "Model (Agent API)"),
            options=ids,
            index=ids.index(current_model) if current_model in ids else 0,
            format_func=lambda i: labels.get(i, i),
            help=tr(
                "Live aus GET /agent/v1/models — genau die Modelle, die dein Workspace über die "
                "Agent-API nutzen kann. Die ID wird unverändert gesendet; Deployment-IDs ändern "
                "sich, daher keine feste Liste im Code.",
                "Live from GET /agent/v1/models — exactly the models your workspace can use via the "
                "Agent API. The ID is sent unchanged; deployment IDs change, so nothing is hardcoded.",
            ),
        )

    st.caption(tr(
        f"{len(ids)} Modelle live geladen · gesendet wird exakt `{model}`",
        f"{len(ids)} models loaded live · sends exactly `{model}`",
    ))
    return model


def render_agent_model_multiselect(
    api_key: str, current_models: list[str], key_prefix: str = "s3",
) -> tuple[list[str], list[dict]]:
    """
    Same as render_agent_model_picker, but for running several models in one go.
    Returns (selected ids, full catalog) — the catalog is handed back so callers can
    check per-model properties (e.g. supportsExtendedThinking) without re-fetching.
    """
    rejected = invalid_model_id()
    if rejected:
        # See render_agent_model_picker: reload now, decide below against the fresh
        # catalog whether this is a real removal or just a spelling flip.
        reload_agent_models()
        clear_invalid_model()

    col_pick, col_reload = st.columns([6, 1])
    with col_reload:
        st.markdown("<div style='height:1.85rem'></div>", unsafe_allow_html=True)
        if st.button("🔄", key=f"{key_prefix}_reload_models",
                     help=tr("Modell-Liste neu laden", "Reload model list")):
            reload_agent_models()
            st.rerun()

    with col_pick:
        if not api_key:
            st.info(tr(
                "API-Key eingeben, um die für die Websuche verfügbaren Modelle zu laden.",
                "Enter an API key to load the models available for web search.",
            ))
            return [], []

        catalog, err = fetch_agent_models(api_key)
        if err:
            st.warning(tr(
                f"Modell-Liste (GET /agent/v1/models) konnte nicht geladen werden: {err}",
                f"Could not load the model list (GET /agent/v1/models): {err}",
            ))
        if not catalog:
            if rejected:
                st.warning(tr(
                    f"Modell '{rejected}' wurde von der Agent-API abgelehnt; der Katalog konnte "
                    "zum Abgleich nicht geladen werden.",
                    f"Model '{rejected}' was rejected by the Agent API, and the catalog could not "
                    "be loaded to check against.",
                ))
            manual = st.text_input(
                tr("Modell-IDs (Agent-API, kommagetrennt)", "Model IDs (Agent API, comma-separated)"),
                value=", ".join(current_models),
                placeholder="z.B. claude-opus-4-6-v1, gpt-5.6-sol",
            )
            return [m.strip() for m in manual.split(",") if m.strip()], []

        ids     = [m["id"] for m in catalog]
        labels  = {m["id"]: agent_model_label(m) for m in catalog}
        # Re-point each remembered selection at the catalog's current spelling; only
        # the ones with no equivalent at all were really dropped.
        remapped = [(m, catalog_equivalent(m, ids)) for m in current_models]
        dropped  = [m for m, eq in remapped if eq is None]
        if dropped:
            st.warning(tr(
                f"Nicht mehr im Katalog und abgewählt: {', '.join(dropped)}",
                f"No longer in the catalog, deselected: {', '.join(dropped)}",
            ))
        if rejected and catalog_equivalent(rejected, ids) is None:
            st.warning(tr(
                f"Modell '{rejected}' wurde von der Agent-API abgelehnt und steht nicht mehr im Katalog. "
                "Liste neu geladen — bitte neu auswählen.",
                f"Model '{rejected}' was rejected by the Agent API and is no longer in the catalog. "
                "List reloaded — please pick again.",
            ))
        default = list(dict.fromkeys(eq for _, eq in remapped if eq)) or ids[:1]
        selected = st.multiselect(
            tr("Modelle (Agent-API)", "Models (Agent API)"),
            options=ids,
            default=default,
            format_func=lambda i: labels.get(i, i),
            help=tr(
                "Mehrere Modelle = jede Frage wird mit jedem Modell gesammelt. Das vervielfacht "
                "die Call-Anzahl, macht die Modelle aber direkt vergleichbar.",
                "Several models = every question is collected with every model. That multiplies the "
                "number of calls but makes the models directly comparable.",
            ),
        )

    st.caption(tr(
        f"{len(ids)} Modelle live geladen · {len(selected)} ausgewählt",
        f"{len(ids)} models loaded live · {len(selected)} selected",
    ))
    return selected, catalog


# One probe per passthrough endpoint. Each endpoint only knows its own provider's
# models (see the catalog notes above), so a single probe would silently drop whole
# providers from the picker. The sentinel model ID is what makes each endpoint answer
# with its accepted list; no real model ID appears here.
def _passthrough_probe_targets() -> dict[str, tuple[str, dict]]:
    ping = [{"role": "user", "content": "ping"}]
    return {
        "OpenAI": (
            LANGDOCK_URL,
            {"model": _PROBE_MODEL_ID, "messages": ping, "max_completion_tokens": 1},
        ),
        "Anthropic": (
            ANTHROPIC_URL,
            {"model": _PROBE_MODEL_ID, "messages": ping, "max_tokens": 1},
        ),
        "Google": (
            GOOGLE_URL_TEMPLATE.format(model=_PROBE_MODEL_ID),
            {"contents": [{"role": "user", "parts": [{"text": "ping"}]}],
             "generationConfig": {"maxOutputTokens": 1}},
        ),
    }


@st.cache_data(ttl=300, show_spinner=False)
def list_completion_models(api_key: str) -> tuple[dict[str, list[str]], list[str]]:
    """
    Live model catalog for the provider-passthrough endpoints (web search off).

    Returns (grouped_by_provider, failed_endpoints). Every ID comes from a live probe
    of the endpoint that serves it — there is no stored fallback list, so an endpoint
    that cannot be probed is simply absent and is named in `failed_endpoints` for the
    UI to report. Endpoint names, not translated strings: the result is cached and
    must not freeze the language it was first fetched in.
    """
    if not api_key:
        return {}, []

    ids: list[str] = []
    failed: list[str] = []
    for endpoint, (url, payload) in _passthrough_probe_targets().items():
        found = _probe_models(url, api_key, payload)
        if found:
            ids.extend(found)
        else:
            failed.append(endpoint)

    # Bucket by what the ID actually is rather than by which endpoint answered: the
    # OpenAI-compatible endpoint also serves third-party models (llama), which belong
    # under their own provider in the picker.
    grouped: dict[str, list[str]] = {}
    for mid in sorted(dict.fromkeys(ids)):   # dedupe — endpoints repeat entries
        grouped.setdefault(_model_provider_label(mid), []).append(mid)

    ordered = {p: grouped[p] for p in _PROVIDER_ORDER if grouped.get(p)}
    ordered.update({p: m for p, m in grouped.items() if p not in ordered})
    return ordered, failed


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
    extended_thinking: bool = False,
) -> tuple[str | None, str | None, dict]:
    """
    Returns (response_text, error_message, usage).
    On success: (text, None, {"prompt_tokens": N, "completion_tokens": N, "total_tokens": N})
    On failure: (None, human-readable error string, {})

    `lang` must be passed explicitly (not read from st.session_state) because this
    function runs inside ThreadPoolExecutor worker threads, where Streamlit's
    session_state is not accessible.

    `temperature` is forwarded on every path, including the web-search one: the Agent
    API's inline agent object documents a `temperature` field (0-1). Without it, the
    N runs of the same question on that path varied only through differing search
    results, which weakened the whole point of running a question repeatedly.
    """
    if _run_abort_reason is not None:
        return None, _abort_error(lang), {}
    if is_model_dead(model):
        return None, _dead_model_error(model, lang), {}

    if web_search:
        # Real web search is only available via the model-agnostic Agent Completions
        # API (capabilities.webSearch) — the provider-native endpoints below don't
        # support a built-in search tool. Routed separately since it's a different
        # request/response shape (Vercel AI SDK UIMessage, SSE streaming).
        return call_langdock_agent(
            api_key, messages, model, lang=lang,
            temperature=temperature, extended_thinking=extended_thinking,
        )

    if model in _TEMPERATURE_UNSUPPORTED:
        # Learned from an earlier 400 — skip the parameter instead of burning one
        # failed request per call (the analysis runs one call per batch).
        temperature = None

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
            status   = e.response.status_code if e.response is not None else "?"
            raw_body = e.response.text if e.response is not None else ""
            body     = _strip_html(raw_body)
            log.error("HTTP error %s: %s", status, body)

            # Newer reasoning models (claude-opus-4-8 among them) reject `temperature`
            # outright: "`temperature` is deprecated for this model." Dropping the
            # parameter is the correct fix — those models are deterministic-ish by
            # design — and without this retry a whole analysis batch was lost.
            if status == 400 and "temperature" in raw_body.lower() and temperature is not None:
                removed = payload.pop("temperature", None)
                if removed is None:
                    removed = payload.get("generationConfig", {}).pop("temperature", None)
                if removed is not None:
                    _TEMPERATURE_UNSUPPORTED.add(model)
                    log.warning("Model %s rejects 'temperature' — retrying without it", model)
                    continue

            # A model-related 400/422 on the passthrough is the same fact as on the
            # Agent path: this ID is not accepted here. Retire the model rather than
            # substituting a different one — IDs are sent exactly as their catalog
            # published them, and the previous "retry with the closest match" was both
            # a silent model swap and a re-send to the endpoint that just refused it.
            if status in (400, 422) and _MODEL_ERROR_RE.search(raw_body):
                mark_model_dead(model)
                return None, _dead_model_error(model, lang), {}

            if status == 429:
                if _FATAL_LIMIT_RE.search(raw_body):
                    # Spending limit / quota — retrying cannot help, and doing it 4×
                    # per call froze the whole run. Stop everything instead.
                    abort_run(_api_message(raw_body))
                    return None, _abort_error(lang), {}
                # TPM/RPM rate limit. Waits must cover the ~60s rolling window;
                # jitter breaks up thundering herd when many parallel calls hit 429 together.
                wait = 15 * (2 ** attempt) + random.uniform(0, 5)  # 15-20s, 30-35s, 60-65s, 120-125s
                log.info("Rate limited (429), waiting %.1fs before retry (attempt %d/4)", wait, attempt + 1)
                if not _sleep_unless_aborted(wait):
                    return None, _abort_error(lang), {}
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
                400: tr(
                    f"Ungültige Anfrage (400) — häufig ein nicht verfügbarer Modell-Name: '{model}'. Antwort: {body}",
                    f"Invalid request (400) — often an unavailable model name: '{model}'. Response: {body}",
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
    temperature: float | None = None,
    extended_thinking: bool = False,
) -> tuple[str | None, str | None, dict]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }
    if _run_abort_reason is not None:
        return None, _abort_error(lang), {}
    if is_model_dead(model):
        return None, _dead_model_error(model, lang), {}

    # capabilities.extendedThinking: supported by the Agents API since 13.05.2026;
    # GET /agent/v1/models flags the compatible models via supportsExtendedThinking,
    # which is what gates the checkbox in the UI. Enabling it on a model that can't
    # do it returns a 400 — handled below by dropping the flag and retrying, so a
    # stale selection costs one request rather than the whole run.
    capabilities = {"webSearch": True}
    if extended_thinking:
        capabilities["extendedThinking"] = True

    # `model` is sent exactly as it came from GET /agent/v1/models — never rewritten.
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
            "capabilities": capabilities,
        },
        "messages": ui_messages,
        "stream":   True,
    }
    if temperature is not None:
        # Documented on the inline agent object, range 0-1.
        payload["agent"]["temperature"] = max(0.0, min(1.0, temperature))

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
                # Support evidence: the exact body sent, the status, and — for errors —
                # the unmodified response body, tied to the GET that supplied `model`.
                # The body is only read on an error: touching r.text on a 2xx would
                # consume the stream the parser below depends on, and a full answer
                # stream is not what the ticket is about. The request body is snapshotted
                # (it is mutated between attempts) and it CONTAINS THE PROMPT TEXT — worth
                # a look before this file is passed to anyone outside.
                evidence.record(
                    "POST /agent/v1/chat/completions",
                    url=AGENT_URL,
                    key_fingerprint=key_fingerprint(api_key),
                    model_requested=model,
                    from_catalog_seq=_last_catalog_seq,
                    attempt=attempt + 1,
                    request_body=json.loads(json.dumps(payload)),
                    http_status=r.status_code,
                    response_headers=response_headers(r),
                    response_body_raw=(redact(r.text, api_key) if r.status_code >= 400 else None),
                )

                # Check the status code before touching the body — reading
                # e.response.text *after* the `with` block has closed the
                # connection (as happens if we let raise_for_status() raise
                # and catch it outside) can return an empty or truncated body.
                if r.status_code >= 400:
                    status   = r.status_code
                    raw_body = r.text          # unshortened — the model list is parsed from this
                    body     = _strip_html(raw_body)
                    log.error("Agent HTTP error %s: %s", status, body)

                    if status == 429:
                        if _FATAL_LIMIT_RE.search(raw_body):
                            # Spending limit / quota — permanent for this run (see abort_run).
                            abort_run(_api_message(raw_body))
                            return None, _abort_error(lang), {}
                        wait = 15 * (2 ** attempt) + random.uniform(0, 5)
                        log.info("Agent API rate limited (429), waiting %.1fs before retry (attempt %d/4)", wait, attempt + 1)
                        if not _sleep_unless_aborted(wait):
                            return None, _abort_error(lang), {}
                        continue

                    # Optional extras: drop them rather than lose the call if this
                    # model or deployment refuses one (same idea as the temperature
                    # handling on the passthrough path).
                    if status == 400 and "temperature" in raw_body.lower() \
                            and payload["agent"].pop("temperature", None) is not None:
                        log.warning("Agent model %s rejects 'temperature' — retrying without it", model)
                        continue
                    if status == 400 and "thinking" in raw_body.lower() \
                            and payload["agent"]["capabilities"].pop("extendedThinking", None) is not None:
                        log.warning("Agent model %s rejects extendedThinking — retrying without it", model)
                        continue

                    # A model-related 400 here is usually REPLICA NOISE, not a fact
                    # about the catalog: the Agent API is served by instances whose
                    # Anthropic model lists disagree, so the very ID GET /agent/v1/models
                    # just handed us can be unknown to the instance that serves the next
                    # request. Measured over this project's log: claude-haiku-4-5@20251001
                    # scored 95×200 vs 7×400 on the same endpoint with the same ID (see
                    # the catalog notes at the top of this file). So retry the SAME ID —
                    # verbatim, never a guessed alternative spelling — and let a
                    # different instance answer. Only when every attempt has failed is
                    # the model genuinely gone: retire it then, and flag the selection so
                    # the UI reloads. Other models in the run are unaffected either way;
                    # only a spending limit (the 429 branch above) stops everything.
                    if status == 400 and _MODEL_ERROR_RE.search(raw_body):
                        if attempt < 3:
                            log.warning(
                                "Agent model '%s' reported unavailable (attempt %d/4) — "
                                "retrying the same ID, this is usually an inconsistent catalog replica",
                                model, attempt + 1,
                            )
                            continue
                        invalidate_model(model)
                        mark_model_dead(model)
                        return None, _dead_model_error(model, lang), {}

                    available = _parse_available_models(raw_body)
                    model_hint = (
                        tr(" Verfügbar: ", " Available: ", lang=lang) + ", ".join(available)
                        if available else ""
                    )
                    messages_map = {
                        400: tr(
                            f"Ungültige Anfrage (400) — Modell '{model}'.{model_hint} Antwort: {body}",
                            f"Invalid request (400) — model '{model}'.{model_hint} Response: {body}",
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
    reset_run_abort()   # this is the user re-checking after fixing something
    reset_dead_models()
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

def generate_questions(
    api_key: str, topic: str, n: int, model: str, web_search: bool = False,
) -> tuple[list[str], str | None]:
    lang = st.session_state.get("lang", "de")
    reset_run_abort()   # new user-initiated action — don't inherit an earlier run's abort
    reset_dead_models()
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
        # Must match the picker the `model` came from: with web search on it is an
        # Agent-API deployment ID, which only the Agent endpoint accepts. Defaulting
        # this to False sent those IDs to a provider passthrough, where the
        # vendor/region-prefixed ones ("eu.anthropic.…") 400 as unavailable.
        web_search=web_search,
    )
    # The Agent-path logger records neither the caller nor the token budget (unlike the
    # passthrough one), so without this line a question-generation call is
    # indistinguishable from any other agent call in the log.
    endpoint = "agent" if web_search else "passthrough"
    if not text:
        log.warning(
            "generate_questions FAILED — model=%s | endpoint=%s | max_tokens=%d | requested=%d | %s",
            model, endpoint, QUESTION_MAX_TOKENS, n, err,
        )
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
        # Used to return silently, which made a zero-question result invisible in the
        # log — the call itself logs as a clean HTTP 200, so nothing marked the failure.
        # Log the reply itself: the only way to tell "model answered with prose" from
        # "model answered with nothing" after the fact.
        log.warning(
            "generate_questions PARSED ZERO — model=%s | endpoint=%s | max_tokens=%d | "
            "requested=%d | json_array=%s | reply[:200]=%r",
            model, endpoint, QUESTION_MAX_TOKENS, n, bool(parsed), text[:200],
        )
        return [], tr(
            "Es konnten keine Fragen aus der Antwort extrahiert werden.",
            "No questions could be extracted from the response.",
            lang=lang,
        )
    if len(questions) > n:
        questions = questions[:n]  # model over-generated; trim to the requested count
    log.info(
        "generate_questions OK — model=%s | endpoint=%s | max_tokens=%d | requested=%d | got=%d",
        model, endpoint, QUESTION_MAX_TOKENS, n, len(questions),
    )
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
    short_answer: bool = False, extended_thinking: bool = False,
    market: str = "",
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
    if market.strip():
        # Langdock's API has no location/region parameter (checked against the agent
        # schema, the capabilities list and the changelog), so the market is steered
        # through the prompt. That is a real lever for this use case: brand
        # recommendations and the sources a search picks differ heavily by market.
        content = _with_market_context(content, market.strip(), lang)

    return call_langdock(
        api_key,
        [{"role": "user", "content": content}],
        model=model,
        max_tokens=max_tokens,
        web_search=web_search,
        lang=lang,
        # Non-zero so repeated runs of the same question vary — otherwise the
        # multiple-runs statistic is meaningless.
        temperature=COLLECTION_TEMPERATURE,
        extended_thinking=extended_thinking,
    )


def _with_market_context(content: str, market: str, lang: str) -> str:
    if lang == "de":
        return (
            f"Kontext: Beantworte die Frage aus Sicht des Marktes „{market}“. "
            f"Berücksichtige Anbieter, Marken und Angebote, die dort tatsächlich verfügbar sind, "
            f"und stütze dich bevorzugt auf Quellen aus diesem Markt.\n\n{content}"
        )
    return (
        f"Context: answer from the perspective of the \"{market}\" market. "
        f"Consider providers, brands and offers actually available there, and prefer "
        f"sources from that market.\n\n{content}"
    )


# ---------------------------------------------------------------------------
# HTML stripper — removes tags and collapses whitespace for clean log output.
# Used so 502 / 503 gateway error bodies don't spam logs with raw HTML.
# ---------------------------------------------------------------------------
def _strip_html(text: str, max_len: int = 600) -> str:
    # 600, not 200: Langdock's "invalid model" body lists every accepted model ID
    # after ~100 characters of preamble, and cutting that off hid the one piece of
    # information needed to fix the error.
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
        if run_abort_reason():
            # Spending limit or similar: the remaining batches would each burn a
            # full retry cycle to fail the same way. Keep what was analyzed.
            errors.append(_abort_error(lang))
            break
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
# Brand matching for the raw export
#
# Deliberately separate from the LLM analysis in phase 2: this is a plain string
# match against the brands the user specified, available immediately after
# collection and without a single extra API call. It also covers the *sources* —
# a brand whose site the model linked to is a visibility hit even when the brand
# name never appears in the answer text.
# ---------------------------------------------------------------------------
def _brand_pattern(brand: str) -> re.Pattern | None:
    r"""'The North Face' → /\bthe[\W_]{0,3}north[\W_]{0,3}face\b/i"""
    parts = [re.escape(p) for p in re.split(r"[\W_]+", (brand or "").strip()) if p]
    if not parts:
        return None
    return re.compile(r"\b" + r"[\W_]{0,3}".join(parts) + r"\b", re.IGNORECASE)


def _source_host(url: str) -> str:
    return re.sub(r"^https?://", "", url or "", flags=re.IGNORECASE).split("/")[0].lower()


def find_brands_in_text(text: str, brands: list[str]) -> list[str]:
    """Brands mentioned in the answer text, in the order they were specified."""
    return [b for b in brands if (p := _brand_pattern(b)) and p.search(text or "")]


def find_brands_in_sources(sources: list[dict], brands: list[str]) -> dict[str, list[str]]:
    """
    {brand: [matching hosts]} for the cited sources. A brand counts as found when
    its name appears in the source title, or when its compact key appears in the
    host — so "nike" matches store.nike.com and nikepartner.de alike, which is the
    point when the question is which sites a model links to.
    """
    hits: dict[str, list[str]] = {}
    for src in sources or []:
        host  = _source_host(src.get("url", ""))
        title = src.get("title", "") or ""
        for b in brands:
            key     = _normalize_brand_key(b)
            pattern = _brand_pattern(b)
            in_host  = bool(key) and key in re.sub(r"[^a-z0-9]", "", host)
            in_title = bool(pattern) and bool(pattern.search(title))
            if (in_host or in_title) and host not in hits.setdefault(b, []):
                hits[b].append(host)
    return {b: h for b, h in hits.items() if h}


def build_raw_export(raw_answers: list[dict], brands: list[str]) -> pd.DataFrame:
    """
    Flat, spreadsheet-friendly view of the collected answers — including whether
    each specified brand was found, separately for the answer text and the cited
    sources. `sources` itself is flattened to a host list so the CSV stays readable.
    """
    rows = []
    for r in raw_answers:
        sources     = r.get("sources", []) or []
        in_answer   = find_brands_in_text(r.get("answer", ""), brands)
        in_sources  = find_brands_in_sources(sources, brands)
        found       = list(dict.fromkeys(in_answer + list(in_sources)))
        rows.append({
            "question":            r.get("question", ""),
            "run":                 r.get("run", ""),
            "model":               r.get("model", ""),
            "brand_found":         bool(found),
            "brands_found":        "; ".join(found),
            "brands_in_answer":    "; ".join(in_answer),
            "brands_in_sources":   "; ".join(in_sources),
            "brands_missing":      "; ".join(b for b in brands if b not in found),
            "source_hosts":        "; ".join(dict.fromkeys(_source_host(s.get("url", "")) for s in sources)),
            "source_urls":         " | ".join(s.get("url", "") for s in sources),
            "n_sources":           len(sources),
            "web_search_used":     r.get("web_search_used", False),
            "citation_count":      r.get("citation_count", 0),
            "tokens_in":           r.get("tokens_in", 0),
            "tokens_out":          r.get("tokens_out", 0),
            "answer":              r.get("answer", ""),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CSV export
# Saves in the same column format as brand_monitor.py so the file is
# compatible with analyze_csv.py for downstream analysis.
# ---------------------------------------------------------------------------
def save_csv(results: list[dict], model: str, brands: list[str] | None = None, filename: str = "") -> str:
    Path("results").mkdir(exist_ok=True)
    if not filename:
        ts       = datetime.now().strftime("%Y%m%d_%H%M")
        filename = f"brand_visibility_{ts}.csv"
    if not filename.endswith(".csv"):
        filename += ".csv"
    path   = f"results/{Path(filename).name}"
    brands = brands or []
    fieldnames = [
        "run_date", "model", "prompt", "repetition",
        "brands_found", "mention_count",
        # Source-side visibility: which of the user's brands the model actually
        # linked to. Independent of whether the name appears in the text.
        "brands_in_sources", "source_hosts", "raw_response",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            sources    = r.get("sources", []) or []
            in_sources = find_brands_in_sources(sources, brands)
            brands_str = "; ".join(b["brand"] for b in r.get("brands_found", []))
            writer.writerow({
                "run_date":          datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                "model":             r.get("model", model),
                "prompt":            r["question"],
                "repetition":        r["run"],
                "brands_found":      brands_str,
                "mention_count":     len(r.get("brands_found", [])),
                "brands_in_sources": "; ".join(in_sources),
                "source_hosts":      "; ".join(dict.fromkeys(_source_host(s.get("url", "")) for s in sources)),
                "raw_response":      r.get("answer", "")[:500].replace("\n", " "),
            })
    return path


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
            model = render_agent_model_picker(api_key, key_prefix="s1")
        else:
            catalog, probe_failed = list_completion_models(api_key)
            render_passthrough_probe_notice(probe_failed)
            flat   = [(p, m) for p, ms in catalog.items() for m in ms]
            ids    = [m for _, m in flat] + [_CUSTOM_MODEL_OPTION]
            labels = {m: f"{p}  ·  {m}" for p, m in flat}
            labels[_CUSTOM_MODEL_OPTION] = custom_label
            if not flat:
                st.info(tr(
                    "API-Key eingeben, um die verfügbaren Modelle zu laden.",
                    "Enter an API key to load the available models.",
                ) if not api_key else tr(
                    "Keine Modell-Liste verfügbar — bitte die Modell-ID manuell eingeben.",
                    "No model list available — please enter the model ID manually.",
                ))
            model_option = st.selectbox(
                tr("Modell", "Model"),
                options=ids,
                index=0,
                format_func=lambda i: labels.get(i, i),
                help=tr(
                    "Live aus deinem Langdock-Workspace geladen (sobald ein API-Key gesetzt ist) — "
                    "je ein Probe-Call pro Anbieter-Endpunkt. 'Benutzerdefiniert...' für andere IDs.",
                    "Loaded live from your Langdock workspace (once an API key is set) — one probe "
                    "call per provider endpoint. 'Custom...' for other IDs.",
                ),
            )
            if model_option == _CUSTOM_MODEL_OPTION:
                model = st.text_input(
                    tr("Modell-Name (benutzerdefiniert)", "Model name (custom)"),
                    placeholder=tr("z.B. gpt-4o-search-preview", "e.g. gpt-4o-search-preview"),
                )
            else:
                model = model_option

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

**Modell:** Aktuell `{model or '—'}`. Die Liste wird live aus deinem Workspace geladen ([Agent-API-Katalog](https://docs.langdock.com/en/developer/agents-api/agent-models) bei Websuche, sonst der Passthrough-Katalog) — bei Fehlern "Verbindung testen" nutzen.

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

**Model:** Currently `{model or '—'}`. The list is loaded live from your workspace ([Agent API catalog](https://docs.langdock.com/en/developer/agents-api/agent-models) when web search is on, the passthrough catalog otherwise) — use "Test connection" if you hit errors.

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
                questions, err = generate_questions(api_key, topic, n_questions, model, web_search=web_search)
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

    # Answers from a collection run that was stopped or interrupted. Streamlit tears
    # the collection down as soon as any button is clicked, so we land back here —
    # without this the already-paid-for answers would be unreachable.
    partial = st.session_state.get("raw_answers", [])
    if partial and not st.session_state.get("phase1_complete", False):
        st.info(tr(
            f"📥 {len(partial)} Antworten aus einem abgebrochenen Lauf sind gespeichert.",
            f"📥 {len(partial)} answers from an interrupted run are saved.",
        ))
        col_use, col_drop = st.columns(2)
        if col_use.button(
            tr(f"→ Mit diesen {len(partial)} Antworten weiter zur Analyse",
               f"→ Continue to analysis with these {len(partial)} answers"),
            type="primary",
        ):
            st.session_state.setdefault("timing", {}).setdefault("p1_calls", [])
            st.session_state.step = 4
            st.rerun()
        if col_drop.button(tr("Verwerfen und neu sammeln", "Discard and collect again")):
            st.session_state.raw_answers   = []
            st.session_state.phase1_errors = []
            st.rerun()
        st.divider()

    st.subheader(tr("Runs konfigurieren", "Configure runs"))

    col_form, col_summary = st.columns([3, 2])

    with col_form:
        # --- Features (what the API can do) -------------------------------
        st.markdown(f"**{tr('Features', 'Features')}**")
        web_search = st.checkbox(
            "🔍 " + tr("Websuche", "Web search"),
            value=cfg.get("web_search", True),
            help=tr(
                "capabilities.webSearch der Agent-API — das Modell bekommt ein echtes Such-Tool. "
                "Schaltet die Modell-Liste auf den Agent-API-Katalog um.",
                "capabilities.webSearch of the Agent API — the model gets a real search tool. "
                "Switches the model list to the Agent API catalog.",
            ),
        )

        current_model = cfg.get("model", "")
        api_key       = cfg.get("api_key", "")
        agent_catalog: list[dict] = []
        models: list[str] = []

        st.markdown(f"**{tr('Modelle', 'Models')}**")
        if web_search:
            models, agent_catalog = render_agent_model_multiselect(
                api_key, cfg.get("models") or ([current_model] if current_model else []), key_prefix="s3",
            )
            model = models[0] if models else ""
        else:
            catalog, probe_failed = list_completion_models(api_key)
            render_passthrough_probe_notice(probe_failed)
            flat    = [(p, m) for p, ms in catalog.items() for m in ms]
            if not flat:
                st.info(tr(
                    "API-Key eingeben, um die verfügbaren Modelle zu laden.",
                    "Enter an API key to load the available models.",
                ) if not api_key else tr(
                    "Keine Modell-Liste verfügbar — bitte die Modell-IDs unten manuell eingeben.",
                    "No model list available — please enter the model IDs manually below.",
                ))
            ids     = [m for _, m in flat]
            labels  = {m: f"{p}  ·  {m}" for p, m in flat}
            default = [m for m in (cfg.get("models") or [current_model]) if m in ids] or ids[:1]
            models  = st.multiselect(
                tr("Modelle für die Datensammlung", "Models for data collection"),
                options=ids,
                default=default,
                format_func=lambda i: labels.get(i, i),
                help=tr(
                    "Mehrere Modelle = jede Frage wird mit jedem Modell gesammelt, direkt vergleichbar.",
                    "Several models = every question is collected with every model, directly comparable.",
                ),
            )
            extra = st.text_input(
                tr("Zusätzliche Modell-IDs (kommagetrennt, optional)",
                   "Additional model IDs (comma-separated, optional)"),
                placeholder="z.B. gpt-4o-search-preview",
            )
            models += [m.strip() for m in extra.split(",") if m.strip() and m.strip() not in models]
            model = models[0] if models else ""

        # --- Optional extras -----------------------------------------------
        st.markdown(f"**{tr('Optionen', 'Options')}**")
        col_opt1, col_opt2 = st.columns(2)

        with col_opt1:
            # Only offered on the Agent path, and only when every selected model
            # reports supportsExtendedThinking — Langdock 400s if it can't do it.
            et_capable = {m["id"] for m in agent_catalog if m.get("supportsExtendedThinking")}
            et_missing = [m for m in models if m not in et_capable] if agent_catalog else models
            et_possible = bool(web_search and models and not et_missing)
            extended_thinking = st.checkbox(
                "🧠 " + tr("Extended Thinking", "Extended thinking"),
                value=cfg.get("extended_thinking", False) and et_possible,
                disabled=not et_possible,
                help=tr(
                    "capabilities.extendedThinking der Agent-API: das Modell denkt vor der Antwort "
                    "ausführlicher nach. Nur für Modelle, die laut GET /agent/v1/models "
                    "supportsExtendedThinking melden. Kostet mehr Tokens und Zeit; der Denkprozess "
                    "landet nicht in der Antwort.",
                    "capabilities.extendedThinking of the Agent API: the model reasons more before "
                    "answering. Only for models that report supportsExtendedThinking in "
                    "GET /agent/v1/models. Costs more tokens and time; the reasoning itself does not "
                    "appear in the answer.",
                ),
            )
            if web_search and et_missing and models:
                st.caption(tr(
                    f"Nicht verfügbar für: {', '.join(et_missing)}",
                    f"Not available for: {', '.join(et_missing)}",
                ))
            elif not web_search:
                st.caption(tr("Nur mit Websuche (Agent-API).", "Only with web search (Agent API)."))

            short_answer = st.toggle(
                "✂️ " + tr("Kurzantwort-Modus", "Short-answer mode"),
                value=cfg.get("short_answer", False),
                help=tr(
                    "LLM gibt nur Marken + einen Begründungssatz zurück. Viel weniger Tokens, schneller, günstiger.",
                    "The LLM returns only brands + one sentence of reasoning. Far fewer tokens, faster, cheaper.",
                ),
            )

        with col_opt2:
            market = st.text_input(
                "🌍 " + tr("Markt / Region", "Market / region"),
                value=cfg.get("market", ""),
                placeholder=tr("z.B. Deutschland, DACH, UK", "e.g. Germany, DACH, UK"),
                help=tr(
                    "Die Langdock-API hat KEINEN Location-Parameter (weder im Agent-Objekt noch in "
                    "capabilities). Der Markt wird deshalb über den Prompt gesteuert: das Modell "
                    "antwortet aus Sicht dieses Marktes und bevorzugt Quellen von dort. Leer lassen = "
                    "keine Vorgabe.",
                    "The Langdock API has NO location parameter (neither on the agent object nor in "
                    "capabilities). The market is therefore steered through the prompt: the model "
                    "answers from that market's perspective and prefers sources from there. Leave "
                    "empty for no constraint.",
                ),
            )
            st.caption(tr(
                f"Temperatur: {COLLECTION_TEMPERATURE} (Sammlung) / {ANALYSIS_TEMPERATURE} (Analyse)",
                f"Temperature: {COLLECTION_TEMPERATURE} (collection) / {ANALYSIS_TEMPERATURE} (analysis)",
            ))

        st.markdown(f"**{tr('Umfang', 'Scope')}**")
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

        # Call estimate. Phase 1 = one collection call per question×run×model.
        # Phase 2 batches the whole dataset regardless of how many models produced it.
        n_models         = max(len(models), 1)
        collection_calls = n_q * runs * n_models
        st.success(tr(
            f"Sammlung: **{collection_calls} Calls** ({n_q} Fragen × {runs} Runs × {n_models} Modelle). "
            f"Analyse: **1 Call** ({ANALYSIS_MODEL}, gesamter Datensatz).",
            f"Collection: **{collection_calls} calls** ({n_q} questions × {runs} runs × {n_models} models). "
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
        st.metric(tr("Modelle", "Models"), len(models))
        st.metric(tr("Sammel-Calls", "Collection calls"), n_q * runs * max(len(models), 1))
        st.metric(tr("Analyse-Calls", "Analysis calls"), 1)
        if models:
            st.caption("· " + "\n\n· ".join(models))
        active = [
            x for x in (
                "🔍 " + tr("Websuche", "Web search") if web_search else "",
                "🧠 Extended Thinking" if extended_thinking else "",
                "✂️ " + tr("Kurzantwort", "Short answer") if short_answer else "",
                f"🌍 {market}" if market.strip() else "",
            ) if x
        ]
        if active:
            st.caption(tr("Aktiv: ", "Active: ") + " · ".join(active))

    st.divider()

    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button(tr("← Zurück", "← Back")):
            st.session_state.step = 2
            st.rerun()
    with col2:
        if st.button("🚀 " + tr("Daten sammeln", "Collect data"), type="primary", disabled=not models):
            st.session_state.config["models"]            = models
            st.session_state.config["model"]             = model  # first model — default for exports
            st.session_state.config["web_search"]        = web_search
            st.session_state.config["runs"]              = runs
            st.session_state.config["delay"]             = delay
            st.session_state.config["parallel_calls"]    = parallel_calls
            st.session_state.config["max_tokens"]        = max_tokens_val
            st.session_state.config["short_answer"]      = short_answer
            st.session_state.config["extended_thinking"] = extended_thinking
            st.session_state.config["market"]            = market
            st.session_state.stop_requested              = False
            _run_phase1()


def download_with_name(label: str, data: bytes, default_name: str, mime: str, key: str) -> None:
    """
    Download button with an editable filename. Streamlit fixes the name at render
    time, so the text input has to come first — the button below then always uses
    whatever is currently in the field.
    """
    ext  = Path(default_name).suffix
    stem = st.text_input(
        tr("Dateiname", "File name"),
        value=Path(default_name).stem,
        key=f"{key}_name",
        label_visibility="collapsed",
        placeholder=Path(default_name).stem,
    )
    # Strip path separators so the name can't escape the download folder.
    safe = re.sub(r"[^\w.\- ]+", "_", (stem or Path(default_name).stem).strip()) or Path(default_name).stem
    st.download_button(label, data, f"{safe}{ext}", mime, key=f"{key}_btn", use_container_width=True)


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
    ext_thinking = cfg.get("extended_thinking", False)
    market       = cfg.get("market", "")
    # One or more models: every question is asked once per model per run, so the
    # models can be compared on identical questions within a single dataset.
    models       = cfg.get("models") or [cfg["model"]]
    lang         = st.session_state.get("lang", "de")  # captured here — worker threads can't read session_state

    total        = len(questions) * runs * len(models)
    reset_run_abort()
    reset_dead_models()
    # The list lives in session_state from the first answer on, and _record_answer
    # mutates it in place. Streamlit tears this function down mid-run whenever a
    # button is clicked (that is how "Stoppen" works), so anything kept only in a
    # local would be lost — which is exactly what made a stopped run unrecoverable.
    raw_answers: list[dict] = []
    st.session_state.raw_answers    = raw_answers
    st.session_state.phase1_complete = False
    errors       = []
    # (question_index, question, run_num, model) to retry once
    failed_tasks: list[tuple[int, str, int, str]] = []
    p1_timings: list[dict] = []

    def _record_answer(question: str, run_num: int, answer: str, usage: dict, model: str):
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
        (i, question, run_num, mdl)
        for i, question in enumerate(questions)
        for run_num in range(runs)
        for mdl in models
    ]

    def _timed_ask(question: str, mdl: str):
        """
        Wraps ask_question so the duration is measured from when a worker actually
        picks the task up. Timing it from submission instead made every queued call
        look like it took as long as the whole phase — a 60-call run with 2 workers
        reported "Ø 547s, max 1167s" for calls that time out after 240s.
        """
        t_start = time.time()
        answer, err, usage = ask_question(
            api_key, question, mdl, lang, web_search, max_tokens, short_answer,
            extended_thinking=ext_thinking, market=market,
        )
        return answer, err, usage, time.time() - t_start

    with ThreadPoolExecutor(max_workers=parallel) as executor:
        future_map = {
            executor.submit(_timed_ask, q, mdl): (i, run_num, q, mdl)
            for i, q, run_num, mdl in all_tasks
        }

        completed = 0
        for future in as_completed(future_map):
            i, run_num, question, task_model = future_map[future]
            completed += 1

            if st.session_state.get("stop_requested", False):
                status_line.warning(tr(
                    f"⏹ Gestoppt nach {len(raw_answers)} von {total} Antworten. "
                    "Analyse wird mit bisherigen Antworten fortgesetzt.",
                    f"⏹ Stopped after {len(raw_answers)} of {total} answers. "
                    "Analysis continues with the answers collected so far.",
                ))
                break

            if run_abort_reason():
                # e.g. the workspace spending limit: no later call can succeed, so
                # drop the queue instead of walking through it call by call.
                for pending in future_map:
                    pending.cancel()
                error_box.error(tr(
                    f"⛔ Lauf abgebrochen: {run_abort_reason()} — "
                    f"{len(raw_answers)} von {total} Antworten wurden gesammelt und bleiben erhalten.",
                    f"⛔ Run aborted: {run_abort_reason()} — "
                    f"{len(raw_answers)} of {total} answers were collected and are kept.",
                ))
                errors.append(run_abort_reason())
                break

            answer, err, usage, call_elapsed = future.result()
            call_times.append(call_elapsed)
            p1_timings.append({
                "question":  question,
                "run":       run_num + 1,
                "model":     task_model,
                "elapsed_s": round(call_elapsed, 3),
                "ok":        answer is not None,
            })

            bar1.progress(completed / total)
            _render_live_metrics(metrics_box, completed, total, time.time() - t_phase1, call_times)

            if not answer:
                # Collect for a single retry pass at the end of the phase rather than
                # giving up now — an exhausted-retry failure here otherwise leaves this
                # question with fewer runs than the others, biasing share-of-voice.
                failed_tasks.append((i, question, run_num, task_model))
                error_box.warning(tr(
                    f"⚠️ Frage {i+1}, Run {run_num+1} ({task_model}) fehlgeschlagen: {err} — wird am Ende erneut versucht.",
                    f"⚠️ Question {i+1}, Run {run_num+1} ({task_model}) failed: {err} — will retry at the end.",
                ))
                log.warning("No answer — Frage %d, Run %d, model=%s: %s", i+1, run_num+1, task_model, err)
            else:
                error_box.empty()
                _record_answer(question, run_num, answer, usage, task_model)
                log.info("Phase1 OK — Frage %d Run %d | model=%s | len=%d | tok_in=%d tok_out=%d | web_search=%s | sources=%d | %.2fs",
                         i+1, run_num+1, task_model, len(answer),
                         usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0),
                         usage.get("web_search_used", False), len(usage.get("sources", [])), call_elapsed)

            q_short = question[:60] + ("…" if len(question) > 60 else "")
            status_line.caption(tr(
                f"Zuletzt: Frage {i+1} · Run {run_num+1} · {task_model} · {call_elapsed:.1f}s · {q_short}",
                f"Latest: Question {i+1} · Run {run_num+1} · {task_model} · {call_elapsed:.1f}s · {q_short}",
            ))

            if parallel == 1:
                time.sleep(delay)

    # ------------------------------------------------------------------
    # Retry pass — one more attempt for calls that failed above. Only runs if the
    # user didn't stop. Anything still failing after this is recorded as an error.
    # ------------------------------------------------------------------
    if failed_tasks and not st.session_state.get("stop_requested", False) and not run_abort_reason():
        status_line.caption(tr(
            f"Wiederhole {len(failed_tasks)} fehlgeschlagene Calls …",
            f"Retrying {len(failed_tasks)} failed calls …",
        ))
        with ThreadPoolExecutor(max_workers=parallel) as retry_executor:
            retry_map = {
                retry_executor.submit(_timed_ask, q, mdl): (i, run_num, q, mdl)
                for i, q, run_num, mdl in failed_tasks
            }
            for future in as_completed(retry_map):
                i, run_num, question, task_model = retry_map[future]
                if st.session_state.get("stop_requested", False):
                    break
                if run_abort_reason():
                    for pending in retry_map:
                        pending.cancel()
                    errors.append(run_abort_reason())
                    break
                answer, err, usage, call_elapsed = future.result()
                call_times.append(call_elapsed)
                p1_timings.append({
                    "question":  question,
                    "run":       run_num + 1,
                    "model":     task_model,
                    "elapsed_s": round(call_elapsed, 3),
                    "ok":        answer is not None,
                })
                if not answer:
                    errors.append(tr(
                        f"Frage {i+1}, Run {run_num+1} ({task_model}): {err}",
                        f"Question {i+1}, Run {run_num+1} ({task_model}): {err}",
                    ))
                    log.warning("Retry failed — Frage %d, Run %d, model=%s: %s", i+1, run_num+1, task_model, err)
                else:
                    _record_answer(question, run_num, answer, usage, task_model)
                    log.info("Phase1 retry OK — Frage %d Run %d | model=%s | len=%d | %.2fs",
                             i+1, run_num+1, task_model, len(answer), call_elapsed)
        _render_live_metrics(metrics_box, len(call_times), len(call_times), time.time() - t_phase1, call_times)

    phase1_elapsed = time.time() - t_phase1
    bar1.progress(1.0)
    _render_live_metrics(metrics_box, completed, total, phase1_elapsed, call_times)
    status_line.caption(tr(
        f"Phase 1 abgeschlossen — {len(raw_answers)}/{total} Antworten in {phase1_elapsed:.1f}s",
        f"Phase 1 complete — {len(raw_answers)}/{total} answers in {phase1_elapsed:.1f}s",
    ))

    # A rejected model must be re-picked before anything else makes sense, and an
    # empty dataset would only produce a second error in the analysis step — in both
    # cases stay here instead of moving on.
    needs_attention = bool(invalid_model_id()) or not raw_answers

    st.session_state.raw_answers     = raw_answers
    st.session_state.phase1_errors   = errors
    st.session_state.stop_requested  = False
    st.session_state.phase1_complete = not needs_attention
    st.session_state.timing = {
        "phase1_s": round(phase1_elapsed, 2),
        "p1_calls": p1_timings,
        "phase2_s": 0.0,
        "p2_calls": [],
    }

    if needs_attention:
        reason = run_abort_reason() or tr(
            "bitte Modell und API-Key prüfen", "please check the model and API key",
        )
        st.error(tr(
            f"Lauf beendet mit {len(raw_answers)} von {total} Antworten — {reason}",
            f"Run ended with {len(raw_answers)} of {total} answers — {reason}",
        ))
        time.sleep(2)
        st.rerun()

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
    reset_run_abort()   # a limit hit during collection must not block a later analysis
    reset_dead_models()  # ditto for a model retired during collection — analysis runs its own model
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
    col_brands  = tr("Brands", "Brands")
    cfg_brands  = cfg.get("brands", []) or []
    raw_rows = []
    for r in raw_answers:
        row = {
            col_frage:   r["question"][:80] + ("..." if len(r["question"]) > 80 else ""),
            "Run":       r["run"],
            col_modell:  r.get("model", ""),
        }
        if cfg_brands:
            # Plain string match, no API call — "🔗" marks a brand found only in the
            # linked sources, which the answer text alone would not have revealed.
            in_answer  = find_brands_in_text(r.get("answer", ""), cfg_brands)
            in_sources = find_brands_in_sources(r.get("sources", []), cfg_brands)
            marks = [b for b in in_answer] + [f"🔗{b}" for b in in_sources if b not in in_answer]
            row[col_brands] = ", ".join(marks) if marks else "—"
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
    brands = cfg.get("brands", []) or []
    if brands:
        st.caption(tr(
            f"Die CSV enthält je Antwort, welche der vorgegebenen Brands gefunden wurden — "
            f"getrennt nach Antworttext und verlinkten Quellen ({', '.join(brands)}).",
            f"The CSV includes, per answer, which of the specified brands were found — "
            f"separately for the answer text and the linked sources ({', '.join(brands)}).",
        ))
    else:
        st.caption(tr(
            "Keine Brands vorgegeben (Automatik-Modus) — die CSV enthält die Quellen, "
            "die Brand-Zuordnung entsteht in der Analyse.",
            "No brands specified (automatic mode) — the CSV contains the sources; brand "
            "attribution happens in the analysis.",
        ))

    export_df = build_raw_export(raw_answers, brands)
    ts        = datetime.now().strftime("%Y%m%d_%H%M")
    col1, col2 = st.columns(2)
    with col1:
        download_with_name(
            "📥 " + tr("Rohdaten als CSV", "Raw data as CSV"),
            export_df.to_csv(index=False).encode("utf-8"),
            f"brand_visibility_raw_{ts}.csv",
            "text/csv",
            key="raw_csv",
        )
    with col2:
        download_with_name(
            "📥 " + tr("Rohdaten als JSON", "Raw data as JSON"),
            json.dumps(raw_answers, ensure_ascii=False, indent=2).encode("utf-8"),
            f"brand_visibility_raw_{ts}.json",
            "application/json",
            key="raw_json",
        )

    if brands:
        hit_rows = int(export_df["brand_found"].sum()) if not export_df.empty else 0
        via_src  = int((export_df["brands_in_sources"] != "").sum()) if not export_df.empty else 0
        st.caption(tr(
            f"Treffer: {hit_rows} von {len(export_df)} Antworten enthalten mindestens eine vorgegebene "
            f"Brand — davon {via_src} über die verlinkten Quellen.",
            f"Hits: {hit_rows} of {len(export_df)} answers contain at least one specified brand — "
            f"{via_src} of them via the linked sources.",
        ))

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

    ts     = datetime.now().strftime("%Y%m%d_%H%M")
    brands = cfg.get("brands", []) or []

    with col1:
        if not df.empty:
            download_with_name(
                "📥 " + tr("Analyse als CSV", "Analysis as CSV"),
                df.to_csv(index=False).encode("utf-8"),
                f"brand_visibility_analysis_{ts}.csv",
                "text/csv",
                key="analysis_csv",
            )

    with col2:
        download_with_name(
            "📥 " + tr("Rohdaten als JSON", "Raw data as JSON"),
            json.dumps(results, ensure_ascii=False, indent=2).encode("utf-8"),
            f"brand_visibility_raw_{ts}.json",
            "application/json",
            key="analysis_json",
        )

    with col3:
        # Saves in brand_monitor.py format — compatible with analyze_csv.py
        local_name = st.text_input(
            tr("Dateiname", "File name"),
            value=f"brand_visibility_{ts}",
            key="local_save_name",
            label_visibility="collapsed",
        )
        if st.button("💾 " + tr("Lokal speichern (results/)", "Save locally (results/)"),
                     use_container_width=True):
            filepath = save_csv(results, cfg.get("model", ""), brands, local_name)
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
