# Code-Erklärung: `app.py` im Detail

> Vollständige, zeilennahe Erklärung des Scripts — für Leserinnen und Leser **ohne Python-Kenntnisse**.
> Zurück zur [README](../README.md) · Begriffe nachschlagen im [Glossar](GLOSSAR.md)

`app.py` ist rund **4.280 Zeilen** lang und besteht aus einer einzigen Datei. Dieses Dokument geht sie Abschnitt für Abschnitt durch: Was steht dort, was bewirkt es, und warum wurde es so gebaut. Verweise wie *(Zeile 863)* zeigen die Fundstelle im Code.

**Wichtig:** Sie müssen dieses Dokument nicht am Stück lesen. Nutzen Sie das Inhaltsverzeichnis, um zu der Stelle zu springen, die Sie interessiert. Unbekannte Fachbegriffe stehen im [Glossar](GLOSSAR.md).

---

## Inhaltsverzeichnis

**[1. Die Landkarte: Wie das Script aufgebaut ist](#1-die-landkarte-wie-das-script-aufgebaut-ist)**

**2. Der Code im Detail**

| | Abschnitt | | Abschnitt |
|---|---|---|---|
| [2.1](#21-kopf-und-imports) | Kopf und Imports | [2.17](#217-eine-frage-stellen) | Eine Frage stellen |
| [2.2](#22-logging) | Logging | [2.18](#218-textwerkzeuge) | Textwerkzeuge |
| [2.3](#23-support-protokoll) | Support-Protokoll | [2.19](#219-die-json-reparatur) | Die JSON-Reparatur |
| [2.4](#24-seitenkonfiguration) | Seitenkonfiguration | [2.20](#220-die-markenanalyse--phase-2-im-detail) | Die Markenanalyse — Phase 2 im Detail |
| [2.5](#25-das-gedächtnis-session-state) | Das Gedächtnis: Session State | [2.21](#221-ergebnisse-in-eine-tabelle-bringen) | Ergebnisse in eine Tabelle bringen |
| [2.6](#26-zweisprachigkeit) | Zweisprachigkeit | [2.22](#222-markenerkennung-ohne-ki) | Markenerkennung ohne KI |
| [2.7](#27-api-adressen-und-grundeinstellungen) | API-Adressen und Grundeinstellungen | [2.23](#223-export) | Export |
| [2.8](#28-umgang-mit-modell-akzeptiert-keine-temperatur) | Umgang mit „Modell akzeptiert keine Temperatur" | [2.24](#224-die-tutorials) | Die Tutorials |
| [2.9](#29-tote-modelle) | Tote Modelle | [2.25](#225-oberfläche-schritt-1) | Oberfläche Schritt 1 |
| [2.10](#210-die-notbremse-circuit-breaker) | Die Notbremse („Circuit Breaker") | [2.26](#226-oberfläche-schritt-2) | Oberfläche Schritt 2 |
| [2.11](#211-batch-einstellungen-für-die-analyse) | Batch-Einstellungen für die Analyse | [2.27](#227-oberfläche-schritt-3) | Oberfläche Schritt 3 |
| [2.12](#212-die-zwei-modellkataloge--das-komplizierteste-thema-im-script) | Die zwei Modellkataloge — das komplizierteste Thema im Script | [2.28](#228-phase-1--der-sammellauf) | Phase 1 — der Sammellauf |
| [2.13](#213-call_langdock--das-herzstück) | `call_langdock()` — das Herzstück | [2.29](#229-phase-2--die-analyse) | Phase 2 — die Analyse |
| [2.14](#214-call_langdock_agent--der-websuche-weg) | `call_langdock_agent()` — der Websuche-Weg | [2.30](#230-oberfläche-schritt-4--rohdaten) | Oberfläche Schritt 4 — Rohdaten |
| [2.15](#215-verbindungstest) | Verbindungstest | [2.31](#231-oberfläche-schritt-5--ergebnisse) | Oberfläche Schritt 5 — Ergebnisse |
| [2.16](#216-fragen-generieren) | Fragen generieren | [2.32](#232-der-router) | Der Router |

**[3. Der Datenfluss auf einen Blick](#3-der-datenfluss-auf-einen-blick)**

**[4. Anhang: Die Nachbardatei `langdock_evidence.py`](#4-anhang-die-nachbardatei-langdock_evidencepy)**

> **Wenn Sie wenig Zeit haben:** Die drei fett markierten Abschnitte 2.12, 2.13/2.14 und 2.28 enthalten den Kern des Programms und die meiste gesammelte Erfahrung.

---

## 1. Die Landkarte: Wie das Script aufgebaut ist

`app.py` ist **eine einzige Datei**. Das ist bei Streamlit üblich, macht die Datei aber lang. Sie ist in klar getrennte Blöcke gegliedert, die immer mit einer Kommentarzeile aus Bindestrichen beginnen:

```python
# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
```

Grobe Aufteilung:

| Zeilen | Block | Was dort passiert |
|---|---|---|
| 1–30 | Kopf & Imports | Welche Bibliotheken werden geladen |
| 32–60 | Logging & Support-Protokoll | Zwei getrennte Protokolldateien |
| 62–132 | Grundgerüst | Seitenlayout, Gedächtnis, Sprachumschaltung |
| 134–330 | Konfiguration | API-Adressen, Zeitlimits, Temperaturen, Fehlerregeln |
| 333–855 | Modellkataloge | Welche Modelle gibt es? Auswahl-Widgets |
| 856–1420 | **API-Kern** | Die zwei Funktionen, die alle Anfragen abwickeln |
| 1422–1650 | Fragen & Antworten | Fragen generieren, Fragen stellen |
| 1651–1781 | Textwerkzeuge | Aufräumen, JSON reparieren |
| 1782–2077 | **Analyse** | Marken- und Sentiment-Erkennung |
| 2080–2225 | Auswertung & Export | Tabellen bauen, CSV schreiben |
| 2227–2300 | Tutorials | Die Hilfetexte je Schritt |
| 2307–2645 | Oberfläche Schritte 1–2 | |
| 2647–2935 | Oberfläche Schritt 3 | |
| 2938–3230 | **Phase 1** | Der Sammellauf |
| 3236–3380 | **Phase 2** | Der Analyselauf |
| 3383–4265 | Oberfläche Schritte 4–5 | Rohdaten und Ergebnisse |
| 4269–4280 | Router | Welche Seite wird angezeigt |

### Namenskonventionen im Code

Damit Sie sich orientieren können:

| Muster | Bedeutung | Beispiel |
|---|---|---|
| `GROSSBUCHSTABEN` | Feste Einstellung, ändert sich nie zur Laufzeit | `MAX_TOKENS`, `ANALYSIS_MODEL` |
| `_unterstrich_am_anfang` | Interne Hilfsfunktion — nur innerhalb dieser Datei gedacht | `_strip_html()`, `_parse_json_array()` |
| `render_…` | Zeichnet etwas auf den Bildschirm | `render_step1()`, `render_tutorial()` |
| `…_RE` | Ein Regex-Suchmuster | `_MODEL_ERROR_RE`, `_CITATION_RE` |
| `tr(…)` | Übersetzung Deutsch/Englisch | `tr("Fragen", "Questions")` |

---

## 2. Der Code im Detail

### 2.1 Kopf und Imports

<sub>📍 Code: Zeilen 1–30</sub>

```python
import csv, json, logging, os, random, re, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from langdock_evidence import EvidenceRecorder, key_fingerprint, redact, response_headers
```

**Was ist ein Import?** Das Einbinden fertiger Werkzeugkästen, damit man das Rad nicht neu erfinden muss.

Die erste Gruppe gehört zu Python selbst (CSV-Dateien schreiben, JSON verarbeiten, Protokoll führen, Umgebungsvariablen lesen, Zufallszahlen, Regex, Zeitmessung). Die zweite Gruppe sind die vier extern installierten Bibliotheken. Die letzte Zeile lädt Funktionen aus der **Nachbardatei `langdock_evidence.py`**, die im selben Ordner liegt.

### 2.2 Logging

<sub>📍 Code: Zeilen 32–42</sub>

```python
log = logging.getLogger("brand_visibility")
if not log.handlers:
    log.setLevel(logging.INFO)
    _fh = logging.FileHandler("brand_visibility.log")
    ...
```

**Logging** heißt: Das Programm schreibt mit, was es tut, in die Datei `brand_visibility.log`. Beispielzeile:

```
2026-08-18 10:22:33,637 [ERROR] HTTP error 400: Invalid model, available models are: …
```

Das ist die erste Anlaufstelle, wenn etwas schiefgeht.

**Die Zeile `if not log.handlers:` ist wichtiger, als sie aussieht.** Streamlit führt das Script bei jedem Klick komplett neu aus. Ohne diese Abfrage würde bei jedem Klick ein weiterer Schreiber angehängt — nach zehn Klicks stünde jede Logzeile elfmal in der Datei. Die Abfrage bedeutet: „Nur einrichten, wenn noch nichts eingerichtet ist."

> **Fallstrick:** Die Logdatei wächst unbegrenzt. Im Repository lag sie bereits bei **2,1 MB**. Sie wird nie automatisch gekürzt oder rotiert. Bei intensiver Nutzung sollte man sie regelmäßig löschen oder archivieren.

### 2.3 Support-Protokoll

<sub>📍 Code: Zeilen 45–60 + Datei `langdock_evidence.py`</sub>

```python
evidence = EvidenceRecorder("support_evidence.jsonl")
```

Dies ist ein **zweites, ganz anders geartetes Protokoll**. Der Hintergrund ist eine reale Support-Geschichte:

Die Langdock Agent-API lehnte gelegentlich Modelle ab, die sie im selben Moment noch selbst als verfügbar gemeldet hatte. Der Langdock-Support konnte damit nichts anfangen, weil er ein **exaktes, ungekürztes Paar** brauchte:
1. die vollständige Antwort auf „Welche Modelle gibt es?" (`GET /agent/v1/models`),
2. die unmittelbar danach gesendete Anfrage mit ihrem **exakten** Inhalt und der **unveränderten** Fehlerantwort,
3. den Beweis, dass beides mit demselben Schlüssel passierte.

Das normale Log liefert nichts davon: Es kürzt Fehlermeldungen auf 600 Zeichen, verwirft die Modellliste nach dem Auswerten und schreibt den Anfrageinhalt gar nicht auf.

Deshalb schreibt `EvidenceRecorder` in eine **JSONL-Datei** (= eine JSON-Struktur pro Zeile) mit zwei eisernen Regeln:

| Regel | Umsetzung |
|---|---|
| Der API-Key wird **niemals** geschrieben | Stattdessen ein `key_fingerprint`: `sha256:a1b2c3d4e5f6/…7h9k`. Der Hash beweist „selber Schlüssel", verrät ihn aber nicht. |
| Antwortinhalte werden **wörtlich** übernommen | Kein Kürzen, kein Verschönern — genau das machte das alte Log unbrauchbar. |

**Was ist ein Hash?** Eine Einwegberechnung: Aus dem Schlüssel wird eine feste Zeichenfolge. Derselbe Schlüssel ergibt immer denselben Hash, aber aus dem Hash lässt sich der Schlüssel nicht zurückrechnen.

> ⚠️ **Datenschutz-Hinweis:** In `support_evidence.jsonl` landet der **komplette Anfrageinhalt** — also auch Ihre Fragen und Prompts. Der Code weist selbst darauf hin (Zeile 1210). Bevor Sie diese Datei an einen externen Support schicken: **hineinschauen**.

### 2.4 Seitenkonfiguration

<sub>📍 Code: Zeilen 62–69</sub>

```python
st.set_page_config(page_title="LLM Brand Visibility", page_icon="📊", layout="wide")
```

Browser-Tab-Titel, Icon, und `layout="wide"` für die volle Bildschirmbreite — bei so vielen Diagrammen sinnvoll.

### 2.5 Das Gedächtnis: Session State

<sub>📍 Code: Zeilen 71–103</sub>

Der **wichtigste Abschnitt zum Verständnis von Streamlit**.

```python
def init_state():
    defaults = {
        "step":            1,
        "config":          {},
        "questions":       [],
        "raw_answers":     [],
        "phase1_errors":   [],
        "results":         [],
        "unlisted_brands": [],
        "analysis_summary": "",
        "stop_requested":  False,
        "phase1_complete": False,
        "timing":          {},
        "lang":            "de",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
```

Weil Streamlit bei jedem Klick alles neu ausführt, wären normale Variablen sofort wieder weg. Der `session_state` ist der einzige Ort, der einen Klick überlebt.

Die entscheidende Zeile ist `if k not in st.session_state:` — **nur setzen, wenn noch nicht vorhanden**. Sonst würde jeder Klick alle Zwischenergebnisse auf Null zurücksetzen.

Was worin steckt:

| Schlüssel | Inhalt |
|---|---|
| `step` | Auf welcher der fünf Seiten wir sind |
| `config` | Alle Einstellungen: Key, Modell, Thema, Marken, Runs, … |
| `questions` | Die Liste der Fragen |
| `raw_answers` | **Phase-1-Ergebnis** — nur Frage/Antwort, noch keine Markenanalyse |
| `results` | **Phase-2-Ergebnis** — mit Marken und Sentiment |
| `unlisted_brands` | Marken, die das Modell fand, die aber nicht auf Ihrer Liste stehen |
| `analysis_summary` | Der zusammenfassende Text der KI |
| `stop_requested` | Wurde „Stoppen" gedrückt? |
| `phase1_complete` | War der Sammellauf vollständig? Wenn `False`, aber Antworten da sind, wurde ein Lauf abgebrochen — und die App bietet an, sie weiterzuverwenden. |
| `timing` | Alle Zeitmessungen für den Laufzeit-Tab |
| `lang` | `"de"` oder `"en"` |

**`reset_process()`** (Zeile 97) löscht alles außer der Spracheinstellung und startet bei Schritt 1. Bewusst wird `lang` verschont — sonst müsste man nach jedem Neustart die Sprache erneut wählen.

> ⚠️ **Wichtig für Anwender:** Der Session State liegt nur im Arbeitsspeicher. **Browser-Tab schließen = alle Daten weg.** Es gibt kein Auto-Save. Deshalb: In Schritt 4 und 5 immer exportieren.

### 2.6 Zweisprachigkeit

<sub>📍 Code: Zeilen 105–132</sub>

```python
def tr(de: str, en: str, lang: str | None = None) -> str:
    if lang is None:
        lang = st.session_state.get("lang", "de")
    return de if lang == "de" else en
```

`tr` steht für *translate*. Der Aufruf `tr("Fragen", "Questions")` liefert je nach eingestellter Sprache das eine oder das andere.

Bewusst wurde **kein** zentrales Übersetzungslexikon verwendet. Vorteil: Man sieht direkt im Code, was dort steht. Nachteil: Übersetzungen sind über die ganze Datei verstreut.

**Der `lang`-Parameter ist technisch notwendig.** In den parallelen Arbeitssträngen (Threads) ist der Session State nicht zugänglich — Streamlit gibt dort eine Warnung aus („missing ScriptRunContext") und würde die Sprache nicht kennen. Deshalb wird die Sprache **vor** dem Start der parallelen Arbeit ausgelesen und dann als Parameter durchgereicht (siehe Zeile 3000).

Der Umschalter selbst (`render_language_switch`, Zeile 119) sitzt rechts oben, umgesetzt über zwei Spalten im Verhältnis 6:1 — die breite linke Spalte ist nur ein Platzhalter, der das Auswahlfeld nach rechts drückt.

### 2.7 API-Adressen und Grundeinstellungen

<sub>📍 Code: Zeilen 134–171</sub>

```python
LANGDOCK_REGION     = os.environ.get("LANGDOCK_REGION", "eu")
LANGDOCK_URL        = f"https://api.langdock.com/openai/{LANGDOCK_REGION}/v1/chat/completions"
ANTHROPIC_URL       = f"https://api.langdock.com/anthropic/{LANGDOCK_REGION}/v1/messages"
GOOGLE_URL_TEMPLATE = "https://api.langdock.com/google/" + LANGDOCK_REGION + "/v1beta/models/{model}:generateContent"
AGENT_URL           = "https://api.langdock.com/agent/v1/chat/completions"
AGENT_MODELS_URL    = "https://api.langdock.com/agent/v1/models"
```

**Der wichtigste Punkt hier: Es gibt zwei grundverschiedene Wege.**

| | **Passthrough** (Durchreiche) | **Agent-API** |
|---|---|---|
| Wann? | Websuche **aus** | Websuche **an** |
| Adressen | Drei — eine je Anbieter | Eine für alle |
| Region in der Adresse | Ja | Nein |
| Echte Websuche | ❌ | ✅ |
| Token-Zählung | ✅ | ❌ (zeigt immer 0) |
| `max_tokens` steuerbar | ✅ | ❌ |
| Übertragung | Komplett auf einmal | Gestreamt (stückweise) |
| Modellnamen | z. B. `claude-opus-5` | z. B. `eu.anthropic.claude-opus-4-7` |

Jeder Anbieter erwartet ein **anderes Anfrageformat** — Google will `contents` und `generationConfig`, Anthropic will `max_tokens`, OpenAI will `max_completion_tokens`. Deshalb ist die zentrale Aufruffunktion so umfangreich: Sie muss drei Dialekte übersetzen.

#### Die Zeitlimits

```python
REQUEST_TIMEOUT      = 180   # Passthrough
AGENT_STREAM_TIMEOUT = 240   # Agent-API mit Websuche
```

**Timeout** = Wie lange wird auf eine Antwort gewartet, bevor abgebrochen wird. Beide Werte stammen aus Praxiserfahrung, die im Code als Kommentar dokumentiert ist:
- 180 s für Passthrough, weil Reasoning-Modelle regelmäßig 30–56 s brauchen und unter Last stark ausschlagen. 120 s führte zu vermeidbaren Abbrüchen.
- 240 s für die Agent-API, weil Websuche (suchen → Seiten lesen → antworten) einfach länger dauert.

#### Die Token-Budgets

```python
MAX_TOKENS          = 8000   # normale Antworten
QUESTION_MAX_TOKENS = 16000  # Fragengenerierung
```

8.000 klingt viel für eine Antwort — das liegt an den Reasoning-Modellen, deren internes Nachdenken aus demselben Budget bezahlt wird. Bei zu kleinem Budget kommt **gar keine** Antwort zurück (siehe Glossar).

#### Das Analyse-Modell

```python
ANALYSIS_MODEL = "claude-opus-4-8"
```

**Eine bewusste Designentscheidung mit großer Wirkung:** Egal mit welchem Modell die Antworten gesammelt wurden — die Markenerkennung läuft **immer** über dasselbe starke Modell. Grund: Die Beurteilung soll zwischen verschiedenen Läufen vergleichbar sein und nicht die Extraktionsfehler schwächerer Modelle erben.

> ⚠️ **Fallstrick:** Dieser Name ist **fest im Code eingetragen**. Wenn `claude-opus-4-8` in Ihrem Workspace nicht freigeschaltet ist oder Langdock die ID ändert, schlägt **Phase 2 komplett fehl** — obwohl Phase 1 einwandfrei lief. Die gesammelten Rohdaten bleiben aber erhalten und exportierbar. Zum Ändern: Zeile 159 anpassen.

### 2.8 Umgang mit „Modell akzeptiert keine Temperatur"

<sub>📍 Code: Zeilen 173–186</sub>

```python
_TEMPERATURE_UNSUPPORTED: set[str] = set()
```

Manche neuere Modelle — darunter ausgerechnet das Analyse-Modell — lehnen den Temperatur-Parameter mit Fehler 400 ab („`temperature` is deprecated for this model").

Das Tool **lernt das zur Laufzeit**: Beim ersten solchen Fehler wird der Modellname in dieser Liste vermerkt; alle weiteren Aufrufe lassen den Parameter dann von vornherein weg. So kostet die Eigenheit **einen** Fehlversuch statt einen pro Aufruf.

**Was ist ein `set`?** Eine Menge ohne Reihenfolge und ohne Duplikate — ideal für Ja/Nein-Merklisten wie diese.

### 2.9 Tote Modelle

<sub>📍 Code: Zeilen 188–231</sub>

```python
_dead_models: set[str] = set()

def mark_model_dead(model_id): _dead_models.add(model_id)
def is_model_dead(model_id):   return model_id in _dead_models
```

Wenn ein Modell endgültig abgelehnt wurde, wird es für den Rest des Laufs übersprungen — sofort und ohne weitere Anfrage.

**Der wichtige Teil steht im Kommentar (Zeilen 202–207):** Diese Sperre gilt **pro Modell**, nicht für den ganzen Lauf. Früher brach ein einziges veraltetes Modell den kompletten Lauf ab, sodass ein Zwei-Modell-Vergleich total ausfiel, obwohl das zweite Modell einwandfrei funktionierte. Jetzt läuft der Rest weiter.

### 2.10 Die Notbremse („Circuit Breaker")

<sub>📍 Code: Zeilen 233–309</sub>

Ein Abschnitt, der aus echtem Schmerz entstanden ist. Der Kommentar erzählt die Geschichte:

> Manche Fehler kommen als „429 — zu viele Anfragen", sind aber für den Rest des Laufs **dauerhaft**. Der Fall, der wirklich zugeschlagen hat: ein erreichtes **Ausgabelimit des Workspace**. Jeder der 60 Aufrufe verbrannte vier Versuche mit 15/30/60/120 Sekunden Wartezeit — ein Lauf, der niemals erfolgreich sein konnte, schlief **rund 45 Minuten** vor sich hin und wirkte eingefroren.

Die Lösung:

```python
_FATAL_LIMIT_RE = re.compile(
    r"spending limit|monthly limit|quota|billing|credit balance|budget|insufficient[_ ]",
    re.IGNORECASE,
)
```

Dieses Suchmuster erkennt in der Fehlermeldung, ob es sich um ein **endgültiges** Limit handelt (Budget aufgebraucht) oder um ein **vorübergehendes** (zu schnell gefragt). Nur beim ersten Typ wird die Notbremse gezogen: Der erste betroffene Aufruf hinterlegt den Grund, alle anderen brechen dann sofort ab.

Das Suchmuster ist **absichtlich eng gefasst** — das normale Tempolimit („exceeded the maximum number of tokens per minute") muss weiter wiederholt werden dürfen, denn das löst sich von selbst.

```python
def _sleep_unless_aborted(seconds: float) -> bool:
    deadline = time.time() + seconds
    while True:
        remaining = deadline - time.time()
        if remaining <= 0:      return True
        if _run_abort_reason is not None:  return False
        time.sleep(min(0.5, remaining))
```

Diese Funktion ist ein **unterbrechbarer Schlaf**. Statt 120 Sekunden am Stück zu schlafen, schläft sie in 0,5-Sekunden-Häppchen und prüft dazwischen, ob inzwischen jemand die Notbremse gezogen hat. So sitzt kein Arbeitsstrang zwei Minuten lang eine Wartezeit ab, die längst sinnlos ist.

### 2.11 Batch-Einstellungen für die Analyse

<sub>📍 Code: Zeilen 311–330</sub>

```python
ANALYSIS_BATCH_MAX_ANSWERS      = 40
ANALYSIS_BATCH_MAX_INPUT_TOKENS = 40000
ANALYSIS_ANSWER_CHARS           = 2000
ANALYSIS_ANSWER_HEAD_FRAC       = 0.7
```

**Was ist ein Batch?** Eine Portion. Statt alle 200 Antworten in einer riesigen Anfrage zu analysieren, werden sie in handliche Pakete aufgeteilt.

**Warum?** Wenn eine einzelne Analyse-Anfrage zu groß wird, reicht das Antwortbudget nicht, das Modell wird mitten in seiner JSON-Ausgabe abgeschnitten — und dann ist die Ausgabe unlesbar. **Ergebnis: Die komplette Analyse geht verloren.** Mit Batches verliert man höchstens ein Paket.

Ein Batch endet, sobald **eine** der beiden Grenzen erreicht ist: 40 Antworten oder geschätzte 40.000 Eingabe-Tokens.

**`ANALYSIS_ANSWER_CHARS = 2000`**: Jede Antwort wird für die Analyse auf 2.000 Zeichen gekürzt.

**`ANALYSIS_ANSWER_HEAD_FRAC = 0.7`** — ein cleveres Detail. Gekürzt wird nicht einfach vorne, sondern **70 % Anfang + 30 % Ende**:

```
[erste 1400 Zeichen] […] [letzte 600 Zeichen]
```

Grund: Bei Antworten vom Typ „Die 15 besten Werkzeuge" stehen die Marken 8 bis 15 **am Ende**. Ein reines Abschneiden nach vorne würde sie systematisch unsichtbar machen und die Sichtbarkeitsmessung verzerren.

> ⚠️ **Bleibt trotzdem ein Fallstrick:** Bei sehr langen Antworten geht die Mitte verloren. Wer das vermeiden will, kann `ANALYSIS_ANSWER_CHARS` erhöhen — zahlt das aber mit mehr Tokens und mehr Batches. Alternativ hilft der **Kurzantwort-Modus**, der von vornherein kompakte Antworten erzeugt.

### 2.12 Die zwei Modellkataloge — das komplizierteste Thema im Script

<sub>📍 Code: Zeilen 333–855</sub>

Wenn Sie nur einen Abschnitt dieser Doku lesen, dann diesen: Hier steckt die meiste Erfahrung und hier entstehen die meisten Fehlermeldungen.

#### Das Grundproblem

Um ein KI-Modell anzusprechen, braucht man seinen **exakten technischen Namen**. Diese Namen sind bei Langdock keine schönen Bezeichnungen, sondern **Deployment-IDs** — interne Installationsnamen, die sich ändern können:

```
claude-opus-4-6-v1
eu.anthropic.claude-opus-4-7
gpt-5-mini-eu
claude-haiku-4-5@20251001
claude-opus-5@default
```

**Und es gibt zwei getrennte Kataloge**, die sich zwar überlappen, aber unterschiedliche Schreibweisen verwenden.

#### Katalog A — Agent-API (Websuche an)

Hier ist es einfach: Es gibt eine offizielle Abfrage.

```python
@st.cache_data(ttl=AGENT_MODELS_TTL, show_spinner=False)
def fetch_agent_models(api_key: str) -> tuple[list[dict], str | None]:
    r = requests.get(AGENT_MODELS_URL, headers={"Authorization": f"Bearer {api_key}"}, timeout=15)
    ...
```

`GET /agent/v1/models` liefert genau die Modelle, die Ihr Workspace über die Agent-API nutzen darf. Die Zeile `@st.cache_data(ttl=60)` ist ein **Dekorator** — eine Anweisung an die Funktion darüber. Sie bedeutet: „Merke dir das Ergebnis 60 Sekunden lang." Ohne sie würde bei jedem Klick erneut abgefragt.

Langdock hat dazu ausdrücklich mitgeteilt (im Code als Kommentar festgehalten, Zeilen 467–478):
> Die `id` muss **unverändert** gesendet werden. Die IDs sind Deployment-Namen und können sich ändern. Es gibt keine garantierten Alternativnamen.

Deshalb die eiserne Regel im ganzen Script: **Keine feste Modellliste im Code, kein Umschreiben von IDs, kein Raten von Alternativen.** Bei Problemen: Liste neu laden und neu auswählen.

#### Katalog B — Passthrough (Websuche aus)

Hier wird es kurios: **Diese Endpoints haben gar keine Abfrage für ihre Modellliste.** Der Trick (Zeilen 445–464):

```python
_PROBE_MODEL_ID = "__langdock_model_probe__"

def _probe_models(url, api_key, payload) -> list[str]:
    r = requests.post(url, json=payload, ...)
    if r.status_code < 400:
        return []
    return _parse_available_models(r.text)
```

Man schickt **absichtlich einen erfundenen Modellnamen**. Der Server antwortet mit Fehler 400 und schreibt dabei hilfreicherweise in seine Fehlermeldung, welche Modelle er denn akzeptieren würde:

> `Invalid model, available models are: gpt-5-mini, gpt-5.6-sol, o3, …`

Aus dieser Fehlermeldung fischt `_parse_available_models` (Zeile 412) die Namen heraus. **Der Trick kostet keine Tokens**, weil nichts erzeugt wird.

Der Sentinel-Name `__langdock_model_probe__` ist so gewählt, dass er unmöglich ein echtes Modell sein kann.

**Und es braucht drei solche Sondierungen**, eine je Anbieter (Zeile 802). Denn — hier steht Messerfahrung im Code, Zeilen 355–363 — jeder Endpoint kennt **nur seine eigenen** Modelle. Der OpenAI-Endpoint liefert kein einziges Claude-Modell. Wer nur einen abfragt, dem fehlen ganze Anbieter in der Auswahlliste.

Ein Detail in Zeile 421:
```python
if mid and not mid.endswith("-"):
    ids.append(mid)
```
Bei langen Listen kürzt die API die Aufzählung ab, sodass der letzte Eintrag ein Fragment sein kann (`claude-`). Solche Bruchstücke dürfen nicht als echtes Modell in die Auswahlliste gelangen.

#### Die große Erkenntnis: Der Agent-Katalog ist nicht in sich konsistent

Der ausführlichste Kommentar der ganzen Datei (Zeilen 365–406) dokumentiert eine Entdeckung, die viel Arbeit gekostet hat:

> **Die Agent-API wird von mehreren Servern bedient, deren Modelllisten sich widersprechen.** Eine ID, die `GET /agent/v1/models` gerade eben zurückgegeben hat, kann dem Server, der die nächste Anfrage bearbeitet, unbekannt sein.

Die Belege aus dem Log:
- Dieselbe ID `claude-haiku-4-5@20251001` am selben Endpoint: **95 × erfolgreich gegen 7 × abgelehnt.** Ein wirklich falscher Name hätte 0 Erfolge.
- Zwei Fehlermeldungen im Abstand von **einer Millisekunde** nannten unterschiedliche Kataloge:
  - `…637`: `claude-opus-4-8@default`, `claude-opus-5`, `claude-sonnet-5`
  - `…638`: `claude-opus-4-8`, `claude-opus-5@default`, `claude-sonnet-5@default`
- Betroffen sind **nur Anthropic-Modelle**. Alle 22 Nicht-Anthropic-Modellfamilien blieben stabil.

**Zwei Konsequenzen, beide im Code umgesetzt:**

**1. Wiederholen statt aufgeben** (Zeile 1270): Wird ein Modell auf dem Agent-Weg als „nicht verfügbar" abgelehnt, wird **dieselbe ID** noch einmal geschickt — nicht eine geratene Alternative. Die Wiederholung landet mit hoher Wahrscheinlichkeit auf einem anderen Server. Erst wenn alle vier Versuche scheitern, gilt das Modell als wirklich weg.

**2. Schreibweisen abgleichen statt warnen** (`catalog_equivalent`, Zeile 572):

```python
def catalog_equivalent(model_id: str, ids: list[str]) -> str | None:
    if model_id in ids:
        return model_id
    want = _normalize_model_id(model_id)
    for cid in ids:
        if _normalize_model_id(cid) == want:
            return cid
    return None
```

Wenn Ihre gemerkte Auswahl im frisch geladenen Katalog fehlt, ist das meist nur eine Schreibvariante. Die Funktion sucht dann nach dem „gleichen" Modell in der aktuellen Schreibweise. Vorher warnte die App bei jeder solchen Variante — mit dem Effekt, dass die Nutzer die Warnung ignorierten und sie auch bei einer **echten** Modellentfernung übersahen.

**Ganz wichtig, und im Code ausdrücklich betont:** Es wird **nie** ein Name erfunden. Der Abgleich entscheidet nur, **welcher** Eintrag aus dem echten, lebenden Katalog gemeint ist. Gesendet wird immer eine Zeichenkette, die der Katalog selbst geliefert hat.

#### Die Normalisierung *(Zeile 427)*

```python
def _normalize_model_id(model: str) -> str:
    m = model.strip().lower()
    m = m.split("@", 1)[0]                                # claude-opus-4-7@default → claude-opus-4-7
    m = re.sub(r"^(eu|us|apac|global)\.", "", m)          # eu.anthropic.… → anthropic.…
    m = re.sub(r"^(anthropic|openai|google|meta|amazon)\.", "", m)
    m = re.sub(r"-(eu|us|apac|global)$", "", m)           # gpt-5-mini-eu → gpt-5-mini
    m = re.sub(r"-\d{8}$", "", m)                         # …-20251001 → …
    m = re.sub(r"-v\d+(:\d+)?$", "", m)                   # …-v1 → …
    return m
```

Diese Funktion schält alle Verzierungen ab, damit man erkennen kann, ob zwei unterschiedlich geschriebene IDs dasselbe Modell meinen. Sie wird **nur zum Vergleichen** benutzt, **nie zum Senden**.

Sie liefert außerdem die Anbieter-Zuordnung für die Anzeige (`_model_provider_label`, Zeile 537): Beginnt der bereinigte Name mit `claude-` → Anthropic, mit `gemini-` → Google, mit `gpt-`/`o1`/`o3`/`o4` → OpenAI, enthält er `llama` → Meta, sonst → „Sonstige".

#### Die Auswahl-Widgets *(Zeilen 596–855)*

Vier Bausteine, die auf dem Obigen aufbauen:

| Funktion | Zeile | Zweck |
|---|---|---|
| `render_agent_model_picker` | 613 | Ein Modell wählen (Schritt 1) |
| `render_agent_model_multiselect` | 707 | Mehrere Modelle wählen (Schritt 3) |
| `list_completion_models` | 822 | Passthrough-Katalog per Sondierung, nach Anbieter gruppiert |
| `render_passthrough_probe_notice` | 596 | Meldet, welche Anbieter nicht abgefragt werden konnten |

Alle enthalten einen **🔄-Button** zum Neuladen. Er löscht den Zwischenspeicher (`fetch_agent_models.clear()`), sodass die nächste Abfrage wirklich beim Server landet.

Beachtenswert in `list_completion_models` (Zeile 822): Der Kommentar erklärt, warum dort **Anbieternamen und keine übersetzten Texte** stehen — das Ergebnis wird zwischengespeichert und würde sonst die Sprache einfrieren, in der es zuerst geladen wurde.

> ⚠️ **Fallstricke bei der Modellauswahl:**
> - Schaltet man die Websuche um, **wechselt die komplette Modellliste**. Eine ID aus dem einen Katalog funktioniert im anderen unter Umständen nicht.
> - Gibt es keine Modellliste (Netzproblem, Key falsch), fällt die App auf ein **freies Textfeld** zurück. Dort muss man den Namen exakt kennen.
> - Der Zwischenspeicher ist an den API-Key gekoppelt. Ein anderer Key = eine andere Liste.

### 2.13 `call_langdock()` — das Herzstück

<sub>📍 Code: Zeile 863</sub>

**Jede** Anfrage an ein KI-Modell läuft durch diese eine Funktion. Sie ist ca. 250 Zeilen lang und macht sehr viel — deshalb hier Schritt für Schritt.

#### Was sie zurückgibt

```python
return (response_text, error_message, usage)
```

Immer ein **Dreier-Paket**:
- Bei Erfolg: `(Text, None, {"prompt_tokens": 120, "completion_tokens": 800, …})`
- Bei Fehler: `(None, "API-Key ungültig oder abgelaufen (401).", {})`

**Das ist eine bewusste Designentscheidung:** Die Funktion zeigt **selbst niemals eine Fehlermeldung an**. Sie gibt sie zurück, und wer sie aufgerufen hat, entscheidet, was damit passiert. Das ist zwingend nötig, weil die Funktion auch in parallelen Arbeitssträngen läuft, wo man nichts auf den Bildschirm schreiben kann.

#### Ablauf

**① Notbremse prüfen** *(Zeilen 887–890)*
```python
if _run_abort_reason is not None:
    return None, _abort_error(lang), {}
if is_model_dead(model):
    return None, _dead_model_error(model, lang), {}
```
Zwei sofortige Abbrüche, bevor überhaupt etwas gesendet wird.

**② Websuche? Dann woanders hin** *(Zeile 892)*
```python
if web_search:
    return call_langdock_agent(...)
```
Die Weiche zur Agent-API. Der Rest der Funktion behandelt nur noch den Passthrough-Weg.

**③ Den richtigen Dialekt wählen** *(Zeilen 907–949)*

Je nach Anbieter wird ein anderes Anfragepaket gebaut:

| Anbieter | Adresse | Struktur |
|---|---|---|
| Anthropic | `ANTHROPIC_URL` | `{"model", "messages", "max_tokens"}` |
| Google | `GOOGLE_URL_TEMPLATE` | `{"contents": [{"role", "parts"}], "generationConfig"}` |
| Alle anderen | `LANGDOCK_URL` | `{"model", "messages", "max_completion_tokens"}` |

Bei Google fällt zusätzlich auf: Die Rolle `assistant` heißt dort `model`, und der Text steckt in `parts`. Diese Umbauten stehen in Zeilen 924–938.

**④ Bis zu vier Versuche** *(Zeile 950)*
```python
for attempt in range(4):
```
Alles Folgende läuft in einer Schleife mit maximal vier Durchläufen.

**⑤ Senden und protokollieren** *(Zeilen 952–967)*
```python
t0 = time.time()
r = requests.post(url, json=payload, headers=headers, timeout=REQUEST_TIMEOUT)
elapsed = time.time() - t0
log.info("HTTP %s | attempt %d | model=%s | tokens=%d | %.2fs", ...)
```

**⑥ Antwort auspacken** *(Zeilen 972–995)*

Auch hier drei Dialekte. Der Text steckt bei Anthropic in `content[0].text`, bei Google in `candidates[0].content.parts[0].text`, bei OpenAI in `choices[0].message.content`. Dasselbe gilt für die Tokenzählung: `input_tokens`/`output_tokens` bei Anthropic, `promptTokenCount`/`candidatesTokenCount` bei Google, `usage` bei OpenAI. Alles wird auf **ein einheitliches Format** gebracht, damit der Rest des Programms sich nicht darum kümmern muss.

**⑦ Der Sonderfall „leere Antwort"** *(Zeilen 996–1013)*
```python
if not text:
    if finish in ("max_tokens", "length", "MAX_TOKENS"):
        return None, tr("Token-Budget erschöpft (max_tokens=…)", …), usage
```
Ein Reasoning-Modell, dessen Budget beim Nachdenken aufgebraucht wurde, liefert HTTP 200 — und **nichts**. Der Grund steht in `finish_reason`, und weil jeder Anbieter ihn anders nennt (`max_tokens` / `length` / `MAX_TOKENS`), werden alle drei geprüft. Ohne diese Behandlung sähe der Fehler aus wie ein Erfolg.

#### Die Fehlerbehandlung — der interessanteste Teil

**Temperatur wird abgelehnt** *(Zeilen 1031–1041)*
```python
if status == 400 and "temperature" in raw_body.lower() and temperature is not None:
    removed = payload.pop("temperature", None)
    ...
    _TEMPERATURE_UNSUPPORTED.add(model)
    continue   # nächster Versuch, jetzt ohne Temperatur
```
Der Parameter wird entfernt, das Modell für die Zukunft vermerkt, und es geht in den nächsten Versuch. Ohne diesen Griff ging früher ein kompletter Analyse-Batch verloren.

**Modell ungültig** *(Zeilen 1045–1048)*
```python
if status in (400, 422) and _MODEL_ERROR_RE.search(raw_body):
    mark_model_dead(model)
    return None, _dead_model_error(model, lang), {}
```
Auf dem Passthrough-Weg wird ein abgelehntes Modell **sofort** stillgelegt — anders als auf dem Agent-Weg, wo widersprüchliche Server der wahrscheinlichere Grund sind.

Der Kommentar ist wichtig: Früher wurde „mit dem ähnlichsten Treffer erneut versucht". Das war ein **stillschweigender Modellwechsel** — die Auswertung enthielt dann Antworten eines Modells, das der Nutzer nie ausgewählt hatte. Diese Logik wurde bewusst entfernt.

**Rate Limit (429)** *(Zeilen 1049–1062)*
```python
if status == 429:
    if _FATAL_LIMIT_RE.search(raw_body):
        abort_run(_api_message(raw_body))     # Budget → Notbremse
        return None, _abort_error(lang), {}
    wait = 15 * (2 ** attempt) + random.uniform(0, 5)   # 15-20s, 30-35s, 60-65s, 120-125s
    if not _sleep_unless_aborted(wait):
        return None, _abort_error(lang), {}
    continue
```
Die Unterscheidung, die in **Abschnitt 2.10** beschrieben wurde: dauerhaftes Limit → alles stoppen; vorübergehendes → warten und wiederholen. Die Wartezeiten müssen das rollierende Minutenfenster abdecken, deshalb starten sie bei 15 Sekunden.

**Alle übrigen Statuscodes** *(Zeilen 1064–1084)*
Werden auf verständliche zweisprachige Meldungen abgebildet — inklusive Hinweis auf die eingestellte Region beim 404 und auf den Modellnamen beim 400/422.

**Netzwerkfehler und Zeitüberschreitungen** *(Zeilen 1086–1102)*
Ein `ConnectionError` (kein Internet) wird sofort zurückgemeldet — Wiederholen hätte keinen Sinn. Ein `Timeout` wird bis zu dreimal wiederholt, mit kurzer Wartezeit von 1, 2, 4 Sekunden.

**Alles andere** *(Zeilen 1110–1113)*
```python
except Exception as e:
    err = tr(f"Unbekannter Fehler: {e}", …)
```
Ein Auffangnetz. Selbst ein völlig unerwarteter Programmfehler bringt nicht die ganze App zum Absturz, sondern wird als Fehlermeldung für diesen einen Aufruf behandelt.

### 2.14 `call_langdock_agent()` — der Websuche-Weg

<sub>📍 Code: Zeile 1134</sub>

Diese Funktion ist die einzige Möglichkeit, den Modellen eine **echte Websuche** zu geben.

#### Der Aufbau der Anfrage *(Zeilen 1163–1207)*

```python
payload = {
    "agent": {
        "name": "Brand Visibility Assistant",
        "instructions": (
            "You have a live web search tool with current results. For any question about "
            "products, brands, providers, rankings, prices, or recommendations, search the web "
            "first and base your answer on what you find — even if you feel you already know the "
            "answer, since your training data is outdated. Do not state or imply you lack "
            "real-time access; you have it. ..."
        ),
        "model":        model,
        "capabilities": {"webSearch": True},
    },
    "messages": ui_messages,
    "stream":   True,
}
```

Es wird ein **temporärer Agent** definiert — quasi ein Wegwerf-Assistent mit einer Rolle und Werkzeugen.

**Der `instructions`-Text ist entscheidend und der Grund dafür steht im Kommentar (Zeilen 1170–1174):**

> `capabilities.webSearch` stellt das Werkzeug nur **zur Verfügung** — das Modell entscheidet selbst, ob es es benutzt. Ohne ausdrücklichen Anstoß greifen Modelle (besonders kleinere wie Haiku) auf ihre antrainierte Standardantwort zurück: „Ich habe keinen Echtzeit-Zugriff." Vor allem bei Fragen, von denen sie überzeugt sind, sie lägen in der Zukunft.

Deshalb sagt die Anweisung ausdrücklich: *Du hast das Werkzeug, dein Trainingswissen ist veraltet, such zuerst, und behaupte nicht, du hättest keinen Zugriff.*

> ⚠️ **Trotzdem bleibt es eine Bitte, kein Befehl.** Deshalb prüft Schritt 4 nach, ob wirklich gesucht wurde.

#### Warum gestreamt wird

```python
"stream": True
```

Kein Komfortfeature, sondern eine Notwendigkeit (Kommentar Zeilen 1124–1128): **Langdock bricht nicht-gestreamte Agent-Anfragen nach 100 Sekunden mit Fehler 524 hart ab.** Eine Websuche (suchen → Seiten lesen → antworten) überschreitet das regelmäßig. Beim Streaming fließt die Antwort in Häppchen und die Verbindung bleibt lebendig.

#### Der Stream wird ausgewertet *(Zeilen 1310–1348)*

```python
for line in r.iter_lines(decode_unicode=True):
    if not line or not line.startswith("data:"):
        continue
    data_str = line[len("data:"):].strip()
    if data_str == "[DONE]":
        break
    event = json.loads(data_str)
    etype = event.get("type", "")
    if etype == "text-delta":
        full_text += event.get("delta", "")
    elif etype == "error":
        error_text = event.get("errorText", "Agent stream error")
    elif etype == "finish":
        break
    elif etype.startswith("tool-"):
        ...
```

Der Server schickt Zeile für Zeile kleine Ereignisse. Jede beginnt mit `data:`. Der Code sammelt:
- `text-delta` → Textstückchen, werden aneinandergehängt
- `tool-…` → das Modell hat ein Werkzeug benutzt
- `source…` → eine zitierte Quelle
- `error` → Fehler mitten im Stream
- `finish` / `[DONE]` → Ende

#### Die Quellen einsammeln *(Zeilen 1296–1309)*

```python
def _harvest_urls(obj):
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
```

Diese Funktion **ruft sich selbst auf** (das nennt man *Rekursion*) und durchwühlt beliebig verschachtelte Datenstrukturen nach allem, was wie eine Web-Adresse aussieht.

**Warum so aufwendig?** Weil jedes Modell die Suchergebnisse anders verpackt. Statt für jedes Modell eine eigene Regel zu schreiben, sucht der Code einfach überall.

#### Der Beweis, dass wirklich gesucht wurde

Das Tool sammelt **drei unabhängige Belege**:

| Beleg | Woher |
|---|---|
| **Werkzeug-Aufruf** | Ein Ereignis vom Typ `tool-…`, dessen Name „search" enthält |
| **Quellen-URLs** | Eingesammelte Links |
| **Zitat-Marker** | Im Antworttext eingebettete Kennzeichen wie `【toolu_vrtx_014m5…-5】` |

Der letzte Punkt ist besonders nützlich (Zeilen 1664–1687):

```python
_CITATION_RE = re.compile(r"[【\[]\s*(?:toolu|call|fc|tool|resp|ws)_[A-Za-z0-9_\-]+\s*[】\]]")
```

Diese Marker verweisen intern auf den Werkzeugaufruf, der die jeweilige Information geliefert hat. Sie sind ein **harter Beweis**, dass gesucht wurde — auch wenn die Stream-Ereignisse eine Form hatten, die der Code nicht erkannt hat. Gleichzeitig würden sie im angezeigten Text und in der Markenanalyse stören, also werden sie entfernt — aber vorher **gezählt**.

```python
full_text, n_citations = _strip_citation_markers(full_text)
if n_citations:
    web_search = True
```

#### Fehlerbehandlung *(Zeilen 1219–1290)*

Zusätzlich zum Bekannten (429, 400, 401, 403) gibt es zwei „Notfall-Vereinfachungen":

```python
if status == 400 and "temperature" in raw_body.lower() and payload["agent"].pop("temperature", None) is not None:
    continue
if status == 400 and "thinking" in raw_body.lower() and payload["agent"]["capabilities"].pop("extendedThinking", None) is not None:
    continue
```

Lehnt das Modell einen optionalen Zusatz ab, wird der Zusatz weggelassen und erneut versucht — statt den ganzen Aufruf zu verlieren.

Und die bereits beschriebene Sonderbehandlung: Bei „Modell nicht verfügbar" wird **dieselbe ID** bis zu dreimal wiederholt, weil es sehr wahrscheinlich nur ein widersprüchlicher Server ist.

> ⚠️ **Die wichtigste Einschränkung dieses Weges:** Die Agent-API meldet **keinen Tokenverbrauch**. In allen Tabellen und Exporten stehen bei aktiver Websuche `tokens_in = 0` und `tokens_out = 0`. Das ist kein Fehler — die Information existiert schlicht nicht. Für die Kostenkontrolle muss man das Langdock-Dashboard heranziehen. Ebenso wirkungslos ist dort der Regler „Max. Tokens" — die Oberfläche blendet ihn deshalb aus (Zeile 2826).

### 2.15 Verbindungstest

<sub>📍 Code: Zeile 1427</sub>

```python
def test_connection(api_key, model, web_search=False) -> tuple[bool, str]:
    reset_run_abort()
    reset_dead_models()
    text, err, _ = call_langdock(
        api_key,
        [{"role": "user", "content": "Say 'OK' and nothing else."}],
        model=model,
        max_tokens=MAX_TOKENS,
        web_search=web_search,
        ...
    )
```

Ein Minimal-Aufruf, der Zugangsdaten und Modellnamen prüft. Drei Details, die kein Zufall sind:

1. **`reset_run_abort()` und `reset_dead_models()`** — Der Nutzer prüft ja gerade nach, ob er ein Problem behoben hat. Frühere Sperren dürfen das Ergebnis nicht verfälschen.
2. **`max_tokens=MAX_TOKENS`, nicht ein kleiner Wert** — Mit einem winzigen Budget würde ein Reasoning-Modell scheitern und einen Fehler melden, obwohl alles in Ordnung ist. Der Test muss dieselben Bedingungen haben wie der Ernstfall.
3. **`web_search` wird durchgereicht** — Bei aktiver Websuche wird der Agent-Weg getestet, also genau der, der später auch benutzt wird.

> 💡 **Empfehlung:** Diesen Button immer vor einem größeren Lauf drücken. Er kostet einen einzigen Aufruf und erspart im Zweifel eine halbe Stunde Fehlersuche.

### 2.16 Fragen generieren

<sub>📍 Code: Zeile 1478</sub>

Der Prompt (Zeilen 1486–1507) fordert genau `n` Fragen, die
- typische Nutzerfragen an einen KI-Assistenten sind,
- verschiedene Aspekte abdecken (Empfehlungen, Vergleiche, Eigenschaften, Anwendungsfälle),
- sich inhaltlich klar unterscheiden,
- **so formuliert sind, dass die Antwort natürlicherweise Marken nennen würde.**

Der letzte Punkt ist der eigentliche Trick: „Was ist ein CRM?" hilft nicht — „Welches CRM eignet sich für 50 Mitarbeiter?" schon.

Die Antwort soll ein **JSON-Array** sein. Weil Modelle sich daran oft nicht halten, gibt es eine dreistufige Aufbereitung:

**Stufe 1 — JSON versuchen:**
```python
parsed = _parse_json_array(text)
if parsed:
    raw = [str(q) for q in parsed if isinstance(q, (str, int, float))]
else:
    raw = text.strip().splitlines()
```
Kein gültiges JSON? Dann wird der Text einfach zeilenweise gelesen.

**Stufe 2 — Aufräumen** *(Zeile 1461)*
```python
_Q_PREFIX_RE = re.compile(r'^\s*(?:\d+[.)]\s*|[-*•]\s*)+')
```
Entfernt Listenzeichen, die Modelle trotz gegenteiliger Anweisung hinzufügen: `1. `, `1) `, `- `, `* `, `• ` — und umschließende Anführungszeichen jeder Art (auch die typografischen `„ "`).

**Stufe 3 — Duplikate entfernen** *(Zeile 1467)*
```python
key = re.sub(r"\s+", " ", q).strip().lower().rstrip("?.!")
```
Der Vergleichsschlüssel ignoriert Groß-/Kleinschreibung, mehrfache Leerzeichen und abschließende Satzzeichen. „Welche CRM-Tools?" und „welche crm-tools" gelten als dieselbe Frage. Die zuerst gesehene Fassung bleibt erhalten.

**Ein hilfreiches Detail (Zeilen 1544–1554):** Wenn nach dem Aufräumen **null** Fragen übrig sind, wird der Anfang der Modellantwort mitprotokolliert:

```python
log.warning("generate_questions PARSED ZERO — … | reply[:200]=%r", ..., text[:200])
```

Vorher war so ein Fall im Log unsichtbar, weil der Aufruf selbst als sauberer HTTP 200 durchging. Jetzt kann man nachträglich unterscheiden, ob das Modell Prosa geliefert hat oder gar nichts.

> ⚠️ **Fallstrick:** Es kommen oft **weniger** Fragen zurück als angefordert — meist weil sich Fast-Duplikate gegenseitig ausgelöscht haben. Das ist kein Fehler, Schritt 2 zeigt die tatsächliche Zahl und man kann von Hand ergänzen.
>
> ⚠️ Der Parameter `web_search` muss hier zwingend mitgegeben werden. Er war früher fest auf `False` — mit der Folge, dass Agent-Modell-IDs an den Passthrough geschickt wurden, wo die anbieterpräfixierten Namen (`eu.anthropic.…`) mit Fehler 400 abgelehnt wurden.

### 2.17 Eine Frage stellen

<sub>📍 Code: Zeile 1575</sub>

```python
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
```

Bewusst schlicht gehalten: Die Antwort soll dem entsprechen, was ein normaler Nutzer bekommen würde. Jede zusätzliche Anweisung würde das Messergebnis verfälschen.

Der **Kurzantwort-Modus** ist die Ausnahme — er erkauft sich deutlich niedrigere Kosten und schnellere Läufe damit, dass die Antwort nicht mehr realistisch ist. Für reine Sichtbarkeitszählungen ist das vertretbar, für Tonalitätsanalysen weniger.

#### Der Markt-Kontext *(Zeile 1637)*

```python
def _with_market_context(content: str, market: str, lang: str) -> str:
    return (
        f"Kontext: Beantworte die Frage aus Sicht des Marktes „{market}". "
        f"Berücksichtige Anbieter, Marken und Angebote, die dort tatsächlich verfügbar sind, "
        f"und stütze dich bevorzugt auf Quellen aus diesem Markt.\n\n{content}"
    )
```

Der Kommentar erklärt die Notlage: **Die Langdock-API hat keinen Standort-Parameter** — weder im Agent-Objekt noch in den Capabilities noch laut Changelog. Der Markt kann deshalb nur über den Prompt gesteuert werden.

Trotzdem ist es eine wirksame Stellschraube: Markenempfehlungen und die Auswahl der Quellen unterscheiden sich stark je Markt.

> ⚠️ **Aber:** Es bleibt eine Bitte an das Modell. Es gibt keine Garantie, dass tatsächlich nur Quellen aus dem Zielmarkt herangezogen werden. Das ist eine schwächere Steuerung als eine echte Standorteinstellung — beim Interpretieren der Ergebnisse mitdenken.

### 2.18 Textwerkzeuge

<sub>📍 Code: Zeilen 1651–1688</sub>

**`_strip_html(text, max_len=600)`** — Entfernt HTML-Tags und mehrfache Leerzeichen aus Fehlerantworten. Manche Serverfehler (502, 503) liefern eine komplette HTML-Seite, die das Log sonst zumüllen würde.

Warum 600 und nicht 200 Zeichen? Weil Langdocks „ungültiges Modell"-Meldung die Liste der akzeptierten Modelle erst **nach etwa 100 Zeichen Vorrede** nennt. Ein Schnitt bei 200 hätte genau die Information abgeschnitten, die man zur Fehlerbehebung braucht.

**`_strip_citation_markers(text)`** — Entfernt die Zitat-Marker und zählt sie (siehe 6.14). Danach wird noch aufgeräumt:
```python
clean = re.sub(r"[ \t]+([.,;:!?])", r"\1", clean)  # " ." → "."
clean = re.sub(r"[ \t]{2,}", " ", clean)           # doppelte Leerzeichen
```
Sonst blieben nach dem Entfernen unschöne Lücken vor Satzzeichen stehen.

### 2.19 Die JSON-Reparatur

<sub>📍 Code: Zeilen 1690–1781</sub>

Zwei fast identische Funktionen — `_parse_json_array` für `[…]` und `_parse_json_object` für `{…}`. Beide versuchen dasselbe in drei Stufen:

**Stufe 1 — direkt versuchen**
```python
stripped = text.strip()
if stripped.startswith("["):
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
```
Der Idealfall: Das Modell hat sauberes JSON geliefert.

**Stufe 2 — Markdown-Codeblock auspacken**
```python
fence = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", text)
```
Sehr häufig verpacken Modelle ihr JSON in einen Codeblock:
````
```json
[{"brand": "Nike"}]
```
````
Obwohl der Prompt ausdrücklich „ohne Markdown" verlangt. Diese Stufe holt den Inhalt heraus.

**Stufe 3 — Klammern von Hand zählen**
```python
start = text.find("[")
while start != -1:
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "[":   depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                ...
```
Die Notlösung, wenn das Modell noch einen Einleitungssatz davorgestellt hat („Hier ist die Analyse: [...]"). Der Code sucht die erste öffnende Klammer und zählt sich zur passenden schließenden durch — **verschachtelte Klammern werden dabei korrekt mitgezählt**. Ein simples Suchmuster würde daran scheitern.

Erst wenn alle drei Stufen versagen, wird aufgegeben — mit einer Protokollzeile, die die ersten 300 Zeichen der Antwort festhält.

> ⚠️ Diese Robustheit ist notwendig, aber sie **verdeckt auch Probleme**: Wenn ein Modell dauerhaft schlechtes JSON liefert, merkt man das nur im Log, nicht in der Oberfläche.

### 2.20 Die Markenanalyse — Phase 2 im Detail

<sub>📍 Code: Zeilen 1782–2077</sub>

Das analytische Herzstück. Hier wird aus rohem Text auswertbare Struktur.

#### Der Analyse-Prompt *(Zeile 1828)*

Alle Antworten eines Batches werden durchnummeriert:

```
[0] Frage: Welche Laufschuhe für Anfänger?
Antwort: Für Einsteiger empfehlen sich …

[1] Frage: Beste Marathon-Schuhe 2026?
Antwort: …
```

Dann wird das Modell aufgefordert, ein JSON-Objekt mit **genau zwei Feldern** zurückzugeben:

**Feld `mentions`** — ein Eintrag je erkannter Marke je Antwort:

| Feld | Bedeutung |
|---|---|
| `index` | Zu welcher nummerierten Antwort gehört das? |
| `brand` | Markenname, vereinheitlicht |
| `sentiment` | `positive` / `neutral` / `negative` |
| `confidence` | `high` / `medium` / `low` |
| `reason` | Ein Satz Begründung für das Sentiment |
| `aspect` | Thema: Qualität, Preis, Empfehlung, Bekanntheit, Funktionen |
| `excerpt` | Der Belegsatz aus der Antwort, max. 200 Zeichen |
| `rank` | Position innerhalb der Antwort (1 = zuerst genannt) |

Ausdrücklich verlangt: *„Wird eine Marke innerhalb derselben Antwort mehrfach genannt, gib sie für diese Antwort nur EINMAL aus."* Ohne diese Regel würde eine Marke, die in einem Text zehnmal vorkommt, die Statistik dominieren, obwohl sie nur **eine** Antwort abdeckt.

**Feld `summary`** — eine Zusammenfassung in 3–5 Sätzen als Fließtext: Welche Marken dominieren, welche fehlen auffällig, wie ist die Tonalität.

**Zwei Modi** *(Zeilen 1836–1850)*:

| Modus | Anweisung an das Modell |
|---|---|
| **Manuell** (Marken vorgegeben) | „Erkenne **alle** Marken. Besonders wichtig sind: Nike, Adidas, ASICS. Verwende für sie exakt diese Schreibweise. Nenne aber auch alle anderen." |
| **Automatisch** | „Erkenne alle Marken. Fasse Schreibvarianten und Rebrandings zu **einem** Namen zusammen (z. B. 'Havas Health' und 'Havas Life' → 'Havas Health / Havas Life')." |

Bemerkenswert am manuellen Modus: Es wird **nicht** nur nach den vorgegebenen Marken gesucht. Das Modell erfasst alles, und erst danach filtert das Programm. So können unerwartete Wettbewerber sichtbar gemacht werden, statt unbemerkt wegzufallen.

#### Die Batch-Aufteilung *(Zeile 1808)*

```python
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
```

Die Token-Schätzung ist bewusst grob: **Zeichen ÷ 4**, plus 40 Tokens Pauschale für das Drumherum jeder Antwort. Genauer wäre nur mit einem Tokenizer möglich — für eine Obergrenze reicht die Näherung.

`j > i` stellt sicher, dass **mindestens eine** Antwort pro Batch drin ist. Sonst könnte eine einzelne riesige Antwort eine Endlosschleife auslösen.

Jeder Batch merkt sich seine **Startposition**. Das Modell nummeriert innerhalb des Batches von 0 an; beim Zusammensetzen wird die Startposition wieder aufaddiert (Zeile 2048).

#### Ein Batch wird ausgewertet *(Zeile 1916)*

```python
obj  = _parse_json_object(text)
flat = obj.get("mentions") if isinstance(obj.get("mentions"), list) else _parse_json_array(text)
summary = (obj.get("summary") or "").strip() if isinstance(obj, dict) else ""
```

Eine **Rückfall-Ebene**: Liefert das Modell statt des Objekts nur ein nacktes Array, wird dieses als Nennungsliste behandelt. Ältere Modelle machen das.

Besonders sauber gelöst ist die Unterscheidung „wirklich keine Marken" von „Antwort war unlesbar":

```python
if not flat and text.strip() not in ("[]", "{}", "") and not summary:
    parse_err = tr("Ein Analyse-Batch konnte nicht als JSON gelesen werden (evtl. abgeschnitten).", …)
```

Wenn das Modell tatsächlich `[]` schickt, ist das ein legitimes „nichts gefunden". Wenn es aber viel Text schickt, der zu nichts auswertbar ist, ist etwas kaputt — und das wird gemeldet statt stillschweigend als „keine Marken" verbucht.

#### Zusammenführen *(Zeile 2000)*

```python
try:
    idx = int(item.get("index"))
except (TypeError, ValueError):
    continue
if not (0 <= idx < len(batch)):
    log.warning("Analysis: index %s out of range … — dropped", idx)
    continue
entry = {k: item.get(k, "") for k in _MENTION_FIELDS}
by_index.setdefault(start + idx, []).append(entry)
```

Eine wichtige Sicherung: Das Modell könnte einen **erfundenen Index** liefern (etwa 47 in einem Batch mit 40 Antworten). Ohne die Bereichsprüfung landete die Nennung unter einem Schlüssel, den niemand mehr liest — sie verschwände lautlos oder würde der falschen Antwort zugeordnet.

#### Die Gesamtzusammenfassung *(Zeile 1955)*

```python
if len(batches) <= 1:
    summary = batch_summaries[0] if batch_summaries else ""
else:
    summary, s_usage, _ = _summarize_dataset(api_key, by_index, len(answers), lang)
```

Ein durchdachtes Detail: Bei **einem** Batch hat das Modell bereits den ganzen Datensatz gesehen — seine Zusammenfassung ist gültig. Bei **mehreren** Batches sah jeder nur einen Ausschnitt; eine dieser Teilzusammenfassungen als Gesamturteil auszugeben wäre irreführend.

Stattdessen wird ein zusätzlicher, günstiger Aufruf gemacht, der nur **aggregierte Statistiken** bekommt:

```
Marken-Abdeckung (Antworten mit Nennung):
Nike: 45/60
Adidas: 38/60
...
Sentiment-Verteilung: positive: 120, neutral: 60, negative: 15
```

> ⚠️ Diese Statistik ist auf die **Top 20 Marken** begrenzt (`coverage.most_common(20)`). Bei sehr vielen erkannten Marken übersieht die Zusammenfassung den langen Schwanz. Die Diagramme zeigen ihn weiterhin vollständig.

#### Fehler zusammenfassen *(Zeilen 2054–2068)*

```python
uniq = list(dict.fromkeys(errors))
error = "; ".join(uniq)
```

Scheitern zehn Batches am selben Rate Limit, sieht der Nutzer **eine** Meldung, nicht zehn identische. `dict.fromkeys` ist der übliche Python-Kniff, um Duplikate zu entfernen **und dabei die Reihenfolge zu behalten** (im Gegensatz zu einem `set`).

### 2.21 Ergebnisse in eine Tabelle bringen

<sub>📍 Code: Zeile 2080</sub>

```python
def build_analysis(results: list[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        for b in r.get("brands_found", []):
            rows.append({
                "question": r["question"], "run": r["run"], "model": r.get("model", ""),
                "brand": b.get("brand", ""), "sentiment": b.get("sentiment", "neutral"),
                ..., "mentions": 1,
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame()
```

Aus der verschachtelten Struktur („Antwort enthält Liste von Marken") wird eine **flache Tabelle**: eine Zeile je Kombination aus Frage, Durchlauf und Marke. Erst dieses Format erlaubt das Gruppieren und Summieren, auf dem alle Diagramme beruhen.

Die Spalte `"mentions": 1` sieht überflüssig aus, ist aber praktisch: Summiert man sie nach Marke, erhält man die Abdeckung. Da innerhalb einer Antwort jede Marke nur einmal auftaucht, entspricht die Summe genau der Zahl der Antworten mit Nennung.

### 2.22 Markenerkennung ohne KI

<sub>📍 Code: Zeilen 2100–2145</sub>

Ein zweiter, völlig unabhängiger Weg der Markenerkennung — **reiner Textvergleich, kein einziger API-Aufruf**. Er ist sofort nach Phase 1 verfügbar, also schon in Schritt 4.

```python
def _brand_pattern(brand: str) -> re.Pattern | None:
    parts = [re.escape(p) for p in re.split(r"[\W_]+", (brand or "").strip()) if p]
    if not parts:
        return None
    return re.compile(r"\b" + r"[\W_]{0,3}".join(parts) + r"\b", re.IGNORECASE)
```

Aus „The North Face" wird ein Suchmuster, das zwischen den Wörtern bis zu drei beliebige Trennzeichen erlaubt. So werden „The North Face", „The-North-Face" und „TheNorthFace" alle gefunden. `\b` markiert Wortgrenzen, damit „Nike" nicht in „Nikelodeon" anschlägt.

#### Marken in den Quellen *(Zeile 2126)*

```python
def find_brands_in_sources(sources, brands) -> dict[str, list[str]]:
    for src in sources or []:
        host  = _source_host(src.get("url", ""))
        title = src.get("title", "") or ""
        for b in brands:
            key = _normalize_brand_key(b)
            in_host  = bool(key) and key in re.sub(r"[^a-z0-9]", "", host)
            in_title = bool(pattern) and bool(pattern.search(title))
```

**Eine konzeptionell wichtige Ergänzung:** Wenn ein Modell auf `store.nike.com` verlinkt, ist Nike sichtbar — auch wenn der Name im Antworttext gar nicht vorkommt. Diese „Sichtbarkeit über die Quelle" wird getrennt erfasst und in der Oberfläche mit einem **🔗** gekennzeichnet.

Für den Domain-Abgleich wird ein *kompakter Schlüssel* verwendet (alles außer Buchstaben und Ziffern entfernt), damit `nike` sowohl in `store.nike.com` als auch in `nikepartner.de` gefunden wird.

#### Der Normalisierungsschlüssel *(Zeile 3225)*

```python
def _normalize_brand_key(name: str) -> str:
    key = (name or "").lower()
    key = re.sub(r"[®™©]", "", key)
    key = re.sub(r"\b(inc|corp|corporation|gmbh|ltd|limited|llc|ag|co|company|group|the)\b", " ", key)
    key = re.sub(r"[^a-z0-9]+", "", key)
    return key
```

Alles kleinschreiben, Markenzeichen-Symbole entfernen, Rechtsformen streichen, alle Sonderzeichen weg. Ergebnis: „Nike Inc.", „nike" und „NIKE®" werden alle zu `nike`.

> ⚠️ **Fallstrick — falsche Treffer möglich.** Das Streichen von `co`, `group` und `the` ist recht aggressiv. Und beim Abgleich mit den vom Modell gefundenen Marken (Zeile 3273) wird zusätzlich **Teilstring-Suche in beide Richtungen** angewandt:
> ```python
> if len(ck) >= 3 and (ck in rk or (len(rk) >= 3 and rk in ck)):
> ```
> Damit findet „Apple" auch „Applebee's", und „Visa" auch „Visage". Die Längenprüfung (mindestens 3 Zeichen) verhindert die schlimmsten Fälle, aber **bei kurzen oder generischen Markennamen sollte man die Ergebnisse stichprobenartig prüfen.**

### 2.23 Export

<sub>📍 Code: Zeilen 2147–2225</sub>

**`build_raw_export`** (Zeile 2147) baut die Rohdaten-Tabelle für Schritt 4 mit diesen Spalten:

| Spalte | Inhalt |
|---|---|
| `question`, `run`, `model` | Woher stammt die Zeile |
| `brand_found` | Wurde überhaupt eine vorgegebene Marke gefunden? |
| `brands_found` | Alle gefundenen (Text + Quellen zusammen) |
| `brands_in_answer` | Nur im Antworttext gefunden |
| `brands_in_sources` | Nur über verlinkte Quellen gefunden |
| `brands_missing` | Vorgegebene Marken, die **nicht** vorkamen |
| `source_hosts`, `source_urls`, `n_sources` | Die zitierten Quellen |
| `web_search_used`, `citation_count` | Suchbelege |
| `tokens_in`, `tokens_out` | Tokenverbrauch (0 bei Websuche!) |
| `answer` | Der Volltext |

Die Spalte `brands_missing` ist strategisch interessant: Sie beantwortet direkt „Wo taucht unsere Marke **nicht** auf?" — oft die wichtigere Frage.

**`save_csv`** (Zeile 2185) schreibt in den Ordner `results/`, bewusst im selben Spaltenformat wie das Vorgängerscript `brand_monitor.py`, damit bestehende Auswertungen mit `analyze_csv.py` weiter funktionieren. Der Antworttext wird hier auf **500 Zeichen** gekürzt und Zeilenumbrüche werden zu Leerzeichen — damit die CSV in Excel lesbar bleibt.

**`download_with_name`** (Zeile 2938) ist ein Kniff:

```python
stem = st.text_input(tr("Dateiname", "File name"), value=Path(default_name).stem, ...)
safe = re.sub(r"[^\w.\- ]+", "_", (stem or Path(default_name).stem).strip()) or Path(default_name).stem
st.download_button(label, data, f"{safe}{ext}", mime, ...)
```

Streamlit legt den Dateinamen eines Download-Buttons beim Zeichnen fest. Deshalb **muss** das Eingabefeld vor dem Button stehen. Die Zeile mit `re.sub` entfernt Pfadtrennzeichen — sonst könnte ein Name wie `../../geheim` die Datei außerhalb des Download-Ordners ablegen. Eine kleine, aber richtige Sicherheitsmaßnahme.

### 2.24 Die Tutorials

<sub>📍 Code: Zeilen 2227–2300</sub>

```python
_TUTORIALS = {
    1: ("❓ Anleitung — Schritt 1: Einrichtung", "❓ Tutorial — Step 1: Setup", """…de…""", """…en…"""),
    ...
}
```

Ein Nachschlagewerk mit einem Eintrag je Schritt, jeweils Titel und Text in beiden Sprachen. `render_tutorial(step)` (Zeile 2294) zeigt sie in einem **eingeklappten** Aufklappbereich — Hilfe für neue Nutzer, ohne Erfahrene zu stören.

### 2.25 Oberfläche Schritt 1

<sub>📍 Code: Zeile 2307</sub>

Zwei Spalten im Verhältnis 3:2 — links die Eingaben, rechts die Hinweise.

**Die Modellauswahl hängt an einer einzigen Bedingung** *(Zeile 2343)*:
```python
if web_search:
    model = render_agent_model_picker(api_key, key_prefix="s1")
else:
    catalog, probe_failed = list_completion_models(api_key)
    ...
```
Websuche an → Agent-Katalog. Websuche aus → drei Sondierungen plus ein Eintrag „Benutzerdefiniert…" für manuelle Eingabe.

**Die Bereitschaftsprüfung** *(Zeilen 2528–2550)* — Der Weiter-Button bleibt gesperrt, solange etwas Notwendiges fehlt, und begründet das jedes Mal:
```python
ready = bool(api_key and model)
if question_mode == opt_gen and not topic.strip():
    ready = False
    st.warning(tr("Bitte ein Thema eingeben, …", …))
```

**Die Kostenvorschau** *(Zeilen 2470–2478)* rechnet live mit — bei automatischen Fragen mit dem Schiebereglerwert, bei eigenen Fragen mit den tatsächlich eingetippten nicht-leeren Zeilen.

Beim Klick auf Weiter wird die gesamte Konfiguration in den Session State geschrieben und — bei automatischem Modus — direkt die Fragengenerierung angestoßen.

### 2.26 Oberfläche Schritt 2

<sub>📍 Code: Zeile 2594</sub>

Ein großes Textfeld, 450 Pixel hoch. Darunter eine Live-Zählung mit Differenzanzeige:

```python
delta = n_lines - orig
delta_str = tr(f" ({'+' if delta >= 0 else ''}{delta} gegenüber generiert)", …) if delta != 0 else ""
```

Man sieht also sofort „23 Fragen (+3 gegenüber generiert)".

### 2.27 Oberfläche Schritt 3

<sub>📍 Code: Zeile 2647</sub>

Die dichteste Seite. Neben den bereits beschriebenen Reglern zwei Besonderheiten:

#### Die Rettung abgebrochener Läufe *(Zeilen 2660–2682)*

```python
partial = st.session_state.get("raw_answers", [])
if partial and not st.session_state.get("phase1_complete", False):
    st.info(tr(f"📥 {len(partial)} Antworten aus einem abgebrochenen Lauf sind gespeichert.", …))
```

Weil Streamlit einen laufenden Sammelvorgang bei jedem Klick abreißt, wären die bereits **bezahlten** Antworten sonst unerreichbar. Die App bietet nun an: mit den vorhandenen Antworten weiter zur Analyse — oder verwerfen und neu sammeln.

#### Extended Thinking wird nur angeboten, wenn es geht *(Zeilen 2750–2752)*

```python
et_capable = {m["id"] for m in agent_catalog if m.get("supportsExtendedThinking")}
et_missing = [m for m in models if m not in et_capable] if agent_catalog else models
et_possible = bool(web_search and models and not et_missing)
```

Der Schalter ist nur aktiv, wenn Websuche läuft **und** **jedes** ausgewählte Modell die Fähigkeit meldet. Sonst wäre bei einem Mehrfach-Modell-Lauf ein Teil der Aufrufe vorprogrammiert fehlerhaft. Darunter wird namentlich aufgelistet, welches Modell es nicht kann.

#### Die Token-Last-Anzeige *(Zeilen 2842–2856)*

```python
assumed_resp_s = 30
calls_per_min  = parallel_calls * (60 / assumed_resp_s)
tpm_estimate   = int(calls_per_min * max_tokens_val)
tpm_pct        = tpm_estimate / 60000 * 100
tpm_color      = "🟢" if tpm_pct < 70 else ("🟡" if tpm_pct < 100 else "🔴")
```

Eine Ampel, die vor Rate Limits warnt, bevor sie eintreten. Sie nimmt 30 Sekunden Antwortzeit an und rechnet mit dem **vollen** Token-Budget je Aufruf.

> ⚠️ Diese Schätzung ist bewusst pessimistisch (die meisten Antworten schöpfen `max_tokens` nicht aus) und der 60.000er-Bezugswert gilt nur für bestimmte Modelle. Als grober Kompass taugt sie, als exakte Vorhersage nicht. Bei aktiver Websuche wird sie gar nicht erst angezeigt, weil `max_tokens` dort keine Wirkung hat.

### 2.28 Phase 1 — der Sammellauf

<sub>📍 Code: Zeile 2984</sub>

Die Funktion, in der die eigentliche Arbeit passiert.

#### Vorbereitung *(Zeilen 2985–3005)*

```python
lang = st.session_state.get("lang", "de")   # HIER auslesen — Threads können das nicht
total = len(questions) * runs * len(models)
reset_run_abort()
reset_dead_models()

raw_answers: list[dict] = []
st.session_state.raw_answers = raw_answers
st.session_state.phase1_complete = False
```

**Die drei markierten Zeilen sind der Kern der Absturzsicherheit:** `raw_answers` wird **sofort** in den Session State gelegt — nicht erst am Ende. Weil in Python Listen als Verweis übergeben werden, landet jede später hinzugefügte Antwort automatisch auch dort. Wird der Lauf durch einen Klick abgerissen, sind alle bis dahin gesammelten Antworten trotzdem gespeichert.

Der Kommentar sagt es deutlich: *Das war genau das, was einen gestoppten Lauf früher unwiederbringlich machte.*

#### Die Aufgabenliste *(Zeilen 3046–3051)*

```python
all_tasks = [
    (i, question, run_num, mdl)
    for i, question in enumerate(questions)
    for run_num in range(runs)
    for mdl in models
]
```

Ein **dreifach verschachtelter Listenaufbau**: jede Frage × jeden Durchlauf × jedes Modell. Bei 20 Fragen, 3 Durchläufen und 2 Modellen ergibt das 120 Aufgaben.

#### Zeitmessung an der richtigen Stelle *(Zeile 3053)*

```python
def _timed_ask(question: str, mdl: str):
    t_start = time.time()
    answer, err, usage = ask_question(...)
    return answer, err, usage, time.time() - t_start
```

Der Kommentar erklärt einen behobenen Messfehler: Wird ab dem **Einreichen** gemessen statt ab dem **Bearbeitungsbeginn**, sieht jede wartende Aufgabe so aus, als hätte sie die gesamte Phase gedauert. Ein Lauf mit 60 Aufrufen und 2 Arbeitern meldete „Ø 547s, max 1167s" — für Aufrufe, die nach spätestens 240 Sekunden abbrechen. Physikalisch unmöglich, aber genau so stand es im Dashboard.

#### Die parallele Ausführung *(Zeilen 3067–3130)*

```python
with ThreadPoolExecutor(max_workers=parallel) as executor:
    future_map = {
        executor.submit(_timed_ask, q, mdl): (i, run_num, q, mdl)
        for i, q, run_num, mdl in all_tasks
    }
    for future in as_completed(future_map):
        ...
```

**Was passiert hier?** Alle Aufgaben werden auf einmal eingereicht, aber es arbeiten nur `parallel` viele gleichzeitig. `as_completed` liefert die Ergebnisse **in der Reihenfolge ihrer Fertigstellung**, nicht in der Einreichungsreihenfolge — deshalb braucht es `future_map`, um zu wissen, welches Ergebnis zu welcher Frage gehört.

In der Schleife wird bei jedem fertigen Ergebnis:
1. auf „Stoppen" geprüft,
2. auf die Notbremse geprüft (dann werden alle noch wartenden Aufgaben abgebrochen),
3. das Ergebnis abgeholt und die Zeit erfasst,
4. Fortschrittsbalken und Live-Dashboard aktualisiert,
5. die Antwort gespeichert **oder** die Aufgabe für den Wiederholungsdurchgang vorgemerkt.

#### Der Wiederholungsdurchgang *(Zeilen 3145–3190)*

```python
if failed_tasks and not st.session_state.get("stop_requested", False) and not run_abort_reason():
```

Alle fehlgeschlagenen Aufrufe bekommen am Ende der Phase **eine** weitere Chance. Die Begründung im Kommentar ist methodisch:

> Ein Fehlschlag hier würde diese Frage sonst mit weniger Durchläufen zurücklassen als die anderen — und damit den Share of Voice verzerren.

Genau richtig gedacht: Wenn Frage 7 nur zweimal statt dreimal beantwortet wurde, sind alle darin genannten Marken systematisch unterrepräsentiert.

#### Abschluss *(Zeilen 3196–3222)*

```python
needs_attention = bool(invalid_model_id()) or not raw_answers
```

Zwei Fälle, in denen **nicht** weitergegangen wird: Ein Modell wurde abgelehnt (muss neu gewählt werden) oder es kam gar nichts zurück (die Analyse würde nur einen zweiten Fehler erzeugen). Sonst geht es zu Schritt 4.

### 2.29 Phase 2 — die Analyse

<sub>📍 Code: Zeile 3236</sub>

```python
reset_run_abort()    # ein Limit während des Sammelns darf die Analyse nicht blockieren
reset_dead_models()  # ebenso ein stillgelegtes Sammelmodell — die Analyse nutzt ihr eigenes
by_index, summary, usage, analysis_err = analyze_dataset(api_key, raw_answers, brands, lang)
```

Die beiden Rücksetzungen sind wichtig: Die Analyse läuft mit einem **anderen** Modell. Ein Problem aus Phase 1 darf sie nicht mitreißen.

Danach werden die Ergebnisse in der **ursprünglichen Reihenfolge** wieder zusammengesetzt und mit den Marken angereichert. Im manuellen Modus filtert `_normalize_brands` auf die vorgegebene Liste — alle anderen gefundenen Marken landen in `unlisted_brands` und werden in Schritt 5 unter „Ebenfalls genannt" angezeigt, statt zu verschwinden.

### 2.30 Oberfläche Schritt 4 — Rohdaten

<sub>📍 Code: Zeile 3383</sub>

Der wichtigste Teil dieser Seite ist die **Websuche-Verifikation** *(Zeilen 3408–3441)*:

```python
n_searched  = sum(1 for r in raw_answers if r.get("web_search_used"))
n_sources   = sum(len(r.get("sources", [])) for r in raw_answers)
n_citations = sum(r.get("citation_count", 0) for r in raw_answers)
```

Daraus wird eine von drei Meldungen:

| Lage | Meldung |
|---|---|
| Alle Antworten haben gesucht | ✅ Grün: „Websuche bestätigt" |
| Manche | ⚠️ Gelb: „teilweise genutzt — die übrigen wurden aus dem Modellwissen beantwortet" |
| Keine | 🔴 Rot: „Keine Antwort hat das Such-Tool nachweislich genutzt" |

**Das ist eine der wertvollsten Funktionen des Tools.** Ohne sie könnte man einen kompletten Lauf auswerten und erst viel später merken, dass die Modelle gar nicht gesucht, sondern aus veraltetem Trainingswissen geantwortet haben — was die Ergebnisse für aktuelle Marktfragen wertlos macht.

Die Wortwahl ist bewusst vorsichtig: „**nachweislich** genutzt". Fehlender Beleg heißt nicht mit Sicherheit „nicht gesucht" — es kann auch sein, dass die API die Belege nicht mitgeliefert hat. Der Code weist an mehreren Stellen darauf hin.

In der Tabelle darunter markiert **🔗** Marken, die nur über die verlinkten Quellen gefunden wurden. Im Aufklappbereich „Alle Antworten" stehen die Volltexte samt anklickbarer Quellenliste.

### 2.31 Oberfläche Schritt 5 — Ergebnisse

<sub>📍 Code: Zeile 3601</sub>

#### Der Konfidenz-Filter *(Zeilen 3694–3711)*

```python
allowed = set(selected_conf) | {"", None}
df = df[df["confidence"].isin(allowed)]
```

Filtert **alle** nachfolgenden Ansichten gleichzeitig, weil sie alle auf demselben `df` beruhen. Nennungen **ohne** Konfidenzangabe bleiben immer sichtbar (das `| {"", None}`) — sonst würden ältere Läufe oder Antworten, in denen das Modell das Feld weggelassen hat, komplett verschwinden.

#### Die Kennzahlen *(Zeilen 3711–3745)*

Vier Zahlen: Antworten, eindeutige Fragen, erkannte Marken, Top-Marke mit Abdeckung.

```python
coverage = df.groupby("brand")["mentions"].sum().sort_values(ascending=False)
```

Weil Marken pro Antwort entdupliziert sind, ist diese Summe die **Abdeckung** — in wie vielen Antworten die Marke vorkam.

#### Tab „Share of Voice"

Drei Darstellungen:

1. **Balkendiagramm der Anteile** — Anteil jeder Marke an allen Nennungen.
2. **Prominenz-Tabelle** — durchschnittliche Position je Marke (`rank`). Niedriger = früher genannt. Ergänzt die reine Häufigkeit um die Frage „wie weit oben?".
3. **Sentiment-Heatmap Marke × Frage** — eine Farbmatrix:
   ```python
   sentiment_score = {"positive": 1, "neutral": 0, "negative": -1}
   ```
   Grün (+1) über Grau (0) bis Rot (−1); Weiß = nicht erwähnt. Fragen erscheinen als Q1, Q2, … , der Volltext steht im „Fragenverzeichnis" darunter und im Tooltip.

#### Tab „Sentiment"

Prozentverteilung je Marke, ein gestapeltes Balkendiagramm und — besonders nützlich — **alle Belegzitate**, gruppiert nach Tonalität. Pro Marke und Sentiment werden bis zu 15 Einträge gezeigt, jeweils mit Begründung, wörtlichem Auszug, Aspekt-Etikett und Konfidenz.

> ⚠️ **Fallstrick:** Die Prozentverteilung wird über `st.columns(len(sorted(df["brand"].unique())))` gezeichnet — **eine Bildschirmspalte je Marke**. Bei 5 Marken sieht das gut aus. Bei 40 automatisch erkannten Marken entstehen 40 hauchdünne Spalten, die praktisch unlesbar sind. Im automatischen Erkennungsmodus mit breitem Thema ist das der wahrscheinlichste Anzeigeärger.

#### Tab „Alle Antworten"

Volltexte mit Filter nach Frage. Zu jeder Antwort werden die erkannten Marken mit farbigem Sentiment-Punkt angezeigt.

#### Tab „Rohdaten"

Tabelle mit Tokenverbrauch. Die Spaltenüberschriften haben Erklärungstexte (`column_config`), die beim Darüberfahren erscheinen. Darunter ein Debug-Bereich mit den ersten drei Antworten im Volltext — gedacht für die Fehlersuche bei der Markenerkennung.

#### Tab „Laufzeit"

Histogramm der Antwortzeiten in Phase 1, Balkendiagramm für Phase 2, Detailtabelle, Erfolgsquote. Nützlich, um die Parallelität sinnvoll einzustellen.

#### Neu analysieren *(Zeilen 3788–3794)*

```python
if st.button(tr("Analyse neu starten", "Restart analysis"), type="primary", key="reanalyze"):
    _run_brand_analysis(raw_answers)
```

Führt **nur Phase 2** erneut aus, auf den bereits gespeicherten Antworten. Die teuren Sammel-Aufrufe entfallen. Praktisch, wenn eine Analyse an einem Rate Limit gescheitert ist oder man die Markenliste anpassen möchte.

### 2.32 Der Router

<sub>📍 Code: Zeilen 4269–4280</sub>

```python
step = st.session_state.step

if step == 1:   render_step1()
elif step == 2: render_step2()
elif step == 3: render_step3()
elif step == 4: render_step4()
elif step == 5: render_step5()
```

Die letzten Zeilen der Datei und zugleich der Kern des Streamlit-Modells: Bei jedem Neudurchlauf wird geschaut, welcher Schritt gespeichert ist, und genau eine Seite gezeichnet. Ein Klick, der `st.session_state.step = 4` setzt und `st.rerun()` aufruft, „navigiert" damit auf die nächste Seite.

---

## 3. Der Datenfluss auf einen Blick

```
   Nutzereingabe (Schritt 1)
        │  Thema, Marken, Modell, API-Key
        ▼
   generate_questions()            ─── 1 API-Call
        │  ["Frage 1?", "Frage 2?", …]
        ▼
   Schritt 2: Nutzer bearbeitet die Fragen
        │
        ▼
   Schritt 3: Runs, Modelle, Parallelität
        │
        ▼
┌─────────────────────────────────────────────────┐
│ PHASE 1  _run_phase1()                          │
│                                                 │
│  Fragen × Runs × Modelle  →  Aufgabenliste      │
│         │                                       │
│    ThreadPoolExecutor (N parallel)              │
│         │                                       │
│    ask_question() → call_langdock()             │
│         ├── web_search=False → Passthrough      │
│         └── web_search=True  → Agent-API (SSE)  │
│         │                                       │
│    Wiederholungsdurchgang für Fehlschläge       │
└─────────────────────────────────────────────────┘
        │  raw_answers = [{question, run, answer,
        │                  model, tokens, sources, …}]
        ▼
   Schritt 4: Kontrollpunkt  ── Export ohne Analyse möglich
        │
        ▼
┌─────────────────────────────────────────────────┐
│ PHASE 2  _run_brand_analysis()                  │
│                                                 │
│  _make_analysis_batches()  → Pakete à ≤40       │
│         │                                       │
│    je Batch: _analyze_batch()   ─── 1 API-Call  │
│              mit claude-opus-4-8                │
│         │                                       │
│    JSON auswerten → Index prüfen → einsortieren │
│         │                                       │
│    bei >1 Batch: _summarize_dataset() ─ 1 Call  │
└─────────────────────────────────────────────────┘
        │  results = [{…, brands_found: [{brand,
        │              sentiment, rank, excerpt, …}]}]
        ▼
   build_analysis() → DataFrame (eine Zeile je Nennung)
        │
        ▼
   Schritt 5: Diagramme, Tabellen, Export
```

### Wie viele API-Aufrufe insgesamt?

| Phase | Formel | Beispiel (20 Fragen, 3 Runs, 2 Modelle) |
|---|---|---|
| Fragen generieren | 1 (entfällt bei eigenen Fragen) | 1 |
| Sammeln | Fragen × Runs × Modelle | 20 × 3 × 2 = **120** |
| Wiederholungen | Nur für Fehlschläge | typisch 0–5 |
| Analyse | ⌈Antworten ÷ 40⌉ | ⌈120/40⌉ = **3** |
| Gesamtzusammenfassung | 1, falls mehr als ein Batch | 1 |
| **Summe** | | **≈ 125** |

Dazu kommen wenige tokenlose Sondierungsaufrufe für die Modelllisten.

---

## 4. Anhang: Die Nachbardatei `langdock_evidence.py`

Eine kleine, in sich abgeschlossene Datei mit vier Bausteinen:

| Baustein | Aufgabe |
|---|---|
| `key_fingerprint(api_key)` | Erzeugt `sha256:<12 Hex>/…<letzte 4 Zeichen>`. Beweist „selber Schlüssel", ohne ihn preiszugeben |
| `redact(text, api_key)` | Sicherheitsnetz: ersetzt den Key, falls er je zurückgespiegelt wird. Alles andere bleibt Byte für Byte erhalten |
| `EvidenceRecorder` | Schreibt je Anfrage eine JSON-Zeile. Mit `run_id` + `seq`, damit eine Anfrage der Katalogabfrage zugeordnet werden kann. Durch eine Sperre (`threading.Lock`) abgesichert, weil aus mehreren Arbeitssträngen geschrieben wird |
| `response_headers(resp)` | Alle Antwort-Header, ungefiltert — die instanzidentifizierenden (`x-request-id`, `cf-ray`, `x-vercel-id`) belegen, dass zwei widersprüchliche Katalogantworten von verschiedenen Servern kamen |

Ein Detail, das Sorgfalt zeigt:

```python
except OSError:
    # Evidence recording must never take a run down with it.
    pass
```

Kann die Beweisdatei nicht geschrieben werden (Festplatte voll, Rechte fehlen), wird das stillschweigend übergangen. Das Protokollieren darf niemals einen laufenden Lauf zum Absturz bringen.

---

*Erstellt am 26.08.2026 · Bezieht sich auf den Stand von `app.py` mit 4.280 Zeilen.*

---

*Teil der Dokumentation des LLM Brand Visibility Trackers · [README](../README.md) · [Glossar](GLOSSAR.md)*
