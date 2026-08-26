# Dokumentation: `app.py` — LLM Brand Visibility Tracker

> **Für wen ist dieses Dokument?**
> Für alle Kolleginnen und Kollegen, die verstehen wollen, **was dieses Tool tut, wie es funktioniert und wo es Fallstricke gibt** — ganz ohne Python-Kenntnisse.
> Jeder Fachbegriff wird beim ersten Auftauchen erklärt. Wenn Sie etwas nachschlagen wollen, nutzen Sie das [Glossar](#2-glossar--alle-fachbegriffe-erklärt).
>
> Die Datei `app.py` ist rund **4.280 Zeilen** lang. Verweise wie *(Zeile 863)* zeigen Ihnen, wo im Code der beschriebene Abschnitt steht.

---

## Inhaltsverzeichnis

1. [Was macht dieses Tool? — Die Kurzfassung](#1-was-macht-dieses-tool--die-kurzfassung)
2. [Glossar — alle Fachbegriffe erklärt](#2-glossar--alle-fachbegriffe-erklärt)
3. [Installation und Start](#3-installation-und-start)
4. [Der Ablauf aus Nutzersicht — die 5 Schritte](#4-der-ablauf-aus-nutzersicht--die-5-schritte)
5. [Die Landkarte: Wie das Script aufgebaut ist](#5-die-landkarte-wie-das-script-aufgebaut-ist)
6. [Der Code im Detail — Abschnitt für Abschnitt](#6-der-code-im-detail--abschnitt-für-abschnitt)
7. [Der Datenfluss auf einen Blick](#7-der-datenfluss-auf-einen-blick)
8. [Potenzielle Probleme und Fallstricke](#8-potenzielle-probleme-und-fallstricke)
9. [Kosten und Laufzeit realistisch einschätzen](#9-kosten-und-laufzeit-realistisch-einschätzen)
10. [Sicherheit und Datenschutz](#10-sicherheit-und-datenschutz)
11. [Stellschrauben — was man wo ändern kann](#11-stellschrauben--was-man-wo-ändern-kann)
12. [Troubleshooting — häufige Fehlermeldungen](#12-troubleshooting--häufige-fehlermeldungen)

---

## 1. Was macht dieses Tool? — Die Kurzfassung

### Die Geschäftsfrage dahinter

Immer mehr Menschen suchen nicht mehr bei Google, sondern fragen einen **KI-Chatbot** („Welche CRM-Software eignet sich für Mittelständler?"). Die Antwort dieses Chatbots entscheidet dann mit, welche Marken überhaupt in die engere Auswahl kommen.

**Die Frage lautet also: Wie sichtbar ist unsere Marke in den Antworten von KI-Modellen — und wird sie positiv oder negativ dargestellt?**

Genau das misst dieses Tool.

### Was das Tool konkret tut

In vier Arbeitsschritten:

| Schritt | Was passiert | Beispiel |
|---|---|---|
| **1. Fragen erzeugen** | Das Tool lässt ein KI-Modell typische Nutzerfragen zu Ihrem Thema formulieren — oder Sie geben eigene Fragen ein. | „Welche Sportschuhe eignen sich für Marathonläufer?" |
| **2. Antworten sammeln** | Jede Frage wird mehrfach an ein oder mehrere KI-Modelle gestellt. Optional mit **echter Websuche**. | 20 Fragen × 3 Wiederholungen × 2 Modelle = 120 Antworten |
| **3. Marken erkennen** | Ein starkes KI-Modell (Claude Opus) liest alle gesammelten Antworten und notiert: Welche Marke kommt vor? Wie wird sie bewertet? An welcher Stelle wird sie genannt? | „Nike — positiv — an 1. Stelle genannt — Aspekt: Qualität" |
| **4. Auswerten** | Diagramme, Tabellen, Exportdateien: Wer dominiert? Wer fehlt? Wie ist die Tonalität? | Balkendiagramm „Share of Voice", Sentiment-Heatmap |

### Warum mehrfach fragen?

KI-Modelle antworten **nicht immer gleich**. Stellt man dieselbe Frage zehnmal, bekommt man zehn leicht unterschiedliche Antworten. Erst durch die Wiederholung entsteht eine belastbare Statistik: „Nike wurde in 8 von 10 Antworten genannt" ist eine Aussage — „Nike wurde einmal genannt" ist Zufall.

Damit die Wiederholungen sich wirklich unterscheiden, wird bewusst eine **Temperatur** von 0.7 gesetzt (siehe Glossar) — sonst würde das Modell immer exakt dieselbe Antwort liefern und die Wiederholung wäre wertlos.

### Was ist Langdock?

**Langdock** ist eine Zwischenschicht („Gateway"). Statt sich einzeln bei OpenAI, Anthropic und Google anzumelden, spricht das Tool nur mit Langdock, und Langdock leitet die Anfragen an die jeweiligen KI-Anbieter weiter. Vorteil: ein Zugang, eine Abrechnung, EU-Hosting möglich. Deshalb dreht sich im Code sehr viel um Langdock-Besonderheiten.

---

## 2. Glossar — alle Fachbegriffe erklärt

Diese Begriffe tauchen in der Doku und im Code laufend auf. Sie können den Abschnitt überspringen und bei Bedarf zurückspringen.

### Grundbegriffe

**LLM (Large Language Model)**
Ein „großes Sprachmodell" — die Technik hinter ChatGPT, Claude, Gemini. Ein Programm, das auf riesigen Textmengen trainiert wurde und darauf spezialisiert ist, Text fortzusetzen. Wenn Sie ihm eine Frage stellen, „errät" es die passendste Antwort. Beispiele im Tool: `gpt-5-mini`, `claude-opus-4-8`, `gemini-…`.

**Modell / Modell-ID**
Die konkrete Version eines LLM, angesprochen über einen technischen Namen wie `claude-opus-4-8` oder `gpt-5-mini-eu`. Dieser Name muss **exakt** stimmen — ein Tippfehler führt zu einer Fehlermeldung. Das ist im Code ein großes Thema (siehe **Abschnitt 6.12**).

**Prompt**
Der Text, den man dem Modell schickt — Frage plus Anweisungen. Beispiel aus dem Code:
> „Du bist ein hilfreicher Assistent. Beantworte die folgende Frage sachlich und ausführlich. Frage: …"

Der Prompt ist die wichtigste Stellschraube für die Qualität der Ergebnisse.

**Token**
Die Abrechnungseinheit von KI-Modellen. Ein Token ist ungefähr ein halbes bis ein ganzes Wort (im Deutschen eher weniger, weil Wörter länger sind). Faustregel im Code: **4 Zeichen ≈ 1 Token**.
- *Input-Tokens (prompt_tokens)*: Was Sie hinschicken.
- *Output-Tokens (completion_tokens)*: Was das Modell antwortet.
Beides kostet Geld. `max_tokens` begrenzt, wie lang die Antwort maximal werden darf.

**Temperatur (temperature)**
Ein Regler zwischen 0 und 1, der steuert, wie „kreativ" bzw. zufällig das Modell antwortet.
- `0.0` = maximal berechenbar, dieselbe Frage → praktisch dieselbe Antwort.
- `0.7` = deutliche Variation.

Im Tool bewusst unterschiedlich gesetzt (Zeilen 169–171):
| Wo | Wert | Warum |
|---|---|---|
| Antworten sammeln | **0.7** | Wiederholungen sollen sich unterscheiden, sonst ist die Statistik wertlos |
| Fragen generieren | **0.8** | Möglichst vielfältige Fragen |
| Analyse | **0.0** | Die Markenerkennung soll reproduzierbar sein, nicht kreativ |

**Reasoning-Modell**
Ein neuerer Modelltyp (z. B. `gpt-5-mini`, `o3`), der vor der eigentlichen Antwort intern „nachdenkt". Dieses Nachdenken **kostet ebenfalls Tokens**, taucht aber nicht in der Antwort auf. Praktische Folge: Setzt man `max_tokens` zu niedrig, verbraucht das Modell sein ganzes Budget fürs Denken und liefert **eine leere Antwort**. Deshalb ist der Standardwert im Tool mit 8.000 relativ hoch (Zeile 153).

### Technische Begriffe

**API (Application Programming Interface)**
Eine „Steckdose" für Programme. Statt eine Website mit der Maus zu bedienen, schickt ein Programm eine strukturierte Anfrage und bekommt eine strukturierte Antwort. Dieses Tool bedient KI-Modelle ausschließlich über APIs.

**API-Key**
Ein Passwort für die API — eine lange Zeichenkette wie `sk--abc123…`. Wer ihn hat, kann auf Ihre Rechnung Anfragen stellen. **Deshalb: niemals in ein öffentliches Repository hochladen.**

**Endpoint (Endpunkt)**
Die konkrete Internetadresse einer API-Funktion. Dieses Tool spricht mit vier verschiedenen Endpoints (Zeilen 137–147):
| Endpoint | Wofür |
|---|---|
| `…/openai/eu/v1/chat/completions` | OpenAI-Modelle direkt |
| `…/anthropic/eu/v1/messages` | Claude-Modelle direkt |
| `…/google/eu/v1beta/models/…` | Gemini-Modelle direkt |
| `…/agent/v1/chat/completions` | **Agent-API** — der einzige Weg mit echter Websuche |

**HTTP-Statuscode**
Eine dreistellige Zahl, mit der ein Server das Ergebnis meldet. Die wichtigsten hier:
| Code | Bedeutung | Im Tool |
|---|---|---|
| **200** | Alles gut | Antwort wird verarbeitet |
| **400** | Anfrage fehlerhaft | Meist ein ungültiger Modellname |
| **401** | Nicht angemeldet | API-Key falsch oder abgelaufen |
| **403** | Verboten | Key hat keine Berechtigung |
| **404** | Nicht gefunden | Meist falsche Region eingestellt |
| **422** | Inhaltlich unverarbeitbar | Wie 400 |
| **429** | Zu viele Anfragen | **Rate Limit** — Tool wartet und versucht es erneut |
| **5xx / 524** | Serverproblem / Zeitüberschreitung | Auf Langdock-Seite |

**Rate Limit**
Eine Obergrenze, wie viel man pro Zeiteinheit anfragen darf. Bei Langdock z. B. **60.000 Tokens pro Minute** für manche Modelle. Wird sie überschritten, kommt ein 429-Fehler. Das Tool wartet dann und probiert es erneut.

**Retry / Backoff**
*Retry* = erneuter Versuch nach einem Fehler. *Backoff* = die Wartezeit verdoppelt sich bei jedem Versuch, damit man den Server nicht weiter überlastet. Im Tool: **15s → 30s → 60s → 120s** (plus ein kleiner Zufallsanteil, siehe „Jitter").

**Jitter**
Ein kleiner Zufallsaufschlag auf die Wartezeit (im Code `random.uniform(0, 5)`, also 0–5 Sekunden). Ohne ihn würden alle parallel laufenden Anfragen exakt gleichzeitig wieder loslegen und sofort erneut ein Rate Limit auslösen („thundering herd" — donnernde Herde).

**JSON**
Ein Textformat zum Austausch strukturierter Daten. Sieht so aus:
```json
{"brand": "Nike", "sentiment": "positive", "rank": 1}
```
Das Tool bittet die KI ausdrücklich um JSON, damit die Antwort maschinell auswertbar ist. **Problem:** KI-Modelle halten sich nicht immer daran — deshalb gibt es im Code aufwendige „Reparatur-Parser" (Abschnitt 6.20).

**Parsen**
Das Zerlegen und Interpretieren von Text durch ein Programm. „JSON parsen" = aus dem Text `{"brand": "Nike"}` eine nutzbare Datenstruktur machen. Schlägt das fehl, spricht man von einem *Parse-Fehler*.

**Regex (Regulärer Ausdruck)**
Ein Suchmuster für Text, das weit über „Strg+F" hinausgeht. `\bnike\b` findet z. B. das Wort „nike" als eigenständiges Wort, aber nicht innerhalb von „nikelodeon". Regex-Muster sind kryptisch, aber extrem mächtig; sie stecken im Code an vielen Stellen (Markenerkennung, Fehleranalyse, Bereinigung).

**Streaming / SSE (Server-Sent Events)**
Normalerweise wartet man, bis die komplette Antwort fertig ist. Beim *Streaming* schickt der Server die Antwort stückweise, sobald sie entsteht (wie das „Tippen" in ChatGPT). Das Tool **muss** für die Websuche streamen, weil Langdock nicht-gestreamte Agent-Anfragen nach 100 Sekunden hart abbricht (Fehler 524) — und eine Websuche dauert oft länger.

**Thread / Parallelität / ThreadPoolExecutor**
Ein *Thread* ist ein paralleler Arbeitsstrang. Statt 120 Anfragen nacheinander zu stellen (sehr langsam), startet das Tool mehrere gleichzeitig. Der `ThreadPoolExecutor` ist die Python-Werkbank dafür: Man legt fest, wie viele „Arbeiter" gleichzeitig arbeiten dürfen (im Tool einstellbar 1–10, Standard 2).
**Trade-off:** mehr Arbeiter = schneller, aber höheres Risiko für Rate Limits (429).

**Cache**
Ein Zwischenspeicher. Statt die Modellliste bei jedem Klick neu abzurufen, merkt sich das Tool sie für 60 bzw. 300 Sekunden. Spart Zeit und Anfragen.

**DataFrame (pandas)**
Eine Tabelle im Arbeitsspeicher — im Prinzip ein Excel-Blatt für Programme, mit Spalten und Zeilen. Die Bibliothek `pandas` liefert dazu Funktionen wie Gruppieren, Summieren, Sortieren. Alle Diagramme im Tool basieren auf DataFrames.

**Plotly**
Die Bibliothek, die die interaktiven Diagramme zeichnet (Balken, Heatmap, Histogramm).

### Streamlit-spezifische Begriffe

**Streamlit**
Ein Framework, das aus einem Python-Script eine Web-Oberfläche macht. Man schreibt `st.button("Klick mich")` und bekommt einen Button im Browser. Das ist der Grund, warum diese „App" nur aus einer einzigen Datei besteht.

**Rerun (Neuausführung)** — *sehr wichtig zu verstehen*
Streamlit funktioniert anders als normale Programme: **Bei jeder Nutzerinteraktion — jedem Klick, jeder Eingabe — wird das komplette Script von Zeile 1 bis Zeile 4280 neu ausgeführt.**
Das erklärt viele merkwürdig anmutende Konstruktionen im Code, unter anderem:
- den Logging-Schutz in Zeile 38 (sonst würde jede Logzeile mehrfach geschrieben),
- den `session_state` (siehe nächster Punkt),
- warum ein Klick auf „Stoppen" einen laufenden Sammelvorgang tatsächlich beendet.

**Session State (`st.session_state`)**
Der einzige Ort, an dem Daten einen Rerun überleben. Ein Gedächtnis, das an die Browser-Sitzung gekoppelt ist. Hier liegen: aktueller Schritt, Konfiguration, Fragen, gesammelte Antworten, Ergebnisse (Zeilen 76–95).
**Wichtig:** Der Session State lebt nur im Arbeitsspeicher. Browser-Tab zu = alles weg. Deshalb die Export-Buttons.

**Widget**
Ein Bedienelement: Button, Schieberegler (`slider`), Textfeld (`text_input`), Auswahlliste (`selectbox`), Mehrfachauswahl (`multiselect`), Schalter (`toggle`/`checkbox`).

### Fachbegriffe der Auswertung

**Sentiment**
Die Tonalität einer Aussage: **positiv**, **neutral** oder **negativ**. „Nike ist der Marktführer bei Laufschuhen" = positiv. „Nike ist teuer" = negativ.

**Confidence (Konfidenz)**
Wie sicher sich das Analyse-Modell bei seiner Sentiment-Einschätzung ist: `high`, `medium`, `low`. Im Ergebnis-Screen kann man danach filtern.

**Share of Voice (SoV)**
Der Anteil einer Marke an allen Markennennungen. Wenn in 100 Nennungen 30-mal Nike vorkommt, hat Nike 30 % Share of Voice. Der klassische Sichtbarkeitswert aus der Media-Analyse.

**Coverage (Abdeckung)**
Anders als SoV: In wie vielen der gesammelten **Antworten** kommt die Marke mindestens einmal vor? „Nike: 45 von 60 Antworten (75 %)". Diese Zahl ist oft aussagekräftiger als SoV, weil sie nicht durch Vielfachnennungen verzerrt wird.

**Rank / Prominenz**
An welcher Stelle innerhalb einer Antwort wird die Marke genannt? `rank = 1` heißt: als Erstes. Eine Marke, die immer an Position 1 steht, ist prominenter als eine, die stets als Achte auftaucht — auch bei gleicher Nennungshäufigkeit.

**Aspect (Aspekt)**
Worum ging es bei der Nennung? Qualität, Preis, Empfehlung, Bekanntheit, Funktionen.

**Excerpt (Auszug)**
Der wörtliche Satz aus der Antwort, in dem die Marke vorkommt — der Beleg für die Einschätzung.

---

## 3. Installation und Start

### Voraussetzungen

- **Python 3.10 oder neuer.** Wichtig: Das Script nutzt die Schreibweise `str | None`, die es erst ab Python 3.10 gibt. Mit älteren Versionen startet es nicht.
- Ein **Langdock API-Key**.
- Eine Internetverbindung.

### Installation (einmalig)

```bash
pip install -r requirements.txt
```

Das installiert die vier benötigten Bibliotheken (`requirements.txt`):

| Bibliothek | Aufgabe |
|---|---|
| `streamlit` | Baut die Web-Oberfläche |
| `pandas` | Tabellenverarbeitung |
| `plotly` | Diagramme |
| `requests` | Kommunikation mit der Langdock-API |

### Start

```bash
streamlit run app.py
```

Der Browser öffnet sich automatisch, üblicherweise auf `http://localhost:8501`.

### Region umstellen

Standardmäßig läuft alles über die EU-Server. Falls Ihr Langdock-Workspace in den USA liegt:

```bash
LANGDOCK_REGION=us streamlit run app.py
```

Der Code liest diese sogenannte **Umgebungsvariable** in Zeile 137 aus. Eine Umgebungsvariable ist eine Einstellung, die man dem Programm beim Start mitgibt, ohne den Code zu ändern. Ist keine gesetzt, gilt `eu`.

> ⚠️ Die **Agent-API** (Websuche) hat bewusst **keine Region in der Adresse** (Zeile 143). Diese Umstellung betrifft nur die drei direkten Anbieter-Endpoints.

### Dateien, die beim Betrieb entstehen

| Datei / Ordner | Inhalt | Im Git? |
|---|---|---|
| `brand_visibility.log` | Technisches Protokoll jedes API-Aufrufs | Nein (`.gitignore`) |
| `support_evidence.jsonl` | Vollständige Mitschriften der Agent-API-Aufrufe für Support-Tickets | Nein |
| `results/` | Lokal gespeicherte CSV-Dateien | Nein |
| `key.txt` | Falls Sie den Key dort ablegen | Nein — **bewusst gesperrt** |

---

## 4. Der Ablauf aus Nutzersicht — die 5 Schritte

Die App führt durch einen festen Ablauf. Ganz unten im Code (Zeilen 4269–4280) steht dafür ein simpler **Router**: Er schaut nach, in welchem Schritt man gerade ist, und zeigt die passende Seite an.

```
Schritt 1  Einrichtung        API-Key, Modell, Thema, Marken
    ↓
Schritt 2  Fragen prüfen      Generierte Fragen bearbeiten
    ↓
Schritt 3  Runs konfigurieren Wie oft? Wie viele parallel? Websuche?
    ↓
    ⚙️  PHASE 1 — Antworten sammeln  (viele API-Calls)
    ↓
Schritt 4  Rohdaten prüfen    Kontrollpunkt vor den Analysekosten
    ↓
    ⚙️  PHASE 2 — Marken & Sentiment (wenige, aber teure API-Calls)
    ↓
Schritt 5  Ergebnisse         Diagramme, Tabellen, Export
```

### Schritt 1 — Einrichtung *(Code: Zeile 2307)*

Hier legen Sie fest:
- **API-Key** — wird nur im Arbeitsspeicher gehalten, nicht gespeichert.
- **Websuche an/aus** — standardmäßig **an**. Das ist eine folgenreiche Entscheidung, weil sie den kompletten technischen Weg umschaltet (siehe unten).
- **Modell** — die Liste wird live aus Ihrem Workspace geladen.
- **Fragen-Modus** — automatisch generieren (Thema angeben) oder eigene Fragen tippen.
- **Brand-Erkennung** — eigene Marken vorgeben oder alle Marken automatisch erkennen lassen.
- **Verbindung testen** — ein einzelner Mini-Aufruf, der Key und Modellnamen prüft, bevor ein langer Lauf startet. **Diesen Button sollten Sie immer benutzen.**

### Schritt 2 — Fragen prüfen *(Zeile 2594)*

Ein großes Textfeld, eine Frage pro Zeile. Sie können frei bearbeiten, löschen, ergänzen. Nur was hier steht, wird tatsächlich gefragt. Leerzeilen werden ignoriert.

### Schritt 3 — Runs konfigurieren *(Zeile 2647)*

Die wichtigste Seite für Kosten und Laufzeit:
- **Runs pro Frage** (1–100): Wie oft jede Frage wiederholt wird.
- **Modelle**: Mehrfachauswahl möglich — jede Frage wird dann mit jedem Modell gestellt.
- **Parallele API-Calls** (1–10, Standard 2): Wie viele Anfragen gleichzeitig.
- **Extended Thinking**: Das Modell denkt länger nach. Nur für Modelle, die das können.
- **Kurzantwort-Modus**: Das Modell antwortet nur mit einer Stichpunktliste „Marke — ein Satz". Deutlich billiger und schneller.
- **Markt/Region**: z. B. „Deutschland". Wird über den Prompt gesteuert, da Langdock keinen Standort-Parameter kennt.
- **Max. Tokens** und **Pause zwischen Calls** — nur ohne Websuche relevant.

Die Kostenformel wird live angezeigt:
> **Fragen × Runs × Modelle = Anzahl Sammel-Calls**

### Schritt 4 — Rohdaten-Kontrollpunkt *(Zeile 3383)*

Bewusst als Zwischenstopp eingebaut: Die teuren Sammel-Calls sind bezahlt, die Analyse-Calls noch nicht. Hier sehen Sie:
- Wie viele Antworten gesammelt wurden, wie viele fehlschlugen.
- **Wurde die Websuche tatsächlich genutzt?** — mit drei Belegarten (siehe **Abschnitt 6.14**).
- Alle Antworten im Volltext, inklusive der zitierten Quellen.
- Export als CSV oder JSON — **auch ohne Analyse**.

Von hier aus: Analyse starten, Einstellungen ändern (gleiche Fragen, anderes Modell) oder neu beginnen.

### Schritt 5 — Ergebnisse *(Zeile 3601)*

Fünf Tabs:
| Tab | Inhalt |
|---|---|
| **Share of Voice** | Balkendiagramm der Nennungsanteile, Prominenz-Tabelle, Sentiment-Heatmap Marke × Frage |
| **Sentiment** | Prozentverteilung je Marke, gestapeltes Balkendiagramm, alle Belegzitate |
| **Alle Antworten** | Volltexte, filterbar nach Frage |
| **Rohdaten** | Tabelle mit Tokenverbrauch je Antwort |
| **Laufzeit** | Wie lange dauerten die Calls? Histogramm und Detailtabelle |

Darüber ein **Konfidenz-Filter**, der alle Ansichten gleichzeitig einschränkt.

---

## 5. Die Landkarte: Wie das Script aufgebaut ist

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

## 6. Der Code im Detail — Abschnitt für Abschnitt

### 6.1 Kopf und Imports *(Zeilen 1–30)*

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

### 6.2 Logging *(Zeilen 32–42)*

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

### 6.3 Support-Protokoll *(Zeilen 45–60 + Datei `langdock_evidence.py`)*

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

### 6.4 Seitenkonfiguration *(Zeilen 62–69)*

```python
st.set_page_config(page_title="LLM Brand Visibility", page_icon="📊", layout="wide")
```

Browser-Tab-Titel, Icon, und `layout="wide"` für die volle Bildschirmbreite — bei so vielen Diagrammen sinnvoll.

### 6.5 Das Gedächtnis: Session State *(Zeilen 71–103)*

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

### 6.6 Zweisprachigkeit *(Zeilen 105–132)*

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

### 6.7 API-Adressen und Grundeinstellungen *(Zeilen 134–171)*

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

### 6.8 Umgang mit „Modell akzeptiert keine Temperatur" *(Zeilen 173–186)*

```python
_TEMPERATURE_UNSUPPORTED: set[str] = set()
```

Manche neuere Modelle — darunter ausgerechnet das Analyse-Modell — lehnen den Temperatur-Parameter mit Fehler 400 ab („`temperature` is deprecated for this model").

Das Tool **lernt das zur Laufzeit**: Beim ersten solchen Fehler wird der Modellname in dieser Liste vermerkt; alle weiteren Aufrufe lassen den Parameter dann von vornherein weg. So kostet die Eigenheit **einen** Fehlversuch statt einen pro Aufruf.

**Was ist ein `set`?** Eine Menge ohne Reihenfolge und ohne Duplikate — ideal für Ja/Nein-Merklisten wie diese.

### 6.9 Tote Modelle *(Zeilen 188–231)*

```python
_dead_models: set[str] = set()

def mark_model_dead(model_id): _dead_models.add(model_id)
def is_model_dead(model_id):   return model_id in _dead_models
```

Wenn ein Modell endgültig abgelehnt wurde, wird es für den Rest des Laufs übersprungen — sofort und ohne weitere Anfrage.

**Der wichtige Teil steht im Kommentar (Zeilen 202–207):** Diese Sperre gilt **pro Modell**, nicht für den ganzen Lauf. Früher brach ein einziges veraltetes Modell den kompletten Lauf ab, sodass ein Zwei-Modell-Vergleich total ausfiel, obwohl das zweite Modell einwandfrei funktionierte. Jetzt läuft der Rest weiter.

### 6.10 Die Notbremse („Circuit Breaker") *(Zeilen 233–309)*

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

### 6.11 Batch-Einstellungen für die Analyse *(Zeilen 311–330)*

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

### 6.12 Die zwei Modellkataloge — das komplizierteste Thema im Script *(Zeilen 333–855)*

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

### 6.13 `call_langdock()` — das Herzstück *(Zeile 863)*

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
Die Unterscheidung, die in **Abschnitt 6.10** beschrieben wurde: dauerhaftes Limit → alles stoppen; vorübergehendes → warten und wiederholen. Die Wartezeiten müssen das rollierende Minutenfenster abdecken, deshalb starten sie bei 15 Sekunden.

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

### 6.14 `call_langdock_agent()` — der Websuche-Weg *(Zeile 1134)*

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

### 6.15 Verbindungstest *(Zeile 1427)*

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

### 6.16 Fragen generieren *(Zeile 1478)*

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

### 6.17 Eine Frage stellen *(Zeile 1575)*

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

### 6.18 Textwerkzeuge *(Zeilen 1651–1688)*

**`_strip_html(text, max_len=600)`** — Entfernt HTML-Tags und mehrfache Leerzeichen aus Fehlerantworten. Manche Serverfehler (502, 503) liefern eine komplette HTML-Seite, die das Log sonst zumüllen würde.

Warum 600 und nicht 200 Zeichen? Weil Langdocks „ungültiges Modell"-Meldung die Liste der akzeptierten Modelle erst **nach etwa 100 Zeichen Vorrede** nennt. Ein Schnitt bei 200 hätte genau die Information abgeschnitten, die man zur Fehlerbehebung braucht.

**`_strip_citation_markers(text)`** — Entfernt die Zitat-Marker und zählt sie (siehe 6.14). Danach wird noch aufgeräumt:
```python
clean = re.sub(r"[ \t]+([.,;:!?])", r"\1", clean)  # " ." → "."
clean = re.sub(r"[ \t]{2,}", " ", clean)           # doppelte Leerzeichen
```
Sonst blieben nach dem Entfernen unschöne Lücken vor Satzzeichen stehen.

### 6.19 Die JSON-Reparatur *(Zeilen 1690–1781)*

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

### 6.20 Die Markenanalyse — Phase 2 im Detail *(Zeilen 1782–2077)*

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

### 6.21 Ergebnisse in eine Tabelle bringen *(Zeile 2080)*

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

### 6.22 Markenerkennung ohne KI *(Zeilen 2100–2145)*

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

### 6.23 Export *(Zeilen 2147–2225)*

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

### 6.24 Die Tutorials *(Zeilen 2227–2300)*

```python
_TUTORIALS = {
    1: ("❓ Anleitung — Schritt 1: Einrichtung", "❓ Tutorial — Step 1: Setup", """…de…""", """…en…"""),
    ...
}
```

Ein Nachschlagewerk mit einem Eintrag je Schritt, jeweils Titel und Text in beiden Sprachen. `render_tutorial(step)` (Zeile 2294) zeigt sie in einem **eingeklappten** Aufklappbereich — Hilfe für neue Nutzer, ohne Erfahrene zu stören.

### 6.25 Oberfläche Schritt 1 *(Zeile 2307)*

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

### 6.26 Oberfläche Schritt 2 *(Zeile 2594)*

Ein großes Textfeld, 450 Pixel hoch. Darunter eine Live-Zählung mit Differenzanzeige:

```python
delta = n_lines - orig
delta_str = tr(f" ({'+' if delta >= 0 else ''}{delta} gegenüber generiert)", …) if delta != 0 else ""
```

Man sieht also sofort „23 Fragen (+3 gegenüber generiert)".

### 6.27 Oberfläche Schritt 3 *(Zeile 2647)*

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

### 6.28 Phase 1 — der Sammellauf *(Zeile 2984)*

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

### 6.29 Phase 2 — die Analyse *(Zeile 3236)*

```python
reset_run_abort()    # ein Limit während des Sammelns darf die Analyse nicht blockieren
reset_dead_models()  # ebenso ein stillgelegtes Sammelmodell — die Analyse nutzt ihr eigenes
by_index, summary, usage, analysis_err = analyze_dataset(api_key, raw_answers, brands, lang)
```

Die beiden Rücksetzungen sind wichtig: Die Analyse läuft mit einem **anderen** Modell. Ein Problem aus Phase 1 darf sie nicht mitreißen.

Danach werden die Ergebnisse in der **ursprünglichen Reihenfolge** wieder zusammengesetzt und mit den Marken angereichert. Im manuellen Modus filtert `_normalize_brands` auf die vorgegebene Liste — alle anderen gefundenen Marken landen in `unlisted_brands` und werden in Schritt 5 unter „Ebenfalls genannt" angezeigt, statt zu verschwinden.

### 6.30 Oberfläche Schritt 4 — Rohdaten *(Zeile 3383)*

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

### 6.31 Oberfläche Schritt 5 — Ergebnisse *(Zeile 3601)*

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

### 6.32 Der Router *(Zeilen 4269–4280)*

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

## 7. Der Datenfluss auf einen Blick

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

## 8. Potenzielle Probleme und Fallstricke

Dieser Abschnitt ist bewusst ausführlich. Vieles davon ist **kein Programmierfehler**, sondern eine bewusste Abwägung oder eine Eigenheit der zugrunde liegenden APIs — man sollte es aber kennen, bevor man Ergebnisse weitergibt.

### 8.1 Datenverlust

| Problem | Erklärung | Was tun |
|---|---|---|
| **Browser-Tab schließen = alles weg** | Alle Daten liegen im Session State, also nur im Arbeitsspeicher. Es gibt keine automatische Speicherung. | Nach Phase 1 (Schritt 4) **immer** exportieren. Der Export dort ist vollwertig und braucht keine Analyse. |
| **Ein Klick reißt einen laufenden Sammelvorgang ab** | Streamlit startet das Script bei jeder Interaktion neu. Genau so funktioniert der Stop-Button — jeder andere Klick tut aber dasselbe. | Während Phase 1 nichts anklicken. Falls doch: Schritt 3 bietet die gesammelten Antworten zur Weiterverwendung an. |
| **Streamlit-Server neu gestartet** | Alle Sitzungen verlieren ihren Zustand. | Exportieren. |

### 8.2 Kosten laufen aus dem Ruder

| Problem | Erklärung |
|---|---|
| **Die Schieberegler erlauben extreme Werte** | 200 Fragen × 100 Runs × mehrere Modelle = über 20.000 Aufrufe. Es gibt **keine Sicherheitsabfrage** und keine Obergrenze. Die Zahl wird angezeigt — man muss sie lesen. |
| **Kein Tokenverbrauch bei Websuche** | Die Agent-API meldet nichts. Alle Tabellen zeigen 0. Die tatsächlichen Kosten sieht man nur im Langdock-Dashboard. |
| **Das Analyse-Modell ist das teuerste im Lauf** | Claude Opus ist fest eingestellt (Zeile 159). Bei großen Datensätzen ist Phase 2 spürbar teuer, obwohl es nur wenige Aufrufe sind. |
| **Mehrfachauswahl von Modellen multipliziert** | Zwei Modelle = doppelte Kosten. Der Regler „Runs" wirkt zusätzlich multiplikativ. |

**Empfehlung für einen ersten Lauf:** 5 Fragen × 1 Run × 1 Modell. Erst wenn die Kette funktioniert, hochskalieren.

### 8.3 Die Modell-IDs

Der häufigste Fehlerherd überhaupt.

| Problem | Symptom | Lösung |
|---|---|---|
| **Zwei getrennte Kataloge** | Modell funktioniert mit Websuche, aber nicht ohne (oder umgekehrt) | Nach dem Umschalten der Websuche das Modell neu wählen |
| **Widersprüchliche Server** | Dasselbe Modell schlägt sporadisch mit „is not available" fehl | Das Tool wiederholt bereits automatisch bis zu 4×. Bei anhaltendem Fehler: 🔄 Liste neu laden |
| **Deployment-IDs ändern sich** | Ein früher funktionierendes Modell verschwindet | Neu laden, neu auswählen. Nichts ist fest im Code hinterlegt |
| **Analyse-Modell fest verdrahtet** | Phase 2 schlägt komplett fehl, Phase 1 lief einwandfrei | Zeile 159 auf ein verfügbares Modell ändern |
| **Freitext-Modellfeld** | Bei fehlender Modellliste muss man den Namen exakt kennen | Erst API-Key prüfen, dann Verbindungstest |

### 8.4 Rate Limits und Zeitverhalten

| Problem | Erklärung |
|---|---|
| **Hohe Parallelität löst 429 aus** | Standard ist 2 Arbeiter. Bei 10 steigt das Risiko deutlich. Das Tool wartet dann 15/30/60/120 Sekunden — der Lauf wirkt eingefroren, arbeitet aber. |
| **Ein Lauf kann sehr lange dauern** | Websuche-Aufrufe brauchen oft 30–120 Sekunden. 120 Aufrufe bei 2 parallel ≈ 40–120 Minuten. |
| **Die ETA ist nur eine Schätzung** | Sie extrapoliert die bisherige Durchschnittsgeschwindigkeit. Ein einzelnes Rate Limit wirft sie komplett um. |
| **Timeouts trotz allem** | 180 s bzw. 240 s. Unter starker Last kann das bei manchen Reasoning-Modellen zu knapp sein. |

### 8.5 Qualität der Ergebnisse

Das sind die Punkte, die man kennen **muss**, bevor man Ergebnisse präsentiert.

| Problem | Erklärung | Gegenmaßnahme |
|---|---|---|
| **Die Websuche ist nicht erzwingbar** | `capabilities.webSearch` stellt das Werkzeug nur bereit. Kleinere Modelle nutzen es trotz ausdrücklicher Anweisung manchmal nicht. | Die Prüfung in Schritt 4 ernst nehmen. Bei roter Meldung: stärkeres Modell wählen. |
| **Antworten werden für die Analyse auf 2.000 Zeichen gekürzt** | 70 % Anfang + 30 % Ende. Marken **in der Mitte** langer Antworten können übersehen werden. | `ANALYSIS_ANSWER_CHARS` erhöhen (Zeile 161) oder Kurzantwort-Modus nutzen. |
| **Das Sentiment ist ein KI-Urteil** | Ein Modell entscheidet über positiv/neutral/negativ. Das ist nicht objektiv und nicht perfekt reproduzierbar — auch wenn Temperatur 0 hilft. | Konfidenz-Filter nutzen, Belegzitate stichprobenartig prüfen. |
| **Fuzzy-Markenabgleich kann falsch treffen** | Teilstring-Suche in beide Richtungen ab 3 Zeichen: „Apple" trifft auch „Applebee's". | Bei kurzen/generischen Markennamen die Rohdaten kontrollieren. |
| **Der Markt ist nur eine Prompt-Bitte** | Langdock hat keinen Standort-Parameter. Es gibt keine Garantie für marktspezifische Quellen. | Ergebnisse entsprechend vorsichtig interpretieren. |
| **Temperatur 0.7 erzeugt Streuung** | Absicht — sonst wären Wiederholungen wertlos. Aber: Zwei Läufe mit gleicher Konfiguration liefern **nicht** dieselben Zahlen. | Für Vergleiche über die Zeit immer dieselbe Konfiguration und ausreichend viele Runs verwenden. |
| **Wenige Runs = keine Statistik** | Bei 1 Run pro Frage ist jedes Ergebnis eine Einzelbeobachtung. | Mindestens 3, besser 5–10 Runs für belastbare Aussagen. |
| **Automatischer Modus erzeugt viele Marken** | Ohne vorgegebene Liste erkennt das Modell alles Markenähnliche — auch generische Begriffe. Der Long Tail wird unübersichtlich. | Für gezielte Fragestellungen den manuellen Modus verwenden. |
| **Im manuellen Modus zeigen die Diagramme nur Ihre Marken** | Alle anderen werden herausgefiltert. Sie erscheinen nur unter „Ebenfalls genannt" (max. 25 sichtbar). | Für Wettbewerbsanalysen den automatischen Modus einsetzen. |
| **Die Gesamtzusammenfassung sieht nur die Top 20** | `most_common(20)` in `_summarize_dataset`. | Die Diagramme zeigen alles — die Zusammenfassung ist nur ein Einstieg. |
| **Kurzantwort-Modus verzerrt die Tonalität** | Erzwungene Stichpunkte sind nicht das, was ein realer Nutzer sähe. | Für Sentiment-Analysen ausgeschaltet lassen. |

### 8.6 Technische und betriebliche Grenzen

| Problem | Erklärung |
|---|---|
| **Globale Sperren gelten prozessweit, nicht pro Sitzung** | `_dead_models`, `_run_abort_reason` und `_TEMPERATURE_UNSUPPORTED` sind normale Modulvariablen. Wenn mehrere Personen **dieselbe** Streamlit-Instanz benutzen, teilen sie sich diese Zustände. Wenn bei Person A ein Budgetlimit greift, kann das auch Person B ausbremsen. **Das Tool ist für den Einzelplatzbetrieb gedacht.** |
| **Keine Zugangsbeschränkung** | Streamlit hat von Haus aus keine Anmeldung. Wer die Adresse kennt, kann die App benutzen. Für eine Bereitstellung im Netzwerk muss ein vorgeschalteter Schutz eingerichtet werden. |
| **Die Logdatei wächst unbegrenzt** | Keine Rotation. Bereits 2,1 MB im Repository-Stand. Regelmäßig aufräumen. |
| **`support_evidence.jsonl` wächst ebenfalls** | Und enthält vollständige Anfrageinhalte. |
| **Python 3.10 ist Mindestvoraussetzung** | Wegen der Schreibweise `str \| None`. Ältere Versionen starten nicht. |
| **Bei über ~40 Marken wird der Sentiment-Tab unlesbar** | Eine Bildschirmspalte je Marke. |
| **Der Fortschrittsbalken sagt „von 4", es gibt aber 5 Schritte** | Kosmetische Ungenauigkeit — Schritt 5 zeigt Ergebnisse und wird nicht mitgezählt. |
| **Eine lokale Variable heißt wie eine globale** | In `render_step4` (Zeile 3414) heißt eine Hilfsvariable `evidence` — genau wie der globale `EvidenceRecorder` aus Zeile 60. Innerhalb dieser Funktion überdeckt sie ihn. Das ist funktional unbedenklich (der Recorder wird dort nicht benutzt), beim Lesen aber verwirrend. |

### 8.7 Was das Tool bewusst **nicht** kann

Damit keine falschen Erwartungen entstehen:

- **Keine Zeitreihen.** Jeder Lauf steht für sich. Wer Entwicklungen über Monate messen will, muss die Exporte selbst zusammenführen.
- **Keine Konkurrenzanalyse über Marken hinaus.** Es misst Nennungen, nicht Marktanteile oder Umsätze.
- **Keine Erklärung, *warum* eine Marke genannt wird.** Der `aspect`- und `reason`-Wert gibt Hinweise, mehr nicht.
- **Keine Steuerung, welche Quellen durchsucht werden.** Das entscheidet die Suchmaschine hinter der Agent-API.
- **Keine Garantie auf Reproduzierbarkeit.** KI-Modelle ändern sich, ohne dass ihre ID sich ändert.

---

## 9. Kosten und Laufzeit realistisch einschätzen

### Die Formel

```
Sammel-Calls  = Fragen × Runs pro Frage × Anzahl Modelle
Analyse-Calls = aufgerundet(Antworten ÷ 40)   [+1 falls mehr als ein Batch]
```

### Beispielszenarien

| Szenario | Fragen | Runs | Modelle | Sammel-Calls | Analyse-Calls | Laufzeit* |
|---|---|---|---|---|---|---|
| **Schnelltest** | 5 | 1 | 1 | 5 | 1 | ~2–5 min |
| **Kleine Erhebung** | 20 | 3 | 1 | 60 | 2+1 | ~20–45 min |
| **Modellvergleich** | 20 | 3 | 3 | 180 | 5+1 | ~1–3 h |
| **Große Erhebung** | 50 | 5 | 2 | 500 | 13+1 | ~4–10 h |

\* Grobe Spanne bei 2 parallelen Aufrufen und aktiver Websuche. Ohne Websuche deutlich schneller.

### Kosten senken

1. **Kurzantwort-Modus** — der größte Hebel. Statt 800 Ausgabe-Tokens je Antwort oft nur 150.
2. **Websuche abschalten**, wenn die Frage nicht tagesaktuell ist.
3. **Weniger Runs.** Von 10 auf 5 halbiert die Kosten und die Statistik bleibt meist tragfähig.
4. **Ein günstigeres Sammelmodell** verwenden — die Analyse läuft ohnehin auf dem starken Modell.
5. **„Sentiment neu analysieren"** in Schritt 5 statt eines kompletten Neulaufs, wenn nur die Analyse angepasst werden soll.
6. **Immer zuerst den Verbindungstest** — er kostet einen Aufruf statt eines fehlgeschlagenen Laufs.

---

## 10. Sicherheit und Datenschutz

### Der API-Key

**Was das Script richtig macht:**
- Der Key wird über ein Passwortfeld eingegeben (`type="password"`) und ist damit nicht im Klartext sichtbar.
- Er wird **nie** in `brand_visibility.log` geschrieben.
- In `support_evidence.jsonl` steht nur ein Hash-Fingerabdruck.
- Sollte der Key jemals in einer Fehlerantwort zurückgespiegelt werden, ersetzt ihn `redact()` durch `<REDACTED_API_KEY>`.
- `key.txt` ist in `.gitignore` gesperrt, mit einem ausdrücklichen Kommentar: *„API key — never commit"*.

**Worauf Sie trotzdem achten müssen:**
- Der Key liegt im Session State, also im Arbeitsspeicher des Servers. Wer Zugriff auf den Server hat, kommt potenziell heran.
- Er wird bei jedem Neustart der App neu abgefragt — bewusst so, aber gewöhnungsbedürftig.
- **Prüfen Sie vor jedem `git push`, dass `key.txt` nicht doch mitgeht:** `git status` sollte sie nicht auflisten.

### Was in Dateien landet

| Datei | Enthält | Risiko |
|---|---|---|
| `brand_visibility.log` | Statuscodes, Modellnamen, Zeiten, gekürzte Fehlermeldungen | Gering — kein Key, keine vollen Prompts |
| `support_evidence.jsonl` | **Vollständige Anfrageinhalte inklusive aller Prompts und Fragen**, ungekürzte Fehlerantworten, Antwort-Header | ⚠️ **Erhöht** — vor Weitergabe hineinschauen |
| `results/*.csv` | Fragen, Antworten (auf 500 Zeichen gekürzt), Markenzuordnungen | Abhängig vom Thema |

Der Code weist an Zeile 1210 selbst darauf hin:
> *„…und es ENTHÄLT DEN PROMPT-TEXT — einen Blick wert, bevor diese Datei nach außen gegeben wird."*

### Was an Langdock übertragen wird

Jede Frage, jeder Prompt und jede Marke, die Sie eingeben, geht an Langdock und von dort an den jeweiligen KI-Anbieter. Bei aktiver Websuche werden zusätzlich Suchanfragen an eine Suchmaschine gestellt.

**Praktische Konsequenz:** Keine vertraulichen Produktnamen, unveröffentlichten Projektbezeichnungen oder personenbezogenen Daten in die Fragen schreiben. Die Region `eu` sorgt für EU-Verarbeitung bei den drei direkten Endpoints — **die Agent-API hat allerdings keine Region in der Adresse** (Zeile 143). Wo genau sie verarbeitet, sollte bei Bedarf mit Langdock geklärt werden.

### Betrieb im Netzwerk

Streamlit bringt **keine Anmeldung** mit. Standardmäßig läuft die App nur lokal (`localhost`). Wird sie im Firmennetz oder im Internet bereitgestellt, kann jeder, der die Adresse kennt, sie benutzen — und dabei den API-Key eingeben, den er selbst mitbringt, oder die Ergebnisse anderer sehen, wenn sie sich eine Instanz teilen. Für eine solche Bereitstellung braucht es einen vorgeschalteten Zugriffsschutz.

---

## 11. Stellschrauben — was man wo ändern kann

Alle wichtigen Werte stehen als Konstanten am Anfang der Datei. Zum Anpassen genügt es, die Zahl zu ändern und die App neu zu starten.

| Was | Zeile | Standard | Wirkung beim Ändern |
|---|---|---|---|
| `LANGDOCK_REGION` | 137 | `eu` | Umgebungsvariable beim Start, nicht im Code ändern |
| `REQUEST_TIMEOUT` | 151 | 180 s | Höher, wenn Modelle regelmäßig in den Timeout laufen |
| `AGENT_STREAM_TIMEOUT` | 152 | 240 s | Höher bei sehr aufwendigen Suchen |
| `MAX_TOKENS` | 153 | 8000 | Standardwert des Reglers in Schritt 3 |
| `QUESTION_MAX_TOKENS` | 154 | 16000 | Budget der Fragengenerierung |
| **`ANALYSIS_MODEL`** | **159** | `claude-opus-4-8` | **Wichtigste Stellschraube** — ändern, falls das Modell nicht verfügbar ist |
| `DATASET_ANALYSIS_MAX_TOKENS` | 160 | 16000 | Ausgabebudget je Analyse-Batch |
| `ANALYSIS_ANSWER_CHARS` | 161 | 2000 | Höher = weniger Kürzung, mehr Kosten |
| `COLLECTION_TEMPERATURE` | 169 | 0.7 | Niedriger = einheitlichere Antworten (Wiederholungen verlieren an Wert) |
| `QUESTION_TEMPERATURE` | 170 | 0.8 | Niedriger = konventionellere Fragen |
| `ANALYSIS_TEMPERATURE` | 171 | 0.0 | Nicht erhöhen — Reproduzierbarkeit geht verloren |
| `_FATAL_LIMIT_RE` | 249 | siehe Code | Suchmuster für endgültige Limits. Erweitern nur mit Bedacht |
| `ANALYSIS_BATCH_MAX_ANSWERS` | 312 | 40 | Kleiner = mehr Aufrufe, geringeres Abschneide-Risiko |
| `ANALYSIS_BATCH_MAX_INPUT_TOKENS` | 313 | 40000 | Eingabegrenze je Batch |
| `ANALYSIS_ANSWER_HEAD_FRAC` | 317 | 0.7 | Verhältnis Anfang/Ende bei der Kürzung |
| `AGENT_MODELS_TTL` | 480 | 60 s | Wie lange die Modellliste zwischengespeichert wird |
| Agent-`instructions` | 1170 | siehe Code | Der Text, der das Modell zum Suchen bewegt |
| Prompts der Fragengenerierung | 1487 | siehe Code | Welche Art Fragen erzeugt wird |
| Prompts der Antwortsammlung | 1585 | siehe Code | **Nur mit Bedacht ändern** — beeinflusst das Messergebnis direkt |
| Analyse-Prompt | 1828 | siehe Code | Welche Felder erkannt werden |

---

## 12. Troubleshooting — häufige Fehlermeldungen

### „API-Key ungültig oder abgelaufen (401)."
Der Key ist falsch, abgelaufen oder wurde widerrufen. Auf Leerzeichen am Anfang oder Ende prüfen — die entstehen beim Kopieren leicht.

### „Zugriff verweigert (403). API-Key prüfen."
Der Key ist gültig, hat aber keine Berechtigung für dieses Modell oder diesen Endpoint. In den Langdock-Workspace-Einstellungen nachsehen.

### „Endpunkt nicht gefunden (404). Region prüfen: 'eu'."
Ihr Workspace liegt vermutlich in einer anderen Region. Neustart mit:
```bash
LANGDOCK_REGION=us streamlit run app.py
```

### „Ungültige Anfrage (400) — häufig ein nicht verfügbarer Modell-Name."
Der häufigste Fehler. In dieser Reihenfolge vorgehen:
1. 🔄 neben der Modellauswahl drücken (Liste neu laden).
2. Modell erneut auswählen — die Schreibweise kann sich geändert haben.
3. Websuche-Schalter prüfen: An und Aus verwenden **verschiedene Kataloge**.
4. „Verbindung testen" — die Fehlermeldung nennt oft die akzeptierten Modelle.

### „Modell 'X' wurde vom Katalog abgelehnt und ist nicht mehr verfügbar."
Alle vier Versuche sind gescheitert. Das Modell wurde für diesen Lauf stillgelegt. **Andere ausgewählte Modelle laufen weiter.** Liste neu laden, neu auswählen.

### „Abgebrochen — API-Limit erreicht: …"
Die Notbremse hat gegriffen — meist ein erreichtes Budget-/Ausgabelimit des Workspace. Weitere Versuche würden nichts bringen. **Die bis dahin gesammelten Antworten bleiben erhalten** und können in Schritt 4 exportiert oder analysiert werden. Zuerst das Limit in Langdock prüfen.

### „Token-Budget erschöpft (max_tokens=8000)."
Ein Reasoning-Modell hat sein Budget beim internen Nachdenken verbraucht. In Schritt 3 „Max. Tokens" erhöhen (bis 16000). Erscheint nur ohne Websuche — bei der Agent-API gibt es diesen Regler nicht.

### „Timeout nach 180s." / „Timeout nach 240s."
Das Modell hat zu lange gebraucht. Mögliche Ursachen: hohe Last, sehr aufwendige Websuche, sehr langer Prompt. Weniger parallele Aufrufe einstellen oder in Zeile 151/152 höher setzen.

### „Es konnten keine Fragen aus der Antwort extrahiert werden."
Das Modell hat auf die Aufforderung nach einem JSON-Array mit Prosa geantwortet. Ein anderes Modell probieren, oder in Schritt 1 auf eigene Fragen umstellen. Im Log steht unter `PARSED ZERO` der Anfang der tatsächlichen Antwort.

### „Ein Analyse-Batch konnte nicht als JSON gelesen werden (evtl. abgeschnitten)."
Die JSON-Ausgabe des Analyse-Modells wurde vermutlich abgeschnitten. `ANALYSIS_BATCH_MAX_ANSWERS` (Zeile 312) auf z. B. 25 senken und über den Button in Schritt 5 neu analysieren — das kostet keine neuen Sammel-Aufrufe.

### „🔍 Keine Antwort hat das Such-Tool nachweislich genutzt."
Die Modelle haben aus ihrem Trainingswissen geantwortet. Ein stärkeres Modell wählen (kleine Modelle ignorieren die Suchanweisung häufiger) und prüfen, ob die Websuche in Schritt 3 wirklich aktiv war. Bei zeitlosen Fragen ist es kein Fehler — dort ist Suchen tatsächlich unnötig.

### „Keine Brands erkannt."
- **Manueller Modus:** Kommen die Namen wirklich in den Antworten vor? Der Tab „Rohdaten" zeigt die Volltexte. Auf Schreibweisen achten.
- **Automatischer Modus:** Die Fragen sind vermutlich zu abstrakt formuliert, sodass die Antworten gar keine Marken nennen. Fragen konkreter stellen („Welche Anbieter empfiehlst du für …?").

### Der Lauf scheint eingefroren
Erst im Log nachsehen:
```bash
tail -f brand_visibility.log
```
Steht dort `Rate limited (429), waiting …`, arbeitet das Tool und wartet nur eine Sperre ab. Es gibt keine Anzeige dafür in der Oberfläche — das ist ein bekannter Schwachpunkt.

### Jede Logzeile steht mehrfach in der Datei
Sollte durch den Schutz in Zeile 38 nicht passieren. Falls doch: Streamlit vollständig beenden und neu starten.

---

## Anhang: Die Nachbardatei `langdock_evidence.py`

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
