# Glossar — alle Fachbegriffe erklärt

> Nachschlagewerk zum **LLM Brand Visibility Tracker**.
> Zurück zur [README](../README.md) · Weiter zur [Code-Erklärung](CODE.md)

Sie müssen dieses Dokument nicht am Stück lesen. Es ist zum Nachschlagen gedacht: Wenn Ihnen in der README oder in der Code-Erklärung ein Begriff begegnet, den Sie nicht kennen, finden Sie ihn hier.

**Schnellzugriff:**
[Grundbegriffe](#grundbegriffe) · [Technische Begriffe](#technische-begriffe) · [Streamlit](#streamlit-spezifische-begriffe) · [Auswertung](#fachbegriffe-der-auswertung)

---


Diese Begriffe tauchen in der Doku und im Code laufend auf. Sie können den Abschnitt überspringen und bei Bedarf zurückspringen.

### Grundbegriffe

**LLM (Large Language Model)**
Ein „großes Sprachmodell" — die Technik hinter ChatGPT, Claude, Gemini. Ein Programm, das auf riesigen Textmengen trainiert wurde und darauf spezialisiert ist, Text fortzusetzen. Wenn Sie ihm eine Frage stellen, „errät" es die passendste Antwort. Beispiele im Tool: `gpt-5-mini`, `claude-opus-4-8`, `gemini-…`.

**Modell / Modell-ID**
Die konkrete Version eines LLM, angesprochen über einen technischen Namen wie `claude-opus-4-8` oder `gpt-5-mini-eu`. Dieser Name muss **exakt** stimmen — ein Tippfehler führt zu einer Fehlermeldung. Das ist im Code ein großes Thema (siehe [Code-Erklärung 2.12](CODE.md#212-die-zwei-modellkataloge--das-komplizierteste-thema-im-script)).

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
Das Tool bittet die KI ausdrücklich um JSON, damit die Antwort maschinell auswertbar ist. **Problem:** KI-Modelle halten sich nicht immer daran — deshalb gibt es im Code aufwendige „Reparatur-Parser" ([Code-Erklärung 2.19](CODE.md#219-die-json-reparatur)).

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
- den Logging-Schutz in Zeile 39 (sonst würde jede Logzeile mehrfach geschrieben),
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

---

*Teil der Dokumentation des LLM Brand Visibility Trackers · [README](../README.md) · [Code-Erklärung](CODE.md)*
