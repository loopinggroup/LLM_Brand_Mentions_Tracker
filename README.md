# LLM Brand Visibility Tracker

**Misst, wie sichtbar Marken in den Antworten von KI-Modellen sind — und in welchem Ton über sie gesprochen wird.**

---

## Inhalt

- [Was macht dieses Tool?](#was-macht-dieses-tool)
- [Voraussetzungen](#voraussetzungen)
- [Installation](#installation)
- [Start](#start)
- [Bedienung — die 5 Schritte](#bedienung--die-5-schritte)
- [Kosten und Laufzeit realistisch einschätzen](#kosten-und-laufzeit-realistisch-einschätzen)
- [Bekannte Probleme und Fallstricke](#bekannte-probleme-und-fallstricke)
- [Troubleshooting — häufige Fehlermeldungen](#troubleshooting--häufige-fehlermeldungen)
- [Sicherheit und Datenschutz](#sicherheit-und-datenschutz)
- [Stellschrauben — was man wo ändern kann](#stellschrauben--was-man-wo-ändern-kann)
- [Weiterführende Dokumentation](#weiterführende-dokumentation)
- [Projektstruktur](#projektstruktur)
- [Schnellstart-Checkliste](#schnellstart-checkliste)

---

## Was macht dieses Tool?

Immer mehr Menschen suchen nicht mehr bei Google, sondern fragen einen KI-Chatbot („Welche CRM-Software eignet sich für Mittelständler?"). Dessen Antwort entscheidet mit, welche Marken überhaupt in die engere Auswahl kommen. Dieses Tool misst genau das: Es stellt einem oder mehreren KI-Modellen automatisiert viele Fragen zu Ihrem Thema — auf Wunsch mit echter Websuche —, sammelt die Antworten und lässt sie anschließend von einem starken Modell auswerten. Heraus kommen Diagramme und Exportdateien dazu, welche Marken wie oft, wie prominent und wie positiv genannt werden.

**Der Ablauf in vier Schritten:**

- **Fragen erzeugen** — Das Tool lässt ein KI-Modell typische Nutzerfragen zu Ihrem Thema formulieren. Alternativ geben Sie eigene Fragen ein.
- **Antworten sammeln** — Jede Frage wird mehrfach an ein oder mehrere Modelle gestellt (z. B. 20 Fragen × 3 Wiederholungen × 2 Modelle = 120 Antworten). Wiederholungen sind nötig, weil KI-Modelle nicht immer gleich antworten — erst dadurch entsteht eine belastbare Statistik.
- **Marken erkennen** — Ein starkes Modell (Claude Opus) liest alle Antworten und notiert je Nennung: Marke, Tonalität (positiv/neutral/negativ), Position in der Antwort, Belegzitat und Themenaspekt.
- **Auswerten** — Share of Voice, Sentiment-Verteilung, Heatmap Marke × Frage, Laufzeitanalyse. Export als CSV und JSON.

**Technischer Unterbau:** Eine [Streamlit](https://streamlit.io)-App in einer einzigen Python-Datei (`app.py`). Alle KI-Anfragen laufen über **Langdock** als Gateway — ein Zugang für OpenAI-, Anthropic- und Google-Modelle, mit EU-Hosting.

> 📖 **Neu bei den Begriffen?** LLM, Token, Temperatur, Share of Voice und alles Weitere sind im **[Glossar](docs/GLOSSAR.md)** erklärt.

---

## Voraussetzungen

| | |
|---|---|
| **Python** | 3.10 oder neuer — ältere Versionen starten das Script nicht |
| **Langdock API-Key** | Aus Ihrem Langdock-Workspace |
| **Git** | Zum Herunterladen des Repositorys ([Download](https://git-scm.com/downloads)) |
| **Internetverbindung** | Alle Anfragen laufen über die Langdock-API |

Python-Version prüfen:

```bash
python3 --version
```

---

## Installation

### 1. Repository herunterladen

```bash
git clone https://github.com/loopinggroup/LLM_Brand_Mentions_Tracker.git
cd LLM_Brand_Mentions_Tracker
```

> **Ohne Git?** Auf [github.com/loopinggroup/LLM_Brand_Mentions_Tracker](https://github.com/loopinggroup/LLM_Brand_Mentions_Tracker) auf den grünen Button **Code** klicken → **Download ZIP** → entpacken → im Terminal in den entpackten Ordner wechseln.

### 2. Abhängigkeiten installieren

```bash
pip install -r requirements.txt
```

Installiert die vier benötigten Bibliotheken:

| Bibliothek | Aufgabe |
|---|---|
| `streamlit` | Baut die Web-Oberfläche |
| `pandas` | Tabellenverarbeitung |
| `plotly` | Diagramme |
| `requests` | Kommunikation mit der Langdock-API |

> 💡 **Empfehlung — virtuelle Umgebung.** Damit die Pakete dieses Projekts nicht mit anderen Python-Projekten auf Ihrem Rechner kollidieren:
> ```bash
> python3 -m venv .venv
> source .venv/bin/activate        # Windows: .venv\Scripts\activate
> pip install -r requirements.txt
> ```
> Beim nächsten Mal genügt der `source`-Befehl vor dem Start.

### 3. Aktualisieren

Wenn es eine neue Version gibt:

```bash
git pull
pip install -r requirements.txt
```

---

## Start

```bash
streamlit run app.py
```

Der Browser öffnet sich automatisch auf `http://localhost:8501`. Den API-Key geben Sie direkt in der Oberfläche ein — er wird **nicht** gespeichert und muss bei jedem Start neu eingegeben werden.

Beenden: im Terminal `Strg + C`.

### Region umstellen

Standardmäßig laufen die Anfragen über die EU-Server. Falls Ihr Langdock-Workspace in den USA liegt:

```bash
LANGDOCK_REGION=us streamlit run app.py
```

> Das betrifft nur die Analyse (Anthropic-Endpoint). Fragengenerierung und Antwortsammlung laufen über die Agent-API, die keine Region in der Adresse hat.

---

## Bedienung — die 5 Schritte

Die App führt Sie durch einen festen Ablauf. Oben rechts lässt sich jederzeit zwischen Deutsch und Englisch umschalten — das gilt für die Oberfläche **und** für die Sprache, in der die Modelle antworten.

```
Schritt 1  Verbindung & Modelle   API-Key, Modelle (Mehrfachauswahl), Verbindungstest
    ↓
Schritt 2  Fragen & Brands       Fragen generieren/eingeben, Brand-Erkennung
    ↓
Schritt 3  Übersicht & Optionen  Alles auf einen Blick, Websuche, Extended Thinking, Runs
    ↓
    ⚙️  PHASE 1 — Antworten sammeln  (viele API-Calls)
    ↓
Schritt 4  Rohdaten prüfen    Kontrollpunkt vor den Analysekosten
    ↓
    ⚙️  PHASE 2 — Marken & Sentiment (wenige, aber teure API-Calls)
    ↓
Schritt 5  Ergebnisse         Diagramme, Tabellen, Export
```

Alle Eingaben aus Schritt 1–3 bleiben erhalten, wenn Sie zwischen den Schritten vor- und zurückspringen. „Neuen Prozess starten" behält API-Key und Modellauswahl und leert den Rest.

### Schritt 1 — Verbindung & Modelle

- **API-Key** — wird nur im Arbeitsspeicher gehalten, nicht gespeichert.
- **Modelle** — Mehrfachauswahl, live aus Ihrem Workspace geladen (`GET /agent/v1/models`). Jede Frage wird mit jedem gewählten Modell gestellt. Das **zuerst gewählte** Modell generiert in Schritt 2 die Fragen.
- **Verbindung testen** — ein Mini-Aufruf pro Modell (parallel), der Key und Modell-IDs prüft, bevor ein langer Lauf startet. **Diesen Button sollten Sie immer benutzen.**

### Schritt 2 — Fragen & Brands

- **Fragen-Modus** — automatisch generieren (Thema + Anzahl angeben, Button „Fragen generieren") oder eigene Fragen tippen. Das Ergebnis landet in einem großen Textfeld, eine Frage pro Zeile, frei editierbar. Nur was dort steht, wird gefragt.
- **Brand-Erkennung** — eigene Marken vorgeben oder alle Marken automatisch erkennen lassen.

### Schritt 3 — Übersicht & Optionen

Die wichtigste Seite für Kosten und Laufzeit. Sie liest sich von oben nach unten:

1. **Einstellungen prüfen** — je ein Block für Verbindung & Modelle, Fragen und Brand-Erkennung, jeweils mit „Bearbeiten"-Sprung zurück.
2. **Optionen festlegen**
   - *Antwortverhalten:* **Websuche** (standardmäßig **an**; aus = nur Trainingswissen — beide Varianten laufen über die Agent-API, deshalb bleibt die Modellauswahl gültig), **Extended Thinking** (nur wählbar, wenn alle gewählten Modelle es unterstützen), **Kurzantwort-Modus** („Marke — ein Satz", billiger und schneller).
   - *Umfang:* **Runs pro Frage** (1–100) und **parallele Calls** (1–10, Standard 2).
3. **Lauf starten** — Rechnung Fragen × Runs × Modelle, was in Phase 1 und Phase 2 passiert (inklusive des automatisch ermittelten Analyse-Modells), dann der Start-Button.

Die Kostenformel wird live angezeigt:
> **Fragen × Runs × Modelle = Anzahl Sammel-Calls**

### Schritt 4 — Rohdaten-Kontrollpunkt

Bewusst als Zwischenstopp eingebaut: Die teuren Sammel-Calls sind bezahlt, die Analyse-Calls noch nicht. Hier sehen Sie:
- Wie viele Antworten gesammelt wurden, wie viele fehlschlugen.
- **Wurde die Websuche tatsächlich genutzt?** — mit drei Belegarten (Details: [Code-Erklärung 2.14](docs/CODE.md)). Die Spalte „🔍 Websuche erfolgreich · Quellen" zeigt das je Antwort samt Anzahl der gelieferten Quellen.
- Alle Antworten im Volltext, inklusive der zitierten Quellen.
- Export als CSV oder JSON — **auch ohne Analyse**.

Von hier aus: Analyse starten, Einstellungen ändern (gleiche Fragen, anderes Modell) oder neu beginnen.

### Schritt 5 — Ergebnisse

Fünf Tabs:
| Tab | Inhalt |
|---|---|
| **Share of Voice** | Balkendiagramm der Nennungsanteile, Prominenz-Tabelle, Sentiment-Heatmap Marke × Frage |
| **Sentiment** | Prozentverteilung je Marke, gestapeltes Balkendiagramm, alle Belegzitate |
| **Alle Antworten** | Volltexte, filterbar nach Frage |
| **Rohdaten** | Tabelle aller Antworten mit erkannten Marken |
| **Laufzeit** | Wie lange dauerten die Calls? Histogramm und Detailtabelle |

Darüber ein **Konfidenz-Filter**, der alle Ansichten gleichzeitig einschränkt.

---

## Kosten und Laufzeit realistisch einschätzen

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

## Bekannte Probleme und Fallstricke

Dieser Abschnitt ist bewusst ausführlich. Vieles davon ist **kein Programmierfehler**, sondern eine bewusste Abwägung oder eine Eigenheit der zugrunde liegenden APIs — man sollte es aber kennen, bevor man Ergebnisse weitergibt.

### Datenverlust

| Problem | Erklärung | Was tun |
|---|---|---|
| **Browser-Tab schließen = alles weg** | Alle Daten liegen im Session State, also nur im Arbeitsspeicher. Es gibt keine automatische Speicherung. | Nach Phase 1 (Schritt 4) **immer** exportieren. Der Export dort ist vollwertig und braucht keine Analyse. |
| **Ein Klick reißt einen laufenden Sammelvorgang ab** | Streamlit startet das Script bei jeder Interaktion neu. Genau so funktioniert der Stop-Button — jeder andere Klick tut aber dasselbe. | Während Phase 1 nichts anklicken. Falls doch: Schritt 3 bietet die gesammelten Antworten zur Weiterverwendung an. |
| **Streamlit-Server neu gestartet** | Alle Sitzungen verlieren ihren Zustand. | Exportieren. |

### Kosten laufen aus dem Ruder

| Problem | Erklärung |
|---|---|
| **Die Schieberegler erlauben extreme Werte** | 200 Fragen × 100 Runs × mehrere Modelle = über 20.000 Aufrufe. Es gibt **keine Sicherheitsabfrage** und keine Obergrenze. Die Zahl wird angezeigt — man muss sie lesen. |
| **Kein Tokenverbrauch der Sammlung** | Die Agent-API meldet nichts, deshalb gibt es keine Token-Spalten für die Antworten. Die tatsächlichen Kosten sieht man nur im Langdock-Dashboard. |
| **Das Analyse-Modell ist das teuerste im Lauf** | Es wird automatisch das neueste Claude Opus gewählt. Bei großen Datensätzen ist Phase 2 spürbar teuer, obwohl es nur wenige Aufrufe sind. |
| **Mehrfachauswahl von Modellen multipliziert** | Zwei Modelle = doppelte Kosten. Der Regler „Runs" wirkt zusätzlich multiplikativ. |

**Empfehlung für einen ersten Lauf:** 5 Fragen × 1 Run × 1 Modell. Erst wenn die Kette funktioniert, hochskalieren.

### Die Modell-IDs

Der häufigste Fehlerherd überhaupt.

| Problem | Symptom | Lösung |
|---|---|---|
| **Widersprüchliche Server** | Dasselbe Modell schlägt sporadisch mit „is not available" fehl | Das Tool wiederholt bereits automatisch bis zu 4×. Bei anhaltendem Fehler: 🔄 Liste neu laden |
| **Deployment-IDs ändern sich** | Ein früher funktionierendes Modell verschwindet | Neu laden, neu auswählen. Nichts ist fest im Code hinterlegt |
| **Opus-Liste nicht abrufbar** | Schritt 3 zeigt „Standardmodell wird verwendet", Phase 2 schlägt evtl. fehl | Das neueste Opus wird live vom Anthropic-Endpoint gelesen. Klappt das nicht, greift `ANALYSIS_MODEL_FALLBACK` (Zeile 206) — bei Bedarf dort anpassen |
| **Freitext-Modellfeld** | Bei fehlender Modellliste muss man den Namen exakt kennen | Erst API-Key prüfen, dann Verbindungstest |

### Rate Limits und Zeitverhalten

| Problem | Erklärung |
|---|---|
| **Hohe Parallelität löst 429 aus** | Standard ist 2 Arbeiter. Bei 10 steigt das Risiko deutlich. Das Tool wartet dann 15/30/60/120 Sekunden — der Lauf wirkt eingefroren, arbeitet aber. |
| **Ein Lauf kann sehr lange dauern** | Websuche-Aufrufe brauchen oft 30–120 Sekunden. 120 Aufrufe bei 2 parallel ≈ 40–120 Minuten. |
| **Die ETA ist nur eine Schätzung** | Sie extrapoliert die bisherige Durchschnittsgeschwindigkeit. Ein einzelnes Rate Limit wirft sie komplett um. |
| **Timeouts trotz allem** | 180 s bzw. 240 s. Unter starker Last kann das bei manchen Reasoning-Modellen zu knapp sein. |

### Qualität der Ergebnisse

Das sind die Punkte, die man kennen **muss**, bevor man Ergebnisse präsentiert.

| Problem | Erklärung | Gegenmaßnahme |
|---|---|---|
| **Die Websuche ist nicht erzwingbar** | `capabilities.webSearch` stellt das Werkzeug nur bereit. Kleinere Modelle nutzen es trotz ausdrücklicher Anweisung manchmal nicht. | Die Prüfung in Schritt 4 ernst nehmen. Bei roter Meldung: stärkeres Modell wählen. |
| **Antworten werden für die Analyse auf 2.000 Zeichen gekürzt** | 70 % Anfang + 30 % Ende. Marken **in der Mitte** langer Antworten können übersehen werden. | `ANALYSIS_ANSWER_CHARS` erhöhen (Zeile 208) oder Kurzantwort-Modus nutzen. |
| **Das Sentiment ist ein KI-Urteil** | Ein Modell entscheidet über positiv/neutral/negativ. Das ist nicht objektiv und nicht perfekt reproduzierbar — auch wenn Temperatur 0 hilft. | Konfidenz-Filter nutzen, Belegzitate stichprobenartig prüfen. |
| **Fuzzy-Markenabgleich kann falsch treffen** | Teilstring-Suche in beide Richtungen ab 3 Zeichen: „Apple" trifft auch „Applebee's". | Bei kurzen/generischen Markennamen die Rohdaten kontrollieren. |
| **Temperatur 0.7 erzeugt Streuung** | Absicht — sonst wären Wiederholungen wertlos. Aber: Zwei Läufe mit gleicher Konfiguration liefern **nicht** dieselben Zahlen. | Für Vergleiche über die Zeit immer dieselbe Konfiguration und ausreichend viele Runs verwenden. |
| **Wenige Runs = keine Statistik** | Bei 1 Run pro Frage ist jedes Ergebnis eine Einzelbeobachtung. | Mindestens 3, besser 5–10 Runs für belastbare Aussagen. |
| **Automatischer Modus erzeugt viele Marken** | Ohne vorgegebene Liste erkennt das Modell alles Markenähnliche — auch generische Begriffe. Der Long Tail wird unübersichtlich. | Für gezielte Fragestellungen den manuellen Modus verwenden. |
| **Im manuellen Modus zeigen die Diagramme nur Ihre Marken** | Alle anderen werden herausgefiltert. Sie erscheinen nur unter „Ebenfalls genannt" (max. 25 sichtbar). | Für Wettbewerbsanalysen den automatischen Modus einsetzen. |
| **Die Gesamtzusammenfassung sieht nur die Top 20** | `most_common(20)` in `_summarize_dataset`. | Die Diagramme zeigen alles — die Zusammenfassung ist nur ein Einstieg. |
| **Kurzantwort-Modus verzerrt die Tonalität** | Erzwungene Stichpunkte sind nicht das, was ein realer Nutzer sähe. | Für Sentiment-Analysen ausgeschaltet lassen. |

### Technische und betriebliche Grenzen

| Problem | Erklärung |
|---|---|
| **Globale Sperren gelten prozessweit, nicht pro Sitzung** | `_dead_models`, `_run_abort_reason` und `_TEMPERATURE_UNSUPPORTED` sind normale Modulvariablen. Wenn mehrere Personen **dieselbe** Streamlit-Instanz benutzen, teilen sie sich diese Zustände. Wenn bei Person A ein Budgetlimit greift, kann das auch Person B ausbremsen. **Das Tool ist für den Einzelplatzbetrieb gedacht.** |
| **Keine Zugangsbeschränkung** | Streamlit hat von Haus aus keine Anmeldung. Wer die Adresse kennt, kann die App benutzen. Für eine Bereitstellung im Netzwerk muss ein vorgeschalteter Schutz eingerichtet werden. |
| **Die Logdatei wächst unbegrenzt** | Keine Rotation. Bereits 2,1 MB im Repository-Stand. Regelmäßig aufräumen. |
| **`support_evidence.jsonl` wächst ebenfalls** | Und enthält vollständige Anfrageinhalte. |
| **Python 3.10 ist Mindestvoraussetzung** | Wegen der Schreibweise `str \| None`. Ältere Versionen starten nicht. |
| **Bei über ~40 Marken wird der Sentiment-Tab unlesbar** | Eine Bildschirmspalte je Marke. |
| **Eine lokale Variable heißt wie eine globale** | In `render_step4` (Zeile 3238) heißt eine Hilfsvariable `evidence` — genau wie der globale `EvidenceRecorder` aus Zeile 60. Innerhalb dieser Funktion überdeckt sie ihn. Das ist funktional unbedenklich (der Recorder wird dort nicht benutzt), beim Lesen aber verwirrend. |

### Was das Tool bewusst **nicht** kann

Damit keine falschen Erwartungen entstehen:

- **Keine Zeitreihen.** Jeder Lauf steht für sich. Wer Entwicklungen über Monate messen will, muss die Exporte selbst zusammenführen.
- **Keine Konkurrenzanalyse über Marken hinaus.** Es misst Nennungen, nicht Marktanteile oder Umsätze.
- **Keine Erklärung, *warum* eine Marke genannt wird.** Der `aspect`- und `reason`-Wert gibt Hinweise, mehr nicht.
- **Keine Steuerung, welche Quellen durchsucht werden.** Das entscheidet die Suchmaschine hinter der Agent-API.
- **Keine Garantie auf Reproduzierbarkeit.** KI-Modelle ändern sich, ohne dass ihre ID sich ändert.

---

## Troubleshooting — häufige Fehlermeldungen

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

### „Token-Budget erschöpft (max_tokens=…)."
Kann nur noch in der Analyse auftreten (die Sammlung läuft über die Agent-API, die kein Token-Limit kennt). `DATASET_ANALYSIS_MAX_TOKENS` (Zeile 207) erhöhen oder `ANALYSIS_BATCH_MAX_ANSWERS` senken.

### „Timeout nach 180s." / „Timeout nach 240s."
Das Modell hat zu lange gebraucht. Mögliche Ursachen: hohe Last, sehr aufwendige Websuche, sehr langer Prompt. Weniger parallele Aufrufe einstellen oder in Zeile 196/197 höher setzen.

### „Es konnten keine Fragen aus der Antwort extrahiert werden."
Das Modell hat auf die Aufforderung nach einem JSON-Array mit Prosa geantwortet. Ein anderes Modell probieren, oder in Schritt 2 auf eigene Fragen umstellen. Im Log steht unter `PARSED ZERO` der Anfang der tatsächlichen Antwort.

### „Ein Analyse-Batch konnte nicht als JSON gelesen werden (evtl. abgeschnitten)."
Die JSON-Ausgabe des Analyse-Modells wurde vermutlich abgeschnitten. `ANALYSIS_BATCH_MAX_ANSWERS` (Zeile 359) auf z. B. 25 senken und über den Button in Schritt 5 neu analysieren — das kostet keine neuen Sammel-Aufrufe.

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
Sollte durch den Schutz in Zeile 39 nicht passieren. Falls doch: Streamlit vollständig beenden und neu starten.

---

## Sicherheit und Datenschutz

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

Der Code weist an Zeile 1151 selbst darauf hin:
> *„…und es ENTHÄLT DEN PROMPT-TEXT — einen Blick wert, bevor diese Datei nach außen gegeben wird."*

### Was an Langdock übertragen wird

Jede Frage, jeder Prompt und jede Marke, die Sie eingeben, geht an Langdock und von dort an den jeweiligen KI-Anbieter. Bei aktiver Websuche werden zusätzlich Suchanfragen an eine Suchmaschine gestellt.

**Praktische Konsequenz:** Keine vertraulichen Produktnamen, unveröffentlichten Projektbezeichnungen oder personenbezogenen Daten in die Fragen schreiben. Die Region `eu` sorgt für EU-Verarbeitung beim Analyse-Endpoint — **die Agent-API, über die alle Fragen gestellt werden, hat allerdings keine Region in der Adresse** (Zeile 188). Wo genau sie verarbeitet, sollte bei Bedarf mit Langdock geklärt werden.

### Betrieb im Netzwerk

Streamlit bringt **keine Anmeldung** mit. Standardmäßig läuft die App nur lokal (`localhost`). Wird sie im Firmennetz oder im Internet bereitgestellt, kann jeder, der die Adresse kennt, sie benutzen — und dabei den API-Key eingeben, den er selbst mitbringt, oder die Ergebnisse anderer sehen, wenn sie sich eine Instanz teilen. Für eine solche Bereitstellung braucht es einen vorgeschalteten Zugriffsschutz.

---

## Stellschrauben — was man wo ändern kann

Alle wichtigen Werte stehen als Konstanten am Anfang der Datei. Zum Anpassen genügt es, die Zahl zu ändern und die App neu zu starten.

| Was | Zeile | Standard | Wirkung beim Ändern |
|---|---|---|---|
| `LANGDOCK_REGION` | 182 | `eu` | Umgebungsvariable beim Start, nicht im Code ändern |
| `REQUEST_TIMEOUT` | 196 | 180 s | Höher, wenn Analyse-Calls regelmäßig in den Timeout laufen |
| `AGENT_STREAM_TIMEOUT` | 197 | 240 s | Höher bei sehr aufwendigen Suchen |
| `MAX_TOKENS` | 198 | 8000 | Standardbudget für Passthrough-Calls |
| `QUESTION_MAX_TOKENS` | 199 | 16000 | Wird nur protokolliert — die Agent-API kennt kein Token-Limit |
| **`ANALYSIS_MODEL_FALLBACK`** | **206** | `claude-opus-4-8` | Nur wenn das neueste Opus nicht live ermittelt werden kann (`resolve_analysis_model`) |
| `DATASET_ANALYSIS_MAX_TOKENS` | 207 | 16000 | Ausgabebudget je Analyse-Batch |
| `ANALYSIS_ANSWER_CHARS` | 208 | 2000 | Höher = weniger Kürzung, mehr Kosten |
| `COLLECTION_TEMPERATURE` | 216 | 0.7 | Niedriger = einheitlichere Antworten (Wiederholungen verlieren an Wert) |
| `QUESTION_TEMPERATURE` | 217 | 0.8 | Niedriger = konventionellere Fragen |
| `ANALYSIS_TEMPERATURE` | 218 | 0.0 | Nicht erhöhen — Reproduzierbarkeit geht verloren |
| `_FATAL_LIMIT_RE` | 296 | siehe Code | Suchmuster für endgültige Limits. Erweitern nur mit Bedacht |
| `ANALYSIS_BATCH_MAX_ANSWERS` | 359 | 40 | Kleiner = mehr Aufrufe, geringeres Abschneide-Risiko |
| `ANALYSIS_BATCH_MAX_INPUT_TOKENS` | 360 | 40000 | Eingabegrenze je Batch |
| `ANALYSIS_ANSWER_HEAD_FRAC` | 364 | 0.7 | Verhältnis Anfang/Ende bei der Kürzung |
| `AGENT_MODELS_TTL` | 522 | 60 s | Wie lange die Modellliste zwischengespeichert wird |
| Agent-`instructions` | 1102 | siehe Code | Der Text, der das Modell zum Suchen bewegt |
| Prompts der Fragengenerierung | 1431 | siehe Code | Welche Art Fragen erzeugt wird |
| Prompts der Antwortsammlung | 1531 | siehe Code | **Nur mit Bedacht ändern** — beeinflusst das Messergebnis direkt |
| Analyse-Prompt | 1753 | siehe Code | Welche Felder erkannt werden |

---

## Weiterführende Dokumentation

Damit diese Datei schlank bleibt, liegen die ausführlichen Teile in eigenen Dokumenten. GitHub stellt sie genauso dar wie diese Seite — einfach anklicken.

| Dokument | Inhalt | Für wen |
|---|---|---|
| **[📖 Glossar](docs/GLOSSAR.md)** | Rund 35 Fachbegriffe alltagssprachlich erklärt — LLM, Token, Temperatur, Rate Limit, Streaming, Session State, Share of Voice … | Alle, die beim Lesen über einen Begriff stolpern |
| **[🔧 Code-Erklärung](docs/CODE.md)** | `app.py` Abschnitt für Abschnitt: was der Code tut, warum er so gebaut ist, welche Erfahrungen dahinterstecken. Mit Landkarte, Datenfluss-Diagramm und Anhang zu `langdock_evidence.py` | Wer das Tool anpassen, erweitern oder tiefer verstehen will |

**Direkt zu den wichtigsten Code-Abschnitten:**

- [Die zwei Modellkataloge](docs/CODE.md#212-die-zwei-modellkataloge--das-komplizierteste-thema-im-script) — die häufigste Fehlerquelle im Betrieb
- [`call_langdock()` — das Herzstück](docs/CODE.md#213-call_langdock--das-herzstück) — jede API-Anfrage läuft hier durch
- [`call_langdock_agent()` — der Websuche-Weg](docs/CODE.md#214-call_langdock_agent--der-websuche-weg)
- [Phase 1 — der Sammellauf](docs/CODE.md#228-phase-1--der-sammellauf)

---

## Projektstruktur

| Datei / Ordner | Inhalt | Im Git? |
|---|---|---|
| `app.py` | Die komplette Anwendung — Oberfläche, API-Anbindung, Analyse (~4.100 Zeilen) | ✅ |
| `langdock_evidence.py` | Protokolliert Agent-API-Anfragen für Langdock-Support-Tickets | ✅ |
| `requirements.txt` | Die vier benötigten Bibliotheken | ✅ |
| `docs/GLOSSAR.md` | Begriffserklärungen | ✅ |
| `docs/CODE.md` | Ausführliche Code-Erklärung | ✅ |
| `support/` | Standalone-Reproduktionsscript und Support-Korrespondenz | ✅ |
| `brand_visibility.log` | Technisches Protokoll jedes API-Aufrufs | ❌ |
| `support_evidence.jsonl` | Vollständige Mitschriften der Agent-API-Aufrufe | ❌ |
| `results/` | Lokal gespeicherte CSV-Exporte | ❌ |
| `key.txt` | Falls Sie den API-Key dort ablegen — **bewusst gesperrt** | ❌ |

> Die mit ❌ markierten Dateien entstehen erst beim Betrieb und sind in `.gitignore` ausgeschlossen. Sie enthalten Betriebsdaten und teils vollständige Prompt-Inhalte — siehe [Sicherheit und Datenschutz](#sicherheit-und-datenschutz).

---

## Schnellstart-Checkliste

Für den ersten Lauf:

1. `git clone …` und `pip install -r requirements.txt`
2. `streamlit run app.py`
3. API-Key eingeben → **„Verbindung testen"** drücken (kostet einen Aufruf, spart viel Zeit)
4. Klein anfangen: **5 Fragen × 1 Run × 1 Modell**
5. In Schritt 4 prüfen, ob die Websuche tatsächlich genutzt wurde
6. **Exportieren** — die Daten liegen nur im Arbeitsspeicher und sind beim Schließen des Tabs weg
7. Erst wenn die Kette funktioniert: hochskalieren

---

*Stand: 10.09.2026 · Bezieht sich auf `app.py` mit 4.280 Zeilen*
