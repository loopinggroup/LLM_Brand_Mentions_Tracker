# Antwort an den Langdock-Support

> Versandfertig. Anhänge: die beiden Dateien aus `support/evidence/` des Laufs
> `langdock_repro_20260819-125518` (Bericht `.md` + vollständiger Mitschnitt `.jsonl`).
> Beide enthalten keinen API-Key, nur eine Kennung — sie können unverändert weitergegeben
> werden.

---

Hallo,

danke für die Rückmeldung. Anbei ein zusammengehöriges Paar aus **einem** Durchlauf, wie
angefragt — GET, unmittelbar folgender POST, Statuscode und unveränderter Response-Body.

**Zum Ablauf:** Ich habe dafür ein eigenständiges Reproduktionsskript geschrieben, das
nichts anderes tut als (A) einmal `GET /agent/v1/models` abzurufen, (B) jede zurückgegebene
`data[].id` unverändert an `POST /agent/v1/chat/completions` zu schicken — je fünfmal, weil
der Fehler intermittierend ist — und (C) am Ende den Katalog erneut abzurufen. Alle 51
Requests laufen in einem Prozess, in derselben Session, gegen `api.langdock.com`, mit
demselben Key.

**Ergebnis des Laufs vom 19.08.2026, 12:55–12:56 Uhr (Run-ID `50b2edc3d123`):**
49 Completion-Requests, davon **31 × HTTP 200 und 18 × HTTP 400**.

Zu Euren beiden Punkten:

**1. „Die ID muss unverändert aus `data[].id` übernommen werden."**

Genau das passiert. Es gibt in meinem Code keinen Pfad, der den Wert anfasst — keine
Anzeigenamen, keine Kürzung, kein Hinzufügen oder Entfernen von `@default` oder
Versionsständen. Im Anhang steht zu jedem POST der exakt gesendete Body; die
`model`-Werte lassen sich Zeichen für Zeichen gegen die GET-Response in `seq 1` prüfen.

Dass die ID nicht die Ursache ist, zeigt der Lauf zweimal unabhängig voneinander:

- **Dieselbe ID liefert 200 *und* 400.** Bei 8 von 9 Anthropic-Modellen wechselt das
  Ergebnis bei identischem String, identischem Key und identischem Endpunkt.
  `claude-opus-4-7@default` etwa: `seq 2` → 400, `seq 3` → 200, `seq 4` → 200,
  `seq 5` → 400, `seq 6` → 200 — fünf identische Requests innerhalb von acht Sekunden.
  Eine grundsätzlich nicht akzeptierte Schreibweise käme auf 0 × 200.
- **Die ablehnende Antwort bietet dasselbe Modell selbst an — nur anders geschrieben.**
  In `seq 2` wird `claude-opus-4-7@default` abgelehnt, und dieselbe Antwort listet unter
  „available models are" `claude-opus-4-7`. In `seq 7` wird `eu.anthropic.claude-opus-4-8`
  abgelehnt und `claude-opus-4-8@default` angeboten. Insgesamt 18 solcher Fälle,
  tabellarisch in Abschnitt 1 des Berichts.

**2. „GET und POST müssen mit demselben API-Key und gegen denselben Host laufen."**

Ist erfüllt. Jeder Eintrag im Mitschnitt trägt dasselbe Feld
`key_fingerprint: sha256:69a42b2eb409/…G04w` — ein SHA-256-Präfix des Keys plus dessen
letzte vier Zeichen. Identischer Fingerprint = identischer Key, ohne dass ich den Key
selbst herausgeben muss. Die Hosts stehen als Konstanten im Skript und sind bei jedem
Request mitprotokolliert.

**Was der Lauf stattdessen zeigt**

Der Katalog ist nicht stabil, und zwar nicht über Tage, sondern innerhalb von Sekunden.
Die beiden GETs desselben Laufs liegen 61 Sekunden auseinander und geben beide 27 Modelle
zurück — aber **8 der 9 Anthropic-Familien unter jeweils anderer Schreibweise**:

| Modell | erster GET (`seq 1`, 12:55:18) | letzter GET (`seq 51`, 12:56:19) |
|---|---|---|
| Opus 4.7 | `claude-opus-4-7@default` | `eu.anthropic.claude-opus-4-7` |
| Opus 4.8 | `eu.anthropic.claude-opus-4-8` | `claude-opus-4-8@default` |
| Opus 5 | `claude-opus-5@default` | `eu.anthropic.claude-opus-5` |
| Haiku 4.5 | `claude-haiku-4-5@20251001` | `eu.anthropic.claude-haiku-4-5-20251001-v1:0` |
| Sonnet 4.5 | `eu.anthropic.claude-sonnet-4-5-20250929-v1:0` | `claude-sonnet-4-5@20250929` |
| Sonnet 4.6 | `claude-sonnet-4-6@default` | `eu.anthropic.claude-sonnet-4-6` |

Einzig `claude-sonnet-5` ist in beiden Abrufen gleich geschrieben. Die vollständige
Gegenüberstellung steht in Abschnitt 2 des Berichts.

Dasselbe Bild bei den 400-Bodies: `seq 11` und `seq 12` liegen **104 Millisekunden**
auseinander und zählen unterschiedliche Kataloge auf — vier Anthropic-Einträge sind
jeweils anders geschrieben (`claude-opus-4-8@default` vs. `claude-opus-4-8`,
`claude-opus-5@default` vs. `claude-opus-5`, `claude-opus-4-6-v1` vs.
`claude-opus-4-6@default`, `claude-sonnet-4-6` vs. `claude-sonnet-4-6@default`).

Betroffen sind ausschließlich Anthropic-Familien. Die Kontrollgruppe im selben Lauf —
`gpt-5-mini-eu` und `gpt-5.6-sol` — kommt auf 4 × 200 und 0 × 400. Über die vergangenen
Wochen deckt sich das mit meinen Produktionslogs: kein einziger dieser 400er betraf ein
Nicht-Anthropic-Modell.

Für mich sieht das danach aus, als würden die Agent-API-Requests von mehreren Backends
beantwortet, deren Anthropic-Deployments unter unterschiedlichen Schreibweisen registriert
sind (`bare` ↔ `@default`, `-<Datum>` ↔ `@<Datum>`, `-v1:0` ↔ `@default`, mit und ohne
`eu.anthropic.`-Präfix). Welches Backend antwortet, entscheidet dann darüber, ob die aus
dem Katalog übernommene ID akzeptiert wird. Ob das an einer Migration der Deployment-Namen
liegt oder an inkonsistenten Replikaten, kann ich von außen nicht sagen — dafür bräuchte
es Euren Blick auf die Instanzen.

**Anhänge**

1. `langdock_repro_<Zeitstempel>.md` — Lesefassung: Ergebnisübersicht, Katalogvergleich und
   anschließend alle 51 Request-/Response-Paare in gesendeter Reihenfolge mit Zeitstempeln.
2. `langdock_repro_<Zeitstempel>.jsonl` — derselbe Mitschnitt maschinenlesbar, ein Objekt
   pro Request, nichts gekürzt, inklusive aller Response-Header (`date`, `x-request-id`
   usw., falls das bei der Zuordnung zu einer konkreten Instanz hilft).

Wenn es hilft, kann ich zusätzlich ältere Produktionslogs schicken oder das Skript mit
mehr Wiederholungen bzw. zu einer von Euch gewünschten Uhrzeit erneut laufen lassen.

Viele Grüße
Ben Seegatz
