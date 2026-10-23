# Moral Prompting — Analysis Pipeline

Empirische Studie zur Frage ob individuelle Moral Foundations Profile (MFQ)
sich in der Sprache widerspiegeln, mit der Menschen KI-Chatbots zu politischen
Themen befragen.

## Projektstruktur

```
data/          Eingangsdaten (bereinigter SoSci-Export, deutsches MFD, Pilot-Prompts)
models/        Word2Vec-Modell (nicht im Repo, siehe Installation)
src/           Gemeinsamer Code: config, Einlesen, Textaufbereitung, Statistik, Embedding-Funktionen
pipeline/      Die Analyseschritte (siehe Ausführungsreihenfolge)
diagnostics/   Zusatzprüfungen: Deskriptives, Wörterbuch-Coverage, Centroid-Sanity-Check
results/       Alle erzeugten Tabellen (xlsx) und Plots
run_all.py     Führt alles in der richtigen Reihenfolge aus
```

| Datei | Beschreibung |
|-------|-------------|
| `pipeline/wordcount.py` | MFD-basiertes Word Counting: Treffer pro Person und Dimension → `results/Scoring_Ergebnisse.xlsx` |
| `pipeline/wordembedding.py` | Word2Vec: Cosine Similarity zwischen Prompt-Centroid und Dictionary-Centroids → `results/Embedding_Scores.xlsx` |
| `pipeline/analysis.py` | Profilkorrelation pro Person (Pearson r, gemittelt) für beide Methoden, plus Zusatzanalyse je einzelner Dimension (Holm-korrigiert) |
| `pipeline/wordembedding_pertopic.py` | Alternative Embedding-Pipeline: Prompts pro Thema einzeln einbetten, Korrelationen per Fisher-Z mitteln. Benötigt die unbereinigten Rohdaten (nicht im Repo) |
| `pipeline/wordcount_pertopic.py` | Wordcounting je Themenblock → `results/Scoring_PerTopic.xlsx` (Rohdaten nötig) |
| `pipeline/rsa_analysis.py` | RSA mit Mantel-Test: Ähnlichkeit im MFQ-Profil vs. Ähnlichkeit in der Sprache über alle Personenpaare, gesamt/je Dimension/je Thema, plus Kontrollen → `results/RSA_Ergebnisse.xlsx` |
| `pipeline/llm_judge_run.py` | LLM-as-a-Judge: Texte (Person, Thema oder Einzelprompt) mit einer Rubrik bewerten lassen, Rohantworten speichern, Lauf setzt nach Abbruch fort. Bewerter: `fake` (Test), `ollama` (lokal), `anthropic` (API, nur mit `--confirm-data-protection`) |
| `pipeline/llm_validation.py` | Instrumentvalidierung: Verlässlichkeit der Läufe (ICC), Zusammenhang mit Wordcounting/Embedding, Hypothesentest mit den Bewerter-Scores |
| `diagnostics/descriptive_analysis.py` | Stichprobenbeschreibung, MFQ-Verteilungen, Prompt-Länge, Treffer pro Dimension (absolut und pro 100 Wörter) |
| `diagnostics/embedding_coverage.py` | Welcher Anteil der Wörterbuch-Einträge wird vom Modell gefunden |
| `diagnostics/embedding_sanity_check.py` | Bootstrap-Test: sind die MFD-Dimensions-Centroide im Embedding-Raum unterscheidbar (vs. Zufalls-Baseline) |
| `src/config.py` | Alle Pfade, Dimensionen und die Zuordnung der Prompt-Felder zu Themen |
| `src/rsa_utils.py` | Distanzmatrizen, Mantel-Test (optional partiell) mit Permutationstest |
| `src/llm_judge.py` | Rubrik (Entwurf), JSON-Schema, Bewerter-Klassen mit Schutzschaltern |
| `src/stats_utils.py` | Holm-Bonferroni, Fisher-Z-Mittelung, Profilkorrelation mit NaN-Regel für varianzlose Profile |

## Benötigte Dateien (nicht im Repository)

| Datei | Beschreibung | Quelle |
|-------|-------------|--------|
| `models/german.model` | Vortrainiertes deutsches Word2Vec Modell | siehe Installation |
| Rohdaten (`Rohdaten.xlsx`) | Unbereinigter SoSci-Export, nur für `wordembedding_pertopic.py`. Pfad in `src/config.py` oder per Umgebungsvariable `MORAL_PROMPTING_ROHDATEN` | SoSci Survey Export |

## Installation

```bash
pip install -r requirements.txt
```

Deutsches Word2Vec Modell herunterladen (devmount):

```bash
curl -L "https://cloud.devmount.de/d2bc5672c523b086/german.model" -o models/german.model
```

## Ausführung

```bash
python run_all.py                     # Hauptpipeline (+ Per-Thema-Variante, falls Rohdaten vorhanden)
python run_all.py --with-diagnostics  # zusätzlich Deskriptives, Coverage, Sanity-Check
```

Oder einzeln, immer aus dem Projektordner:

```bash
python -m pipeline.wordcount
python -m pipeline.wordembedding
python -m pipeline.analysis
python -m pipeline.wordembedding_pertopic
python -m pipeline.wordcount_pertopic
python -m pipeline.rsa_analysis
```

Tests (einmalig `pip install -r requirements-dev.txt`):

```bash
python -m pytest tests
```

## Dimensionen

Die Analyse umfasst 5 Moral Foundations (MFQ30, Haidt & Joseph, 2004; siehe Literatur):

- **Care/Harm** — Fürsorge, Schutz, Schaden
- **Fairness/Cheating** — Gerechtigkeit, Gleichheit, Betrug
- **Loyalty/Betrayal** — Loyalität, Gruppenidentität, Verrat
- **Authority/Subversion** — Autorität, Hierarchie, Gehorsam
- **Purity/Degradation** — Reinheit, Heiligkeit, Entweihung

## Methoden

**Word Count:** Für jede VP wird gezählt wie oft Terme aus dem deutschen MFD in den Prompts vorkommen
(Präfix-Matching für Wildcard-Terme). Die Treffer werden durch die individuelle Wortzahl geteilt (Textlänge und
Trefferzahl korrelieren stark), anschließend z-normalisiert und per Pearson-Korrelation mit dem MFQ-Profil verglichen.
Haben alle fünf Dimensionen einer Person denselben Rohwert (i.d.R. null Treffer), ist die Korrelation nicht definiert
(NaN) und wird nicht mitgemittelt. Die individuellen r-Werte werden über alle VPs gemittelt.

**Word Embedding:** Für jede MFD-Dimension wird ein Centroid-Vektor aus dem Word2Vec-Modell berechnet. Der Prompt-Text
jeder VP wird ebenfalls als Centroid-Vektor repräsentiert. Die Cosine Similarity zwischen Text-Centroid und
Dictionary-Centroids ergibt einen 5-dimensionalen Score-Vektor, der z-normalisiert und per Pearson-Korrelation mit dem
MFQ-Profil verglichen wird. Die individuellen r-Werte werden über alle VPs gemittelt.

**Per-Thema-Variante (Embedding):** Statt alle Prompts einer Person zu einem Text zu aggregieren, wird pro Thema ein
eigener Centroid gebildet, einzeln korreliert und die drei Korrelationen per Fisher-Z gemittelt.

**Dimensionseffekte:** Zusätzlich wird je Dimension einzeln über alle Personen korreliert (between-subjects, N=161 statt
5 Datenpunkte pro Korrelation), mit Holm-Korrektur über alle Tests.

**RSA (Mantel-Test):** Statt je Person fünf Werte zu korrelieren, wird für alle Personenpaare der Abstand im MFQ-Profil
mit dem Abstand im Sprach-Profil verglichen (euklidisch, beide Seiten je Dimension z-transformiert). Der Mantel-r ist die
Korrelation dieser Abstände; die Signifikanz kommt aus einem Permutationstest (10.000 Permutationen, fester Seed), weil die
Paare nicht unabhängig sind. Hypothese gerichtet (r > 0). Holm-Korrektur über die 18 Basis-Tests. Kontrollen: Niveau
entfernt, partieller Mantel-Test (Textlänge, Alter, Geschlecht, KI-Nutzung), Ausschluss von Personen ohne Treffer.
Es gibt keine Einzelwerte pro Person. Bei N=161 hat der Test hohe Power erst ab Korrelationen von etwa .3 je Dimension
(Simulation: Power .35 bei .2, .86 bei .3).

**LLM-as-a-Judge (Instrumentvalidierung):** Ein Sprachmodell bewertet jeden Text nach einer festen Rubrik (fünf
Dimensionen, Skala 0 bis 5, JSON-Ausgabe). Jeder Text wird mehrfach bewertet (die aktuellen Modelle erlauben keine
feste Temperature), ausgewertet wird der Median, die Stabilität über die Läufe wird als ICC berichtet. Die Rubrik wird an
einer festen Entwicklungsteilmenge (30 Personen, `--subset dev`) verfeinert und dann eingefroren, die Validierung läuft
auf dem Rest (`--subset holdout`). Jede Rubrikänderung ändert den Fingerabdruck, Ergebnisse verschiedener Rubriken werden
nie vermischt. **Datenschutz:** Der API-Bewerter sendet Teilnehmertexte an einen externen Dienst und startet nur mit
`--confirm-data-protection`. Rohantworten (`results/llm_judge/`) enthalten Teilnehmertexte und sind bewusst nicht im Repo.
Die Rubrik ist ein Entwurf. Ohne menschliche Bewertung einer Teilstichprobe zeigt die Validierung nur, dass automatische
Methoden untereinander übereinstimmen.

## Hinweise

- Umlaute werden vor dem Modell-Lookup normalisiert (ä→ae, ö→oe, ü→ue, ß→ss), da das devmount-Modell mit dieser
  Schreibweise trainiert wurde. Die Wortliste liegt bereits in dieser Schreibweise vor
- Coverage des Word2Vec-Modells mit Präfix-Matching: 87–96 % je Dimension (`diagnostics/embedding_coverage.py`).
  Offen ist die Qualität der Wildcard-Treffer: sehr kurze Stämme (z. B. `gut*`, `tod*`) matchen im Modell hunderte
  teils unpassende Wörter
- Der bereinigte Datensatz (`data/Daten_bereinigt.xlsx`) enthält die Prompts bereits über alle Themen aggregiert
  (`prompts_all`). Themengetrennte Prompts gibt es nur in den Rohdaten
- `data/Prompts.xlsx` sind die Prompts der Pilotstudie und werden von keinem Skript mehr verwendet

## Literatur

Haidt, J., & Joseph, C. (2004). Intuitive ethics: How innately prepared 
intuitions generate culturally variable virtues. *Daedalus*, 133(4), 55–66. http://www.jstor.org/stable/20027945

Bos, L. and Minihold, S. (2022), The Ideological Predictors of Moral Appeals by European Political Elites; An Exploration of the Use of Moral Rhetoric in Multiparty Systems. Political Psychology, 43: 45-63. https://doi.org/10.1111/pops.12739
