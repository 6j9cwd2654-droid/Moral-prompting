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
| `diagnostics/descriptive_analysis.py` | Stichprobenbeschreibung, MFQ-Verteilungen, Prompt-Länge, Treffer pro Dimension (absolut und pro 100 Wörter) |
| `diagnostics/embedding_coverage.py` | Welcher Anteil der Wörterbuch-Einträge wird vom Modell gefunden |
| `diagnostics/embedding_sanity_check.py` | Bootstrap-Test: sind die MFD-Dimensions-Centroide im Embedding-Raum unterscheidbar (vs. Zufalls-Baseline) |
| `src/config.py` | Alle Pfade, Dimensionen und die Zuordnung der Prompt-Felder zu Themen |
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
