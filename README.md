# Moral Prompting — Analysis Pipeline

Empirische Studie zur Frage ob individuelle Moral Foundations Profile (MFQ) 
sich in der Sprache widerspiegeln, mit der Menschen KI-Chatbots zu politischen 
Themen befragen.

## Projektstruktur

| Datei | Beschreibung |
|-------|-------------|
| `Word_count.py` | MFD-basiertes Word Counting: zählt Häufigkeiten von Dictionary-Termen pro VP und Dimension |
| `wordembedding.py` | Word2Vec-basiertes Embedding: berechnet Cosine Similarity zwischen Prompt-Centroids und Dictionary-Centroids |
| `analysis.py` | Statistische Analyse: Pearson-Korrelation pro VP und Mittelung der r-Werte für beide Methoden |
| `mfd_de_referenz.xlsx` | Deutsches Moral Foundations Dictionary (Bos & Minihold 2022) |
| `Prompts.xlsx` | Prompt-Daten der Pilot Teilnehmenden (VP, Aggregiert) | später SoSci Survey Export |


## Benötigte Dateien (nicht im Repository)

| Datei | Beschreibung | Quelle |
|-------|-------------|--------|
| `german.model` | Vortrainiertes deutsches Word2Vec Modell | [devmount.github.io/GermanWordEmbeddings](https://devmount.github.io/GermanWordEmbeddings/) |
| `MFQ_Ergebnisse.xlsx` | MFQ-Scores der Teilnehmenden pro Dimension | SoSci Survey Export |

## Installation

```bash
pip install pandas openpyxl numpy gensim scipy
```

## Ausführungsreihenfolge

```bash
python Word_count.py          # → Scoring_Ergebnisse.xlsx
python wordembedding.py       # → Embedding_Scores.xlsx
python wordcount_analysis.py  # → Wordcounting_Analysis.xlsx, Embedding_Analysis.xlsx
```

## Dimensionen

Die Analyse umfasst 5 Moral Foundations (MFQ, Haidt & Joseph):

- **Care/Harm** — Fürsorge, Schutz, Schaden
- **Fairness/Cheating** — Gerechtigkeit, Gleichheit, Betrug
- **Loyalty/Betrayal** — Loyalität, Gruppenidentität, Verrat
- **Authority/Subversion** — Autorität, Hierarchie, Gehorsam
- **Purity/Degradation** — Reinheit, Heiligkeit, Entweihung

## Methoden

**Word Count:** Für jede VP wird gezählt wie oft Terme aus dem deutschen MFD in den Prompts vorkommen (Präfix-Matching für Wildcard-Terme). 
Die rohen Häufigkeiten werden z-normalisiert und anschließend per Pearson-Korrelation mit den MFQ-Profilen verglichen. Die individuellen
r-Werte werden über alle VPs gemittelt um eine Gesamtaussage über den Zusammenhang zu erhalten.

**Word Embedding:** Für jede MFD-Dimension wird ein Centroid-Vektor aus dem Word2Vec-Modell berechnet. Der Prompt-Text jeder VP 
wird ebenfalls als Centroid-Vektor repräsentiert. Die Cosine Similarity zwischen Text-Centroid und Dictionary-Centroids ergibt einen 
5-dimensionalen Score-Vektor der direktper Pearson-Korrelation mit dem MFQ-Profil verglichen wird. Die individuellen r-Werte werden über 
alle VPs gemittelt.

## Hinweise

- Umlaute werden vor dem Modell-Lookup normalisiert (ä→ae, ö→oe, ü→ue, ß→ss), da das devmount-Modell mit dieser Schreibweise trainiert wurde
- Coverage des Word2Vec-Modells liegt bei 50-72% pro Dimension aufgrund von Wildcard-Stämmen — Überlegen das Modell zu wechseln
- `MFQ_Ergebnisse.xlsx` ist noch ein Platzhalter bis die SoSci-Daten vorliegen
