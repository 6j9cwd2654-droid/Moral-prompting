"""
Alternative Embedding-Pipeline: Prompts pro Thema statt aggregiert einbetten
==============================================================================

Testet den im Ausblick/Diskussion des Praktikumsberichts vorgeschlagenen Ansatz:
Statt die Prompts einer Person ueber alle 3 Themen hinweg zu einem Text zu
aggregieren und EINEN Centroid zu bilden (so wie es wordembedding.py tut),
wird hier pro Thema ein eigener Centroid gebildet, einzeln mit dem MFQ-Profil
korreliert, und die drei resultierenden Korrelationskoeffizienten anschliessend
ueber Fisher-Z gemittelt. Grund: Themenspezifische moralische Sprache koennte
sich im gemeinsamen Centroid gegenseitig ausloeschen.

Eigenstaendige Pipeline, veraendert nichts an wordembedding.py/analysis.py -
beide Methoden sollen nebeneinander existieren und vergleichbar sein.

Voraussetzung: Rohdaten.xlsx (enthaelt die noch nicht themenweise aggregierten
Prompt-Felder P101-P103/P201-P203/P301-P303), liegt im Masterarbeit-Ordner,
nicht im Repo (enthaelt unbereinigte Rohdaten).

Nutzung:
    python wordembedding_pertopic.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from gensim.models import KeyedVectors

from dictionary_utils import load_dictionaries, DIMS
from embedding_utils import build_vocab_index, calculate_dict_centroid, calculate_text_centroid

ROHDATEN_PATH = (
    "/Users/finn.recktenwald/Library/CloudStorage/OneDrive-adessoGroup/Dokumente/"
    "Privat/Master Kognitionswissenschaft/Praktikum und Masterarbeit/"
    "Arbeitsdokumente/Rohdaten.xlsx"
)
MODEL_PATH = "german.model"
PLOT_DIR = "plots"

# P1xx = Waffenlieferungen, P2xx = Gesellschaftsdienst, P3xx = aktive Sterbehilfe.
# Empirisch bestaetigt (Inhalt der Prompt-Texte): die Spaltenzuordnung ist fix,
# unabhaengig von der randomisierten Anzeigereihenfolge im Fragebogen.
TOPICS = {
    "waffenlieferungen": ["P101_01", "P101_02", "P101_03"],
    "gesellschaftsdienst": ["P201_01", "P201_02", "P201_03"],
    "sterbehilfe": ["P301_01", "P301_02", "P301_03"],
}


def build_topic_text(row, cols):
    parts = [str(row[c]).strip() for c in cols if pd.notna(row[c]) and str(row[c]).strip()]
    return " ".join(parts) if parts else None


def score_embedding_safe(text, model, dict_centroids, vocab_index):
    if text is None:
        return {dim: np.nan for dim in DIMS}
    text_centroid = calculate_text_centroid(text, model, vocab_index)
    if text_centroid is None:
        return {dim: np.nan for dim in DIMS}
    scores = {}
    for dim in DIMS:
        if dict_centroids[dim] is not None:
            scores[dim] = 1 - cosine(text_centroid, dict_centroids[dim])
        else:
            scores[dim] = np.nan
    return scores


def correlate_row(row, raw_cols, z_cols, mfq_cols):
    raw_vals = row[raw_cols].values.astype(float)
    if np.isnan(raw_vals).any():
        return np.nan
    if np.ptp(raw_vals) == 0:
        return np.nan
    mfq_vec = row[mfq_cols].values.astype(float)
    score_vec = row[z_cols].values.astype(float)
    r, _ = __import__("scipy.stats", fromlist=["pearsonr"]).pearsonr(mfq_vec, score_vec)
    return r


def fisher_z_average(r_values):
    r_values = [r for r in r_values if not np.isnan(r)]
    if not r_values:
        return np.nan
    z_values = np.arctanh(np.clip(r_values, -0.999999, 0.999999))
    return np.tanh(np.mean(z_values))


def main():
    from scipy import stats

    os.makedirs(PLOT_DIR, exist_ok=True)

    print("Lese Rohdaten (unaggregierte Prompts pro Thema) ...")
    raw = pd.read_excel(ROHDATEN_PATH, sheet_name="ag4_moral_prompting_01")
    raw["participant_id"] = pd.to_numeric(raw["CASE"], errors="coerce")
    raw = raw.dropna(subset=["participant_id"])
    raw["participant_id"] = raw["participant_id"].astype(int)

    mfq_cols = ["mfq_care_score", "mfq_fairness_score", "mfq_loyalty_score",
                "mfq_authority_score", "mfq_purity_score"]
    mfq = pd.read_excel("Daten_bereinigt.xlsx", sheet_name="Data")[["participant_id"] + mfq_cols]

    # Nur die bereits bereinigte Stichprobe (N=161) verwenden, nicht die rohen 167/168 Faelle
    raw = raw[raw["participant_id"].isin(mfq["participant_id"])].copy()
    print(f"N nach Abgleich mit bereinigter Stichprobe: {len(raw)}")

    for topic, cols in TOPICS.items():
        raw[f"text_{topic}"] = raw.apply(lambda row: build_topic_text(row, cols), axis=1)

    print("Lade Word2Vec-Modell ...")
    model = KeyedVectors.load_word2vec_format(MODEL_PATH, binary=True)
    vocab_index = build_vocab_index(model)
    dictionaries = load_dictionaries()
    dict_centroids = {dim: calculate_dict_centroid(dictionaries[dim], model, vocab_index) for dim in DIMS}

    data = raw[["participant_id"] + [f"text_{t}" for t in TOPICS]].merge(mfq, on="participant_id", how="inner")

    # Pro Thema: Score-Vektor berechnen, dann getrennt pro Thema z-normalisieren
    # (entfernt sowohl den dimensionsspezifischen Bias als auch moegliche
    # themenspezifische Baseline-Unterschiede in der Centroid-Aehnlichkeit)
    per_topic_raw_cols = {}
    per_topic_z_cols = {}
    for topic in TOPICS:
        print(f"Berechne Centroids fuer Thema '{topic}' ...")
        results = data[f"text_{topic}"].apply(lambda t: score_embedding_safe(t, model, dict_centroids, vocab_index))
        raw_cols = []
        for dim in DIMS:
            col = f"emb_{topic}_{dim.lower()}"
            data[col] = results.apply(lambda x: x[dim])
            raw_cols.append(col)
        z_cols = [c + "_z" for c in raw_cols]
        for raw_col, z_col in zip(raw_cols, z_cols):
            data[z_col] = stats.zscore(data[raw_col], nan_policy="omit")
        per_topic_raw_cols[topic] = raw_cols
        per_topic_z_cols[topic] = z_cols

    # Pearson r pro Person UND Thema
    for topic in TOPICS:
        data[f"pearson_r_{topic}"] = data.apply(
            lambda row: correlate_row(row, per_topic_raw_cols[topic], per_topic_z_cols[topic], mfq_cols), axis=1
        )

    # Fisher-Z-Mittelung ueber die vorhandenen Themen pro Person
    r_cols = [f"pearson_r_{t}" for t in TOPICS]
    data["pearson_r_fisherz"] = data[r_cols].apply(lambda row: fisher_z_average(row.values), axis=1)
    data["n_topics_valid"] = data[r_cols].notna().sum(axis=1)

    for topic in TOPICS:
        col = f"pearson_r_{topic}"
        print(f"Thema '{topic}': Mean r = {data[col].mean():.4f} (N={data[col].notna().sum()})")

    mean_r_fisherz = data["pearson_r_fisherz"].mean()
    n_valid = data["pearson_r_fisherz"].notna().sum()
    print(f"\nPer-Thema + Fisher-Z gemittelt: Mean r = {mean_r_fisherz:.4f} (N={n_valid})")

    # Vergleich mit der bestehenden, aggregierten Pipeline (unveraendert, nur zum Einlesen)
    try:
        aggregated = pd.read_excel("Embedding_Analysis.xlsx")[["participant_id", "pearson_r"]]
        aggregated = aggregated.rename(columns={"pearson_r": "pearson_r_aggregiert"})
        data = data.merge(aggregated, on="participant_id", how="left")
        mean_r_aggregated = data["pearson_r_aggregiert"].mean()
        print(f"Zum Vergleich, bestehende aggregierte Pipeline: Mean r = {mean_r_aggregated:.4f}")
    except FileNotFoundError:
        mean_r_aggregated = None
        print("Embedding_Analysis.xlsx nicht gefunden, kein Vergleich moeglich.")

    data.to_excel("Embedding_PerTopic_Analysis.xlsx", index=False)

    # Plot: Vergleich der Verteilungen aggregiert vs. per-Thema+Fisher-Z
    fig, ax = plt.subplots(figsize=(6, 4.5))
    box_data = [data["pearson_r_fisherz"].dropna()]
    labels = ["Per-Thema\n+ Fisher-Z"]
    if mean_r_aggregated is not None:
        box_data = [data["pearson_r_aggregiert"].dropna(), data["pearson_r_fisherz"].dropna()]
        labels = ["Aggregiert\n(bestehend)", "Per-Thema\n+ Fisher-Z"]
    ax.boxplot(box_data, showmeans=True, meanline=True,
               meanprops={"color": "red", "linestyle": "--"}, medianprops={"linewidth": 0})
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Pearson r pro Person")
    ax.set_title("Embedding: aggregiert vs. per-Thema + Fisher-Z-Mittelung")
    means = [d.mean() for d in box_data]
    for i, m in enumerate(means, start=1):
        ax.text(i, -0.17, f"Mean r = {m:.3f}", ha="center", va="top", fontsize=9,
                 transform=ax.get_xaxis_transform())
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.28)
    fig.savefig(os.path.join(PLOT_DIR, "11_embedding_aggregiert_vs_pertopic.png"), dpi=150)
    plt.close(fig)

    print(f"\nErgebnisse gespeichert: Embedding_PerTopic_Analysis.xlsx")
    print(f"Plot gespeichert: {PLOT_DIR}/11_embedding_aggregiert_vs_pertopic.png")


if __name__ == "__main__":
    main()
