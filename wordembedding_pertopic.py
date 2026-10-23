"""
Alternative Embedding-Pipeline: Prompts pro Thema statt aggregiert einbetten
==============================================================================

Testet den im Ausblick/Diskussion des Praktikumsberichts vorgeschlagenen Ansatz:
Statt die Prompts einer Person über alle 3 Themen zu einem Text zu aggregieren und
EINEN Centroid zu bilden (so wie wordembedding.py), wird hier pro Thema ein eigener
Centroid gebildet, einzeln mit dem MFQ-Profil korreliert und die drei Korrelationen
anschließend über Fisher-Z gemittelt.

Eigenständige Pipeline: verändert wordembedding.py/analysis.py nicht, beide Methoden
existieren nebeneinander. Benötigt die unbereinigten Rohdaten (config.ROHDATEN_PATH),
die nicht im Repo liegen.
"""

import matplotlib.pyplot as plt
import pandas as pd
from gensim.models import KeyedVectors
from scipy import stats

import config
from config import DIMS, MFQ_COLS, TOPICS
from data_utils import load_mfq, load_raw_topic_prompts
from dictionary_utils import load_dictionaries
from embedding_utils import build_vocab_index, calculate_dict_centroid, embedding_scores
from stats_utils import fisher_z_average, profile_correlation
from text_utils import build_topic_text


def plot_comparison(data, has_aggregated):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    box_data = [data["pearson_r_fisherz"].dropna()]
    labels = ["Per-Thema\n+ Fisher-Z"]
    if has_aggregated:
        box_data = [data["pearson_r_aggregiert"].dropna(), data["pearson_r_fisherz"].dropna()]
        labels = ["Aggregiert\n(bestehend)", "Per-Thema\n+ Fisher-Z"]
    ax.boxplot(box_data, showmeans=True, meanline=True,
               meanprops={"color": "red", "linestyle": "--"}, medianprops={"linewidth": 0})
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Pearson r pro Person")
    ax.set_title("Embedding: aggregiert vs. per-Thema + Fisher-Z-Mittelung")
    for i, d in enumerate(box_data, start=1):
        ax.text(i, -0.17, f"Mean r = {d.mean():.3f}", ha="center", va="top", fontsize=9,
                transform=ax.get_xaxis_transform())
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.28)
    fig.savefig(config.PLOT_DIR / "11_embedding_aggregiert_vs_pertopic.png", dpi=150)
    plt.close(fig)


def main():
    config.PLOT_DIR.mkdir(exist_ok=True)

    raw = load_raw_topic_prompts()
    print(f"N nach Abgleich mit bereinigter Stichprobe: {len(raw)}")
    for topic, cols in TOPICS.items():
        raw[f"text_{topic}"] = raw.apply(lambda row: build_topic_text(row, cols), axis=1)

    model = KeyedVectors.load_word2vec_format(str(config.MODEL_PATH), binary=True)
    vocab_index = build_vocab_index(model)
    dictionaries = load_dictionaries()
    dict_centroids = {dim: calculate_dict_centroid(dictionaries[dim], model, vocab_index) for dim in DIMS}

    data = raw[["participant_id"] + [f"text_{t}" for t in TOPICS]].merge(load_mfq(), on="participant_id")

    # Pro Thema Score-Vektor berechnen und getrennt pro Thema z-normalisieren
    # (entfernt dimensionsspezifischen Bias und themenspezifische Baseline-Unterschiede)
    raw_cols, z_cols = {}, {}
    for topic in TOPICS:
        print(f"Berechne Centroids für Thema '{topic}' ...")
        scores = pd.DataFrame(
            [embedding_scores(t, model, dict_centroids, vocab_index) for t in data[f"text_{topic}"]]
        )
        raw_cols[topic] = [f"emb_{topic}_{dim.lower()}" for dim in DIMS]
        z_cols[topic] = [c + "_z" for c in raw_cols[topic]]
        for dim, raw_col, z_col in zip(DIMS, raw_cols[topic], z_cols[topic]):
            data[raw_col] = scores[dim].values
            data[z_col] = stats.zscore(data[raw_col], nan_policy="omit")
        data[f"pearson_r_{topic}"] = data.apply(
            lambda row: profile_correlation(row[raw_cols[topic]].values, row[z_cols[topic]].values,
                                            row[MFQ_COLS].values),
            axis=1,
        )

    r_cols = [f"pearson_r_{t}" for t in TOPICS]
    data["pearson_r_fisherz"] = data[r_cols].apply(lambda row: fisher_z_average(row.values), axis=1)
    data["n_topics_valid"] = data[r_cols].notna().sum(axis=1)

    for col in r_cols:
        print(f"{col}: Mean r = {data[col].mean():.4f} (N={data[col].notna().sum()})")
    print(f"\nPer-Thema + Fisher-Z gemittelt: Mean r = {data['pearson_r_fisherz'].mean():.4f} "
          f"(N={data['pearson_r_fisherz'].notna().sum()})")

    # Vergleich mit der bestehenden aggregierten Pipeline (nur lesend, optional)
    has_aggregated = config.EMBEDDING_ANALYSIS_PATH.exists()
    if has_aggregated:
        aggregated = pd.read_excel(config.EMBEDDING_ANALYSIS_PATH)[["participant_id", "pearson_r"]]
        data = data.merge(aggregated.rename(columns={"pearson_r": "pearson_r_aggregiert"}),
                          on="participant_id", how="left")
        print(f"Zum Vergleich, bestehende aggregierte Pipeline: Mean r = {data['pearson_r_aggregiert'].mean():.4f}")
    else:
        print(f"{config.EMBEDDING_ANALYSIS_PATH.name} nicht gefunden, kein Vergleich möglich.")

    data.to_excel(config.PERTOPIC_ANALYSIS_PATH, index=False)
    plot_comparison(data, has_aggregated)
    print(f"\nErgebnisse gespeichert: {config.PERTOPIC_ANALYSIS_PATH.name}")


if __name__ == "__main__":
    main()
