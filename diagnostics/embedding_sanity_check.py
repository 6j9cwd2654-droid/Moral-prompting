"""
Sanity-Check: Trennschärfe der MFD-Dimensions-Centroide im Embedding-Raum
==========================================================================

Beantwortet zwei Fragen, bevor die Embedding-Scores der Prompts inhaltlich
interpretiert werden:

1. Sind die fünf MFD-Dimensions-Centroide (Care, Fairness, Loyalty,
   Authority, Purity) im Word2Vec-Raum überhaupt unterscheidbar, oder
   kollabieren sie auf denselben Punkt? (Bootstrap-Test)
2. Ist der gefundene Unterschied praktisch bedeutsam, oder liegt er im
   selben Bereich wie die Ähnlichkeit zwischen komplett zufälligen,
   inhaltlich unverbundenen Wortgruppen? (Zufalls-Baseline-Vergleich)

Voraussetzungen: gensim, numpy, pandas, scipy, matplotlib
Erwartete lokale Module: dictionary_utils.py, embedding_utils.py
(bereits vorhandene Pipeline-Bausteine, siehe wordcount.py / wordembedding.py)

Nutzung:
    python -m diagnostics.embedding_sanity_check
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from gensim.models import KeyedVectors

from src import config
from src.config import DIMS
from src.dictionary_utils import load_dictionaries
from src.embedding_utils import build_vocab_index, calculate_dict_centroid

# ------------------------------------------------------------------
# Konfiguration
# ------------------------------------------------------------------
MODEL_PATH = config.MODEL_PATH
PLOT_DIR = config.PLOT_DIR
N_BOOTSTRAP = 1000
N_RANDOM_GROUPS = 5          # Anzahl zufälliger Wortgruppen pro Iteration
VOCAB_LIMIT = 50000          # nur die häufigsten N Wörter für Zufallsgruppen zulassen
SEED = 42


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ------------------------------------------------------------------
# 1. Centroid-Ähnlichkeitsmatrix (Punktschätzung, ohne Bootstrap)
# ------------------------------------------------------------------
def centroid_similarity_matrix(dict_centroids, dims=DIMS):
    """5x5-Matrix der paarweisen Cosine Similarity zwischen den
    MFD-Dimensions-Centroiden selbst (nicht Text-zu-Centroid!)."""
    n = len(dims)
    matrix = np.zeros((n, n))
    for i, d1 in enumerate(dims):
        for j, d2 in enumerate(dims):
            matrix[i, j] = 1 - cosine(dict_centroids[d1], dict_centroids[d2])
    return pd.DataFrame(matrix, index=dims, columns=dims)


def plot_centroid_similarity(sim_matrix, out_path):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    # vmin auf 0.90 gesetzt statt 0, damit die real vorhandenen
    # Unterschiede im engen oberen Wertebereich sichtbar werden
    im = ax.imshow(sim_matrix.values, vmin=0.90, vmax=1.0, cmap="RdBu_r")
    labels = sim_matrix.columns.tolist()
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{sim_matrix.values[i, j]:.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Cosine similarity")
    ax.set_title("MFD-Dimensions-Centroide: paarweise Ähnlichkeit")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------
# 2. Bootstrap: sind die MFD-Dimensionspaare signifikant distinct?
# ------------------------------------------------------------------
def bootstrap_centroid_separation(dictionaries, model, vocab_index, dims=DIMS,
                                    n_bootstrap=N_BOOTSTRAP, seed=SEED):
    """
    Resampled die Wortlisten pro Dimension (mit Zurücklegen), berechnet
    die Centroide neu und sammelt die paarweisen Similarities über
    n_bootstrap Wiederholungen. Ergebnis: 95%-CI pro Dimensionspaar.
    Schließt das CI die 1.0 (=identisch) aus, ist der Unterschied
    zwischen den beiden Centroiden statistisch robust.
    """
    rng = np.random.default_rng(seed)
    pair_sims = {(d1, d2): [] for i, d1 in enumerate(dims) for d2 in dims[i + 1:]}

    for _ in range(n_bootstrap):
        boot_centroids = {}
        for dim in dims:
            words = dictionaries[dim]
            sample = rng.choice(words, size=len(words), replace=True).tolist()
            boot_centroids[dim] = calculate_dict_centroid(sample, model, vocab_index)

        for (d1, d2) in pair_sims:
            if boot_centroids[d1] is not None and boot_centroids[d2] is not None:
                sim = 1 - cosine(boot_centroids[d1], boot_centroids[d2])
                pair_sims[(d1, d2)].append(sim)

    rows = []
    all_sims = []
    for (d1, d2), sims in pair_sims.items():
        sims = np.array(sims)
        all_sims.append(sims)
        ci_low, ci_high = np.percentile(sims, [2.5, 97.5])
        rows.append({
            "dim_1": d1, "dim_2": d2,
            "mean_similarity": sims.mean(),
            "ci_low": ci_low, "ci_high": ci_high,
            "significantly_distinct": ci_high < 0.999,
        })
    result_df = pd.DataFrame(rows)
    return result_df, np.concatenate(all_sims)


# ------------------------------------------------------------------
# 3. Zufalls-Baseline: wie ähnlich sind inhaltlich unverbundene Wortgruppen?
# ------------------------------------------------------------------
def random_word_sample(model, size, rng, exclude_words, vocab_limit=VOCAB_LIMIT):
    """Zieht eine zufällige Wortgruppe aus den häufigsten vocab_limit
    Wörtern des Modells, ohne Überschneidung mit den MFD-Wortlisten."""
    vocab = list(model.key_to_index.keys())[:vocab_limit]
    vocab = [w for w in vocab if w.lower() not in exclude_words]
    sample = rng.choice(vocab, size=size, replace=False)
    return sample.tolist()


def random_baseline_similarities(model, vocab_index, dictionaries, dims=DIMS,
                                   n_groups=N_RANDOM_GROUPS,
                                   n_bootstrap=N_BOOTSTRAP, seed=SEED):
    """
    Baseline: paarweise Similarities zwischen n_groups zufälligen,
    inhaltlich unverbundenen Wortgruppen (Größe = Durchschnitt der
    MFD-Listengrößen), wiederholt per Bootstrap wie beim echten Test.
    """
    rng = np.random.default_rng(seed + 1)  # anderer Seed als Haupt-Bootstrap
    all_mfd_words = set(
        w.lower().rstrip("*") for dim in dims for w in dictionaries[dim]
    )
    avg_size = int(np.mean([len(dictionaries[d]) for d in dims]))

    pair_sims = []
    for _ in range(n_bootstrap):
        groups = [
            random_word_sample(model, avg_size, rng, exclude_words=all_mfd_words)
            for _ in range(n_groups)
        ]
        centroids = [calculate_dict_centroid(g, model, vocab_index) for g in groups]
        centroids = [c for c in centroids if c is not None]

        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                sim = 1 - cosine(centroids[i], centroids[j])
                pair_sims.append(sim)

    pair_sims = np.array(pair_sims)
    return {
        "mean": pair_sims.mean(),
        "ci_low": np.percentile(pair_sims, 2.5),
        "ci_high": np.percentile(pair_sims, 97.5),
        "raw": pair_sims,
    }


# ------------------------------------------------------------------
# 4. Vergleichsplot: MFD-Dimensionspaare vs. Zufalls-Baseline
# ------------------------------------------------------------------
def plot_mfd_vs_baseline(mfd_sims, baseline_sims, out_path):
    """Überlagerte Histogramme: Bootstrap-Verteilung der MFD-Dimensionspaare
    gegen die Bootstrap-Verteilung zufälliger Wortgruppen-Paare."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(
        min(mfd_sims.min(), baseline_sims.min()),
        max(mfd_sims.max(), baseline_sims.max()),
        40,
    )
    ax.hist(baseline_sims, bins=bins, alpha=0.5, label="Zufällige Wortgruppen (Baseline)",
            color="#888780", density=True)
    ax.hist(mfd_sims, bins=bins, alpha=0.6, label="MFD-Dimensionspaare",
            color="#0F6E56", density=True)
    ax.set_xlabel("Cosine similarity zwischen Centroiden")
    ax.set_ylabel("Dichte")
    ax.set_title("MFD-Dimensionen vs. Zufalls-Baseline: Centroid-Ähnlichkeit")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    import os
    os.makedirs(PLOT_DIR, exist_ok=True)

    section("Modell und Wortlisten laden")
    model = KeyedVectors.load_word2vec_format(str(MODEL_PATH), binary=True)
    vocab_index = build_vocab_index(model)
    dictionaries = load_dictionaries()

    dict_centroids = {
        dim: calculate_dict_centroid(dictionaries[dim], model, vocab_index)
        for dim in DIMS
    }

    section("1. Centroid-Ähnlichkeitsmatrix (Punktschätzung)")
    sim_matrix = centroid_similarity_matrix(dict_centroids)
    print(sim_matrix.round(3))
    plot_centroid_similarity(sim_matrix, os.path.join(PLOT_DIR, "07_centroid_similarity_heatmap.png"))

    section("2. Bootstrap: Trennschärfe der MFD-Dimensionspaare")
    separation_df, mfd_all_sims = bootstrap_centroid_separation(dictionaries, model, vocab_index)
    print(separation_df.round(3))
    separation_df.to_excel(config.BOOTSTRAP_PATH, index=False)

    section("3. Zufalls-Baseline: inhaltlich unverbundene Wortgruppen")
    baseline = random_baseline_similarities(model, vocab_index, dictionaries)
    print(f"Baseline Mean = {baseline['mean']:.3f}, "
          f"95%-CI = [{baseline['ci_low']:.3f}, {baseline['ci_high']:.3f}]")
    print(f"MFD-Dimensionspaare Mean = {mfd_all_sims.mean():.3f}, "
          f"Range = [{mfd_all_sims.min():.3f}, {mfd_all_sims.max():.3f}]")

    section("4. Vergleichsplot")
    plot_mfd_vs_baseline(mfd_all_sims, baseline["raw"],
                          os.path.join(PLOT_DIR, "08_mfd_vs_random_baseline.png"))

    print(f"\nPlots gespeichert in: {PLOT_DIR}")


if __name__ == "__main__":
    main()