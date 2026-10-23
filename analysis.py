"""Hauptanalyse: Profilkorrelation MFQ <-> Sprach-Score (Wordcounting und Embedding)
sowie Zusatzanalyse je einzelner Dimension."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

import config
from config import DIMS, MFQ_COLS
from data_utils import load_mfq
from stats_utils import holm_bonferroni, profile_correlation

DIM_LOWER = [d.lower() for d in DIMS]
RAW_COUNT_COLS = [f"score_{d}" for d in DIM_LOWER]
PERWORD_COLS = [c + "_perword" for c in RAW_COUNT_COLS]
EMB_RAW_COLS = [f"emb_{d}" for d in DIM_LOWER]


def add_zscores(df, cols):
    for col in cols:
        df[col + "_z"] = stats.zscore(df[col])
    return [c + "_z" for c in cols]


def add_profile_correlation(df, raw_cols, z_cols):
    df["pearson_r"] = df.apply(
        lambda row: profile_correlation(row[raw_cols].values, row[z_cols].values, row[MFQ_COLS].values),
        axis=1,
    )


def prepare_wordcounting(mfq):
    df = pd.read_excel(config.SCORING_PATH).merge(mfq, on="participant_id", how="inner")
    # Treffer relativ zur Wortzahl: Textlänge und moralische Sprache sonst konfundiert
    for raw_col, perword_col in zip(RAW_COUNT_COLS, PERWORD_COLS):
        df[perword_col] = df[raw_col] / df["n_words"]
    z_cols = add_zscores(df, PERWORD_COLS)
    add_profile_correlation(df, RAW_COUNT_COLS, z_cols)
    return df


def prepare_embedding(mfq):
    df = pd.read_excel(config.EMBEDDING_SCORES_PATH).merge(mfq, on="participant_id", how="inner")
    z_cols = add_zscores(df, EMB_RAW_COLS)
    add_profile_correlation(df, EMB_RAW_COLS, z_cols)
    return df


def plot_r_distribution(data_counting, data_embedding):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    box_data = [data_counting["pearson_r"].dropna(), data_embedding["pearson_r"].dropna()]
    ax.boxplot(box_data, showmeans=True, meanline=True,
               meanprops={"color": "red", "linestyle": "--"}, medianprops={"linewidth": 0})
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Wordcounting", "Word Embedding"])
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Pearson r pro Person")
    ax.set_title("Streuung der individuellen Korrelationen")
    for i, d in enumerate(box_data, start=1):
        ax.text(i, -0.08, f"Mean r = {d.mean():.3f}", ha="center", va="top",
                fontsize=10, transform=ax.get_xaxis_transform())
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    fig.savefig(config.PLOT_DIR / "07_pearson_r_verteilung_boxplot.png", dpi=150)
    plt.close(fig)


def dimension_effects(data_counting, data_embedding):
    """Zusatzanalyse: je Dimension EINZELN über alle Personen korrelieren (between-subjects,
    N=161 statt 5 Datenpunkte) - andere Frage als die Profilkorrelation innerhalb einer Person."""
    rows = []
    for dim, perword_col, emb_col, mfq_col in zip(DIMS, PERWORD_COLS, EMB_RAW_COLS, MFQ_COLS):
        r_wc, p_wc = stats.pearsonr(data_counting[mfq_col], data_counting[perword_col])
        r_emb, p_emb = stats.pearsonr(data_embedding[mfq_col], data_embedding[emb_col])
        rows.append({"dimension": dim, "r_wordcounting": r_wc, "p_wordcounting": p_wc,
                     "r_embedding": r_emb, "p_embedding": p_emb})
    result = pd.DataFrame(rows)

    # Holm-Korrektur über alle 10 Tests (5 Dimensionen x 2 Methoden) gemeinsam
    adjusted = holm_bonferroni(np.concatenate([result["p_wordcounting"], result["p_embedding"]]))
    result["p_wordcounting_holm"] = adjusted[:len(DIMS)]
    result["p_embedding_holm"] = adjusted[len(DIMS):]
    return result


def plot_dimension_effects(result):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(DIMS))
    width = 0.35
    ax.bar(x - width / 2, result["r_wordcounting"], width, label="Wordcounting")
    ax.bar(x + width / 2, result["r_embedding"], width, label="Embedding")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(DIMS)
    ax.set_ylabel("Pearson r (between-subjects, N=161)")
    ax.set_title("MFQ vs. Sprach-Score je Dimension einzeln betrachtet")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.PLOT_DIR / "12_dimension_effekte.png", dpi=150)
    plt.close(fig)


def main():
    config.PLOT_DIR.mkdir(exist_ok=True)
    mfq = load_mfq()

    data_counting = prepare_wordcounting(mfq)
    data_embedding = prepare_embedding(mfq)

    n_undefined = data_counting["pearson_r"].isna().sum()
    print(f"Wordcounting: {n_undefined}/{len(data_counting)} Personen mit varianzlosem Rohsignal "
          f"(Korrelation nicht definiert)")
    print(f"Mean Pearson r for counting (N={data_counting['pearson_r'].notna().sum()}): "
          f"{data_counting['pearson_r'].mean()}")
    print(f"Mean Pearson r for embedding: {data_embedding['pearson_r'].mean()}")

    data_counting.to_excel(config.WORDCOUNT_ANALYSIS_PATH, index=False)
    data_embedding.to_excel(config.EMBEDDING_ANALYSIS_PATH, index=False)
    plot_r_distribution(data_counting, data_embedding)

    effects = dimension_effects(data_counting, data_embedding)
    print("\nEffekte je einzelner Dimension (between-subjects, Holm-korrigiert über alle 10 Tests):")
    print(effects.round(4).to_string(index=False))
    effects.to_excel(config.DIMENSION_EFFECTS_PATH, index=False)
    plot_dimension_effects(effects)

    print(f"\nPlots gespeichert in: {config.PLOT_DIR}")


if __name__ == "__main__":
    main()
