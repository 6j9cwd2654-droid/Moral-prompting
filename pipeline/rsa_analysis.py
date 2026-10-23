"""
RSA (Representational Similarity Analysis) mit Mantel-Test
===========================================================

Frage: Sind sich Personen mit ähnlichem MFQ-Profil auch in ihrer Prompt-Sprache ähnlich?
Statt je Person fünf Werte zu korrelieren (df=3) werden alle Personenpaare verglichen
(161 Personen -> 12.880 Paare). Das beantwortet eine andere Frage als die Profilkorrelation:
Das Gesamtniveau einer Person bleibt erhalten, dafür gibt es keine Einzelwerte pro Person.

Eigenständige, parallele Pipeline; verändert die bestehenden Analysen nicht.

Varianten
---------
Basis (18 Tests, Holm-Korrektur über diese 18; Hypothese gerichtet: Mantel-r > 0):
    gesamt      5 Dimensionen, aggregierte Prompts              (Wordcounting, Embedding)
    dimension   je Dimension einzeln, aggregierte Prompts       (Wordcounting, Embedding)
    thema       5 Dimensionen, Prompts je Thema einzeln         (Wordcounting, Embedding)
Kontrollen/Diagnosen (nur beschreibend, nicht korrigiert):
    Niveau entfernt          nur das Muster über die Dimensionen, ohne Gesamtniveau
    Kontrolle Textlänge      partieller Mantel-Test, Distanz in log(Wortzahl) herausgerechnet
    Kontrolle Kovariaten     zusätzlich Alter, Geschlecht, KI-Nutzung
    ohne Nullprofile         Personen ohne einen einzigen Treffer ausgeschlossen (nur Wordcounting)

Beide Seiten (MFQ und Sprach-Scores) werden je Dimension z-transformiert, damit alle
Dimensionen gleich stark in den Abstand eingehen. Sprach-Score Wordcounting: Treffer pro Wort.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.config import DIMS, MFQ_COLS, N_PERMUTATIONS, RANDOM_SEED, TOPICS
from src.data_utils import load_clean_data
from src.rsa_utils import (abs_difference_matrix, center_rows, euclidean_distance_matrix,
                           mantel_test, mismatch_matrix, zscore_columns)
from src.stats_utils import holm_bonferroni

DIM_LOWER = [d.lower() for d in DIMS]
METHODS = {"wordcounting": "Wordcounting", "embedding": "Embedding"}


def load_inputs():
    df = load_clean_data(["participant_id"] + MFQ_COLS + ["age", "gender_code", "ai_usage_freq_code"])
    n = len(df)

    scoring = pd.read_excel(config.SCORING_PATH)
    df = df.merge(scoring[["participant_id", "n_words"] + [f"score_{d}" for d in DIM_LOWER]], on="participant_id")
    emb = pd.read_excel(config.EMBEDDING_SCORES_PATH)
    df = df.merge(emb[["participant_id"] + [f"emb_{d}" for d in DIM_LOWER]], on="participant_id")

    have_topics = config.PERTOPIC_ANALYSIS_PATH.exists() and config.SCORING_PERTOPIC_PATH.exists()
    if have_topics:
        pertopic_emb = pd.read_excel(config.PERTOPIC_ANALYSIS_PATH)
        emb_cols = [f"emb_{t}_{d}" for t in TOPICS for d in DIM_LOWER]
        df = df.merge(pertopic_emb[["participant_id"] + emb_cols], on="participant_id")
        pertopic_wc = pd.read_excel(config.SCORING_PERTOPIC_PATH)
        wc_cols = [c for c in pertopic_wc.columns if c != "participant_id"]
        df = df.merge(pertopic_wc[["participant_id"] + wc_cols], on="participant_id")
    else:
        print("Themengetrennte Ergebnisse nicht gefunden, Themen-Varianten werden übersprungen.")

    assert len(df) == n, "Zusammenführung hat Personen verloren oder verdoppelt"
    return df, have_topics


def language_inputs(df, have_topics):
    """(Methode, Thema) -> (rohe Scores n x 5, Maske der Personen ohne einen einzigen Treffer)."""
    inputs = {}
    counts = df[[f"score_{d}" for d in DIM_LOWER]].values.astype(float)
    inputs[("wordcounting", None)] = (counts / df["n_words"].values[:, None], counts.sum(axis=1) == 0)
    inputs[("embedding", None)] = (df[[f"emb_{d}" for d in DIM_LOWER]].values.astype(float),
                                    np.zeros(len(df), dtype=bool))
    if have_topics:
        for topic in TOPICS:
            counts = df[[f"{topic}_score_{d}" for d in DIM_LOWER]].values.astype(float)
            inputs[("wordcounting", topic)] = (counts / df[f"{topic}_n_words"].values[:, None],
                                                counts.sum(axis=1) == 0)
            inputs[("embedding", topic)] = (df[[f"emb_{topic}_{d}" for d in DIM_LOWER]].values.astype(float),
                                             np.zeros(len(df), dtype=bool))
    return inputs


def covariate_matrices(df):
    log_words = abs_difference_matrix(np.log(df["n_words"].values))
    others = [abs_difference_matrix(df["age"].values),
              mismatch_matrix(df["gender_code"].values),
              abs_difference_matrix(df["ai_usage_freq_code"].values)]
    return log_words, others


def run_rsa(mfq_raw, lang_raw, dims=None, center=False, mask=None, covariates=None, return_all=False):
    keep = np.ones(len(mfq_raw), dtype=bool) if mask is None else mask
    cols = list(range(mfq_raw.shape[1])) if dims is None else dims
    mfq = zscore_columns(mfq_raw[keep][:, cols])
    lang = zscore_columns(lang_raw[keep][:, cols])
    if center:
        mfq, lang = center_rows(mfq), center_rows(lang)
    d_mfq, d_lang = euclidean_distance_matrix(mfq), euclidean_distance_matrix(lang)
    cov = None if covariates is None else [c[np.ix_(keep, keep)] for c in covariates]
    result = mantel_test(d_mfq, d_lang, covariates=cov, n_perm=N_PERMUTATIONS,
                         seed=RANDOM_SEED, return_null=return_all)
    result["n"] = int(keep.sum())
    if return_all:
        result["d_mfq"], result["d_lang"] = d_mfq, d_lang
    return result


def build_results(df, have_topics):
    mfq_raw = df[MFQ_COLS].values.astype(float)
    langs = language_inputs(df, have_topics)
    log_words, other_covs = covariate_matrices(df)

    rows, details = [], {}

    def add(variante, method, topic, dim, control, base, mask=None, **kwargs):
        res = run_rsa(mfq_raw, langs[(method, topic)][0], mask=mask,
                      return_all=(variante == "gesamt" and control == "keine"), **kwargs)
        n_zero = int(langs[(method, topic)][1][mask if mask is not None else slice(None)].sum())
        rows.append({"variante": variante, "methode": METHODS[method], "thema": topic or "alle",
                     "dimension": dim or "alle 5", "kontrolle": control, "basistest": base,
                     "n": res["n"], "n_paare": res["n_pairs"], "n_nullprofile": n_zero,
                     "mantel_r": res["r"], "p_einseitig": res["p_one_sided"],
                     "p_zweiseitig": res["p_two_sided"]})
        if "d_mfq" in res:
            details[method] = res

    for method in METHODS:
        add("gesamt", method, None, None, "keine", True)
    for method in METHODS:
        for d, name in enumerate(DIMS):
            add("dimension", method, None, name, "keine", True, dims=[d])
    if have_topics:
        for topic in TOPICS:
            for method in METHODS:
                add("thema", method, topic, None, "keine", True)

    for method in METHODS:
        add("gesamt", method, None, None, "Niveau entfernt", False, center=True)
        add("gesamt", method, None, None, "Kontrolle Textlänge", False, covariates=[log_words])
        add("gesamt", method, None, None, "Kontrolle Textlänge, Alter, Geschlecht, KI-Nutzung", False,
            covariates=[log_words] + other_covs)
    add("gesamt", "wordcounting", None, None, "ohne Nullprofile", False,
        mask=~langs[("wordcounting", None)][1])
    if have_topics:
        for topic in TOPICS:
            add("thema", "wordcounting", topic, None, "ohne Nullprofile", False,
                mask=~langs[("wordcounting", topic)][1])

    results = pd.DataFrame(rows)
    base = results["basistest"]
    results.loc[base, "p_holm_einseitig"] = holm_bonferroni(results.loc[base, "p_einseitig"].values)
    return results, details


def plot_gesamt(details):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for col, (method, label) in enumerate(METHODS.items()):
        res = details[method]
        i, j = np.triu_indices(res["d_mfq"].shape[0], k=1)
        ax = axes[0, col]
        ax.hexbin(res["d_mfq"][i, j], res["d_lang"][i, j], gridsize=35, cmap="Blues", mincnt=1)
        ax.set_xlabel("Abstand im MFQ-Profil (z-Werte)")
        ax.set_ylabel(f"Abstand im Sprach-Profil ({label})")
        ax.set_title(f"{label}: jedes Personenpaar ein Punkt")

        ax = axes[1, col]
        ax.hist(res["null"], bins=50, color="#bbbbbb", edgecolor="white")
        ax.axvline(res["r"], color="red", linewidth=2)
        ax.set_xlabel("Mantel-r")
        ax.set_ylabel("Anzahl Permutationen")
        ax.set_title(f"Beobachtet r = {res['r']:.3f} (einseitig p = {res['p_one_sided']:.3f})")
    fig.suptitle("RSA gesamt: Passen Ähnlichkeit im Fragebogen und Ähnlichkeit in der Sprache zusammen?")
    fig.tight_layout()
    fig.savefig(config.PLOT_DIR / "13_rsa_gesamt.png", dpi=150)
    plt.close(fig)


def plot_varianten(results):
    base = results[results["basistest"]].copy()
    base["label"] = base.apply(
        lambda r: {"gesamt": "Gesamt (5 Dim.)", "dimension": r["dimension"], "thema": r["thema"].capitalize()}[r["variante"]],
        axis=1,
    )
    labels = list(dict.fromkeys(base["label"]))
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(labels) + 1.5))
    for offset, (method, color) in zip((-0.18, 0.18), (("Wordcounting", "#1f77b4"), ("Embedding", "#ff7f0e"))):
        sub = base[base["methode"] == method].set_index("label").reindex(labels)
        ax.scatter(sub["mantel_r"], y + offset, color=color, label=method, zorder=3)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Mantel-r")
    ax.set_title("RSA: Mantel-r aller Basis-Varianten")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.PLOT_DIR / "14_rsa_varianten.png", dpi=150)
    plt.close(fig)


def main():
    config.PLOT_DIR.mkdir(exist_ok=True)
    df, have_topics = load_inputs()
    print(f"N = {len(df)} Personen, {len(df) * (len(df) - 1) // 2} Paare, "
          f"{N_PERMUTATIONS} Permutationen, Seed {RANDOM_SEED}")

    results, details = build_results(df, have_topics)
    results.to_excel(config.RSA_PATH, index=False)
    plot_gesamt(details)
    plot_varianten(results)

    show = ["variante", "methode", "thema", "dimension", "kontrolle", "n", "n_nullprofile",
            "mantel_r", "p_einseitig", "p_holm_einseitig"]
    pd.set_option("display.width", 250, "display.max_colwidth", 60)
    print("\n--- Basis-Tests (Holm über alle Basis-Tests) ---")
    print(results[results["basistest"]][show].round(4).to_string(index=False))
    print("\n--- Kontrollen / Diagnosen (nicht korrigiert) ---")
    print(results[~results["basistest"]][show[:-1]].round(4).to_string(index=False))
    print(f"\nGespeichert: {config.RSA_PATH.name}, Plots 13_ und 14_")


if __name__ == "__main__":
    main()
