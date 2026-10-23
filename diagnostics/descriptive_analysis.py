"""
Deskriptive Basisanalysen - Moral Prompting Studie (AG4)
==========================================================

Liest den bereinigten Analyse-Datensatz (Sheet "Data") und gibt eine
deskriptive Übersicht auf der Konsole aus. Zusätzlich werden ein paar
Plots als PNG-Dateien gespeichert (Ordner "plots/"), wo eine visuelle
Übersicht mehr hergibt als reine Zahlen.

Voraussetzungen: pandas, matplotlib, openpyxl
    pip install pandas matplotlib openpyxl

Nutzung:
    python -m diagnostics.descriptive_analysis
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

from src import config

# ------------------------------------------------------------------
# Konfiguration
# ------------------------------------------------------------------
DATA_PATH = config.DATA_CLEAN
SHEET_NAME = config.DATA_CLEAN_SHEET
SCORE_PATH = config.SCORING_PATH
SCORE_COLS = [f"score_{d.lower()}" for d in config.DIMS]
PLOT_DIR = config.PLOT_DIR

MFQ_DIMENSIONS = config.MFQ_COLS

# Value-Label-Mappings aus dem SoSci-Codebook (siehe Codebook-Sheet der Datei)
GENDER_LABELS = {1: "weiblich", 2: "männlich", 3: "divers", 4: "keine Angabe"}
EDUCATION_LABELS = {
    1: "kein Abschluss", 2: "Hauptschulabschluss", 4: "Realschulabschluss",
    6: "(Fach-)Abitur", 9: "Ausbildung/Lehre", 8: "Hochschulabschluss",
    10: "Bachelor", 11: "Master/Magister/Diplom", 12: "Promotion",
}
EMPLOYMENT_LABELS = {
    1: "Schüler/in", 2: "In Ausbildung", 3: "Student/in", 4: "Angestellte/r",
    5: "Beamte/r", 6: "Selbstständig", 7: "Arbeitslos/Arbeit suchend", 8: "Sonstiges",
}
AI_USAGE_LABELS = {
    1: "Täglich", 2: "Mehrmals pro Woche", 3: "Einmal pro Woche",
    4: "Mehrmals im Monat", 5: "Seltener", 6: "Noch nie",
}


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def freq_table(series, labels=None):
    """Häufigkeiten + Prozente, optional mit Klartext-Labels."""
    counts = series.value_counts(dropna=False).sort_index()
    pct = (counts / counts.sum() * 100).round(1)
    table = pd.DataFrame({"n": counts, "%": pct})
    if labels:
        table.index = [labels.get(i, f"unbekannt ({i})") for i in table.index]
    print(table)


# ------------------------------------------------------------------
# 1. Stichprobenbeschreibung
# ------------------------------------------------------------------
def describe_sample(df):
    section("1. Stichprobenbeschreibung")

    print(f"\nN = {len(df)}")

    print("\n--- Alter ---")
    print(df["age"].describe().round(2))

    print("\n--- Geschlecht ---")
    freq_table(df["gender_code"], GENDER_LABELS)

    print("\n--- Höchster Bildungsabschluss ---")
    freq_table(df["education_code"], EDUCATION_LABELS)

    print("\n--- Beschäftigung ---")
    freq_table(df["employment_code"], EMPLOYMENT_LABELS)

    print("\n--- KI-Nutzungshäufigkeit ---")
    freq_table(df["ai_usage_freq_code"], AI_USAGE_LABELS)

    # Plot: Altersverteilung
    os.makedirs(PLOT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    df["age"].plot(kind="hist", bins=15, ax=ax, edgecolor="white")
    ax.set_xlabel("Alter (Jahre)")
    ax.set_ylabel("Anzahl Personen")
    ax.set_title("Altersverteilung der Stichprobe")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "01_altersverteilung.png"), dpi=150)
    plt.close(fig)

    # Plot: KI-Nutzungshäufigkeit (Balkendiagramm, da geordnete Kategorie)
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["ai_usage_freq_code"].value_counts().sort_index()
    labels = [AI_USAGE_LABELS.get(i, str(i)) for i in counts.index]
    ax.bar(labels, counts.values)
    ax.set_ylabel("Anzahl Personen")
    ax.set_title("KI-Nutzungshäufigkeit")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "02_ki_nutzungshaeufigkeit.png"), dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------
# 2. MFQ-Scores deskriptiv
# ------------------------------------------------------------------
def describe_mfq(df):
    section("2. MFQ30-Dimensionsscores (deskriptiv)")

    desc = df[MFQ_DIMENSIONS].describe().T
    desc["median"] = df[MFQ_DIMENSIONS].median()
    print(desc.round(2))

    # Plot: Boxplot aller 5 Dimensionen nebeneinander
    labels = [c.replace("mfq_", "").replace("_score", "") for c in MFQ_DIMENSIONS]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.boxplot([df[c].dropna() for c in MFQ_DIMENSIONS])
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score (1-6)")
    ax.set_title("Verteilung der MFQ30-Dimensionsscores")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "03_mfq_boxplot.png"), dpi=150)
    plt.close(fig)

# ------------------------------------------------------------------
# 2b. Individualizing vs. Binding: Profildominanz pro Person
# ------------------------------------------------------------------
def describe_moral_profile(df):
    section("2b. Individualizing vs. Binding Profil (pro Person)")
 
    individualizing = df[["mfq_care_score", "mfq_fairness_score"]].mean(axis=1)
    binding = df[["mfq_loyalty_score", "mfq_authority_score", "mfq_purity_score"]].mean(axis=1)
    diff = individualizing - binding
 
    n = len(diff)
    n_individualizing = (diff > 0).sum()
    n_binding = (diff < 0).sum()
    pct_individualizing = round(n_individualizing / n * 100, 1)
    pct_binding = round(n_binding / n * 100, 1)
 
    print(f"\nIndividualizing-dominant (diff > 0): {n_individualizing}/{n} ({pct_individualizing}%)")
    print(f"Binding-dominant (diff < 0): {n_binding}/{n} ({pct_binding}%)")
    print(f"\nDifferenz (individualizing - binding), deskriptiv:")
    print(diff.describe().round(3))
 
    # Plot: Histogramm der Differenzwerte pro Person
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.hist(diff, bins=20, edgecolor="white", color="#4C72B0")
    ax.axvline(0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("Individualizing − Binding (pro Person)")
    ax.set_ylabel("Anzahl Personen")
    ax.set_title("Profildominanz: Individualizing vs. Binding")
    ax.text(0.02, 0.95, f"{pct_individualizing}% individualizing-dominant",
            transform=ax.transAxes, ha="left", va="top", fontsize=9)
    ax.text(0.02, 0.88, f"{pct_binding}% binding-dominant",
            transform=ax.transAxes, ha="left", va="top", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "09_individualizing_vs_binding.png"), dpi=150)
    plt.close(fig)

     # Alternative Darstellung: Streudiagramm Individualizing vs. Binding pro Person
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    colors = ["#4C72B0" if d > 0 else "#C44E52" for d in diff]
    ax.scatter(individualizing, binding, c=colors, alpha=0.7, edgecolor="white", s=40)
 
    lims = [1, 6]
    ax.plot(lims, lims, color="black", linewidth=1, linestyle="--", label="Individualizing = Binding")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Individualizing-Score (Care/Fairness, M)")
    ax.set_ylabel("Binding-Score (Loyalty/Authority/Purity, M)")
    ax.set_title("Individualizing vs. Binding pro Person")
    ax.legend(loc="upper left", fontsize=8)
    ax.text(0.98, 0.05, f"{pct_individualizing}% unterhalb der Diagonale\n(individualizing-dominant)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "10_individualizing_vs_binding_scatter.png"), dpi=150)
    plt.close(fig)

# ------------------------------------------------------------------
# 3. Interkorrelationen zwischen den MFQ-Dimensionen
# ------------------------------------------------------------------
def correlate_mfq(df):
    section("3. Interkorrelationen der MFQ30-Dimensionen (Pearson r)")

    corr = df[MFQ_DIMENSIONS].corr(method="pearson").round(2)
    print(corr)

    # Plot: Heatmap
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r")
    labels = [c.replace("mfq_", "").replace("_score", "") for c in MFQ_DIMENSIONS]
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title("MFQ30-Dimensionen: Interkorrelationen")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "04_mfq_korrelation_heatmap.png"), dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------
# 4. Prompt-Textmerkmale
# ------------------------------------------------------------------
def describe_prompts(df):
    section("4. Prompt-Textmerkmale (prompts_all)")

    text = df["prompts_all"].fillna("")
    word_count = text.str.split().str.len()
    char_count = text.str.len()

    print("\n--- Wortanzahl pro Person ---")
    print(word_count.describe().round(2))
    print("\n--- Zeichenanzahl pro Person ---")
    print(char_count.describe().round(2))

    fig, ax = plt.subplots(figsize=(6, 4))
    word_count.plot(kind="hist", bins=20, ax=ax, edgecolor="white")
    ax.set_xlabel("Wortanzahl (alle Prompts einer Person zusammen)")
    ax.set_ylabel("Anzahl Personen")
    ax.set_title("Verteilung der Prompt-Länge (Wörter)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "05_prompt_wortanzahl.png"), dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------
# 5. Fehlende Werte
# ------------------------------------------------------------------
def missing_report(df):
    section("5. Fehlende Werte je Spalte")

    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        print("\nKeine fehlenden Werte (außer erwartbaren Freitextfeldern).")
    else:
        pct = (missing / len(df) * 100).round(1)
        print(pd.DataFrame({"n_fehlend": missing, "%": pct}))

# ------------------------------------------------------------------
# 6. Wordcounting-Ansatz: Treffer pro Person nach Dimension
# ------------------------------------------------------------------
def describe_wordcount_scores(scores_path=SCORE_PATH):
    section("6. Wordcounting-Ansatz: MFD-Treffer pro Person")

    scores = pd.read_excel(scores_path)

    desc = scores[SCORE_COLS].describe().T
    desc["median"] = scores[SCORE_COLS].median()
    print(desc.round(2))

    labels = [c.replace("score_", "") for c in SCORE_COLS]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.boxplot([scores[c].dropna() for c in SCORE_COLS])
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=20)
    ax.set_ylabel("Treffer pro Person")
    ax.set_title("Verteilung der MFD-Treffer pro Person nach Dimension")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "06_hits_per_dimension_boxplot.png"), dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------
# 6b. Wordcounting-Ansatz: Treffer relativ zur individuellen Textlaenge
# ------------------------------------------------------------------
def describe_wordcount_rates(scores_path=SCORE_PATH):
    section("6b. Wordcounting-Ansatz: MFD-Treffer relativ zur Textlänge")

    scores = pd.read_excel(scores_path)
    rate_cols = [c.replace("score_", "rate_") for c in SCORE_COLS]
    for score_col, rate_col in zip(SCORE_COLS, rate_cols):
        scores[rate_col] = scores[score_col] / scores["n_words"] * 100

    desc = scores[rate_cols].describe().T
    desc["median"] = scores[rate_cols].median()
    print(desc.round(3))

    labels = [c.replace("rate_", "") for c in rate_cols]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.boxplot([scores[c].dropna() for c in rate_cols])
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=20)
    ax.set_ylabel("Treffer pro 100 Wörter")
    ax.set_title("Verteilung der MFD-Treffer pro Person nach Dimension\n(relativ zur individuellen Wortzahl)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "06b_hits_per_word_boxplot.png"), dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)

    describe_sample(df)
    describe_mfq(df)
    describe_moral_profile(df)
    correlate_mfq(df)
    describe_prompts(df)
    missing_report(df)
    describe_wordcount_scores()
    describe_wordcount_rates()
    print(f"\nPlots gespeichert in: {PLOT_DIR}")


if __name__ == "__main__":
    main()
