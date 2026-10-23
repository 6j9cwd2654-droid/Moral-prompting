"""Zentrale Konfiguration: Pfade, Dimensionen, Themen-Zuordnung."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

# Eingangsdaten
DATA_CLEAN = DATA_DIR / "Daten_bereinigt.xlsx"
DATA_CLEAN_SHEET = "Data"
MFD_PATH = DATA_DIR / "mfd_de_referenz.xlsx"
MFD_SHEET = "MFD_DE"
MODEL_PATH = ROOT / "models" / "german.model"

# Unbereinigte SoSci-Rohdaten liegen bewusst nicht im Repo. Pfad kann per
# Umgebungsvariable MORAL_PROMPTING_ROHDATEN ueberschrieben werden.
ROHDATEN_PATH = Path(os.environ.get(
    "MORAL_PROMPTING_ROHDATEN",
    "/Users/finn.recktenwald/Library/CloudStorage/OneDrive-adessoGroup/Dokumente/"
    "Privat/Master Kognitionswissenschaft/Praktikum und Masterarbeit/"
    "Arbeitsdokumente/Rohdaten.xlsx",
))
ROHDATEN_SHEET = "ag4_moral_prompting_01"

# Zwischen- und Endergebnisse
SCORING_PATH = RESULTS_DIR / "Scoring_Ergebnisse.xlsx"
EMBEDDING_SCORES_PATH = RESULTS_DIR / "Embedding_Scores.xlsx"
WORDCOUNT_ANALYSIS_PATH = RESULTS_DIR / "Wordcounting_Analysis.xlsx"
EMBEDDING_ANALYSIS_PATH = RESULTS_DIR / "Embedding_Analysis.xlsx"
DIMENSION_EFFECTS_PATH = RESULTS_DIR / "Dimension_Effekte.xlsx"
PERTOPIC_ANALYSIS_PATH = RESULTS_DIR / "Embedding_PerTopic_Analysis.xlsx"
SCORING_PERTOPIC_PATH = RESULTS_DIR / "Scoring_PerTopic.xlsx"
RSA_PATH = RESULTS_DIR / "RSA_Ergebnisse.xlsx"
BOOTSTRAP_PATH = RESULTS_DIR / "bootstrap_centroid_separation.xlsx"
PLOT_DIR = RESULTS_DIR / "plots"

# Moral-Foundations-Dimensionen (Reihenfolge ist Konvention im gesamten Projekt)
DIMS = ["Care", "Fairness", "Loyalty", "Authority", "Purity"]
MFQ_COLS = [f"mfq_{d.lower()}_score" for d in DIMS]

# Prompt-Felder je Thema in den SoSci-Rohdaten. Spaltenzuordnung ist fix,
# unabhaengig von der randomisierten Anzeigereihenfolge (am Inhalt geprueft).
TOPICS = {
    "waffenlieferungen": ["P101_01", "P101_02", "P101_03"],
    "gesellschaftsdienst": ["P201_01", "P201_02", "P201_03"],
    "sterbehilfe": ["P301_01", "P301_02", "P301_03"],
}

# RSA (Mantel-Test)
N_PERMUTATIONS = 10000
RANDOM_SEED = 42
