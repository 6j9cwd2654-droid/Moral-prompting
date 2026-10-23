"""Einlesen der Eingangsdaten (immer mit explizitem Sheet-Namen)."""

import pandas as pd

import config


def load_clean_data(columns=None):
    df = pd.read_excel(config.DATA_CLEAN, sheet_name=config.DATA_CLEAN_SHEET)
    return df[columns] if columns is not None else df


def load_mfq():
    return load_clean_data(["participant_id"] + config.MFQ_COLS)


def load_raw_topic_prompts():
    """Rohdaten (unaggregierte Prompts je Thema), eingeschränkt auf die bereinigte Stichprobe."""
    raw = pd.read_excel(config.ROHDATEN_PATH, sheet_name=config.ROHDATEN_SHEET)
    raw["participant_id"] = pd.to_numeric(raw["CASE"], errors="coerce")
    raw = raw.dropna(subset=["participant_id"])
    raw["participant_id"] = raw["participant_id"].astype(int)
    clean_ids = load_clean_data(["participant_id"])["participant_id"]
    return raw[raw["participant_id"].isin(clean_ids)].copy()
