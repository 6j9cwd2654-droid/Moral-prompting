"""Bewertungseinheiten für den LLM-Bewerter: Person, Thema oder Einzelprompt."""

import numpy as np
import pandas as pd

from src import config
from src.data_utils import load_clean_data, load_raw_topic_prompts
from src.text_utils import build_topic_text

UNITS = ("person", "topic", "prompt")


def build_units(unit):
    """DataFrame mit unit_id, participant_id, topic, text."""
    rows = []
    if unit == "person":
        clean = load_clean_data(["participant_id", "prompts_all"])
        for pid, text in zip(clean["participant_id"], clean["prompts_all"]):
            rows.append((f"p{pid}", pid, "alle", text))
    elif unit in ("topic", "prompt"):
        raw = load_raw_topic_prompts()
        for _, row in raw.iterrows():
            pid = int(row["participant_id"])
            for topic, cols in config.TOPICS.items():
                if unit == "topic":
                    text = build_topic_text(row, cols)
                    if text:
                        rows.append((f"p{pid}_{topic}", pid, topic, text))
                else:
                    for col in cols:
                        if pd.notna(row[col]) and str(row[col]).strip():
                            rows.append((f"p{pid}_{topic}_{col[-2:]}", pid, topic, str(row[col]).strip()))
    else:
        raise ValueError(f"Unbekannte Einheit {unit!r}, erlaubt: {UNITS}")
    return pd.DataFrame(rows, columns=["unit_id", "participant_id", "topic", "text"])


def dev_participant_ids(participant_ids, size=config.DEV_SUBSET_SIZE, seed=config.RANDOM_SEED):
    """Feste, zufällige Entwicklungsteilmenge. Die Rubrik wird nur hieran verfeinert."""
    ids = np.sort(np.asarray(list(participant_ids)))
    return set(np.random.default_rng(seed).choice(ids, size=size, replace=False).tolist())


def select_subset(units, subset):
    """subset: 'dev' (Entwicklung), 'holdout' (Rest, für die Validierung) oder 'all'."""
    if subset == "all":
        return units
    dev = dev_participant_ids(units["participant_id"].unique())
    mask = units["participant_id"].isin(dev)
    return units[mask] if subset == "dev" else units[~mask]
