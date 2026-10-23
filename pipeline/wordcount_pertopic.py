"""Wordcounting pro Thema: MFD-Treffer und Wortzahl je Person und Themenblock
-> results/Scoring_PerTopic.xlsx

Gegenstück zu wordembedding_pertopic.py. Benötigt die unbereinigten Rohdaten
(config.ROHDATEN_PATH), die nicht im Repo liegen.
"""

import pandas as pd

from src import config
from src.config import DIMS, TOPICS
from src.data_utils import load_raw_topic_prompts
from src.dictionary_utils import load_dictionaries
from src.text_utils import build_topic_text, find_matches, preprocess_text


def main():
    raw = load_raw_topic_prompts()
    dictionaries = load_dictionaries()

    result = raw[["participant_id"]].copy()
    for topic, cols in TOPICS.items():
        texts = raw.apply(lambda row: build_topic_text(row, cols), axis=1)
        words = texts.apply(lambda t: preprocess_text(t) if t else [])
        result[f"{topic}_n_words"] = words.apply(len)
        for dim in DIMS:
            result[f"{topic}_score_{dim.lower()}"] = words.apply(
                lambda w: len(find_matches(w, dictionaries[dim]))
            )

    result.to_excel(config.SCORING_PERTOPIC_PATH, index=False)
    print(f"Gespeichert: {config.SCORING_PERTOPIC_PATH.name} (N={len(result)})")


if __name__ == "__main__":
    main()
