"""Wordcounting: zählt MFD-Treffer pro Person und Dimension -> Scoring_Ergebnisse.xlsx."""

import pandas as pd

import config
from config import DIMS
from data_utils import load_clean_data
from dictionary_utils import load_dictionaries
from text_utils import preprocess_text


def find_matches(words, dictionary):
    """Alle Wörter, die auf einen Wörterbuch-Eintrag passen (Wildcard '*' = Präfix-Match)."""
    matches = []
    for pattern in dictionary:
        if pattern.endswith("*"):
            stem = pattern.rstrip("*")
            matches += [w for w in words if w.startswith(stem)]
        else:
            matches += [w for w in words if w == pattern]
    return matches


def score_person(text, dictionaries):
    if pd.isna(text):
        raise ValueError("Text is null")
    words = preprocess_text(text)
    row = {"n_words": len(words)}
    for dim in DIMS:
        matches = find_matches(words, dictionaries[dim])
        row[f"score_{dim.lower()}"] = len(matches)
        row[f"matches_{dim.lower()}"] = ", ".join(matches)
    return row


def main():
    data = load_clean_data(["participant_id", "prompts_all"])
    dictionaries = load_dictionaries()

    scored = pd.DataFrame([score_person(t, dictionaries) for t in data["prompts_all"]])
    result = pd.concat([data[["participant_id"]], scored], axis=1)

    column_order = ["participant_id", "n_words"]
    for dim in DIMS:
        column_order += [f"score_{dim.lower()}", f"matches_{dim.lower()}"]
    result[column_order].to_excel(config.SCORING_PATH, index=False)


if __name__ == "__main__":
    main()
