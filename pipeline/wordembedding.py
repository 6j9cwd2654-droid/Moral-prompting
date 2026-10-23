"""Embedding: Cosine-Similarity zwischen Prompt-Centroid und MFD-Centroiden -> Embedding_Scores.xlsx."""

import pandas as pd
from gensim.models import KeyedVectors

from src import config
from src.config import DIMS
from src.data_utils import load_clean_data
from src.dictionary_utils import load_dictionaries
from src.embedding_utils import build_vocab_index, calculate_dict_centroid, embedding_scores


def main():
    model = KeyedVectors.load_word2vec_format(str(config.MODEL_PATH), binary=True)
    vocab_index = build_vocab_index(model)

    data = load_clean_data(["participant_id", "prompts_all"])

    dictionaries = load_dictionaries()
    dict_centroids = {dim: calculate_dict_centroid(dictionaries[dim], model, vocab_index) for dim in DIMS}

    scores = pd.DataFrame([embedding_scores(t, model, dict_centroids, vocab_index) for t in data["prompts_all"]])
    for dim in DIMS:
        data[f"emb_{dim.lower()}"] = scores[dim].values

    data.to_excel(config.EMBEDDING_SCORES_PATH, index=False)


if __name__ == "__main__":
    main()
