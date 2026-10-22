import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
from scipy.spatial.distance import cosine
from dictionary_utils import load_dictionaries, DIMS
from embedding_utils import build_vocab_index, calculate_dict_centroid, calculate_text_centroid


# Modell laden
MODEL_PATH = 'german.model'
model = KeyedVectors.load_word2vec_format(MODEL_PATH, binary=True)

# Vokabular-Index für case-insensitive und Prefix-Lookups bauen
vocab_index = build_vocab_index(model)


# Einlesen
data = pd.read_excel('Daten_bereinigt.xlsx') # später mit SoSci-Daten ersetzen
data = data[['participant_id', 'prompts_all']]


# Dictionaries laden (mit Sternchen-Wildcards damit diese als Prefix erkannt werden)
dictionaries = load_dictionaries()


# Centroid-Berechnung für die Dictionaries
dict_centroids = {}
for dim in DIMS:
    dict_centroids[dim] = calculate_dict_centroid(dictionaries[dim], model, vocab_index)
    

# Cosine Similarity zwischen Text-Centroid und Dictionary-Centroids
def score_embedding(text, model, dict_centroids, vocab_index):
    text_centroid = calculate_text_centroid(text, model, vocab_index)
    if text_centroid is None:
        return {dim: None for dim in DIMS}
    scores = {}
    for dim in DIMS:
        if dict_centroids[dim] is not None:
            # cosine distance gibt 1 - similarity, daher 1 - cosine(...)
            scores[dim] = 1 - cosine(text_centroid, dict_centroids[dim])
        else:
            scores[dim] = None
    return scores

# Scores pro VP berechnen
results = data['prompts_all'].apply(lambda t: score_embedding(t, model, dict_centroids, vocab_index))
for dim in DIMS:
    data[f'emb_{dim.lower()}'] = results.apply(lambda x: x[dim])

# Ergebnisse speichern
data.to_excel('Embedding_Scores.xlsx', index=False)
