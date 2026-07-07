import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from gensim.utils import tokenize
from scipy.spatial.distance import cosine
from dictionary_utils import load_dictionaries, DIMS

# Modell laden
MODEL_PATH = 'german.model'
model = KeyedVectors.load_word2vec_format(MODEL_PATH, binary=True)



# Einlesen
data = pd.read_excel('Prompts.xlsx') # später mit SoSci-Daten ersetzen
data = data[['VP', 'Aggregiert']]

# Umlaut-Normalisierung
def normalize_umlauts(text):
    text = text.replace('ä', 'ae')
    text = text.replace('ö', 'oe')
    text = text.replace('ü', 'ue')
    text = text.replace('ß', 'ss')
    return text

# Dictionaries bauen (Sternchen entfernen)
raw_dictionaries = load_dictionaries()
dictionaries = {dim: [t.rstrip('*') for t in raw_dictionaries[dim]] for dim in DIMS}

# generische Centroid - Berechnung
def calculate_centroid(items, model):
    if isinstance(items, str):
        text = normalize_umlauts(items.lower())
    elif isinstance (items, list):
        text = normalize_umlauts(' '.join(items).lower())
    else:
        raise ValueError("Input must be a string or a list of strings")
    tokens = list(tokenize(text, lower=True))
    vectors = [model[token] for token in tokens if token in model]
    if not vectors:
        return None
    return np.mean(vectors, axis=0)

# Aufruf der Centroid-Berechnung für die Dictionaries
dict_centroids = {}
for dim in DIMS:
    dict_centroids[dim] = calculate_centroid(dictionaries[dim], model)

# Cosine Similarity zwischen Text-Centroid und Dictionary-Centroids
def score_embedding(text, model, dict_centroids):
    text_centroid = calculate_centroid(text, model)
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
results = data['Aggregiert'].apply(lambda t: score_embedding(t, model, dict_centroids))
for dim in DIMS:
    data[f'emb_{dim.lower()}'] = results.apply(lambda x: x[dim])

# Ergebnisse speichern
data.to_excel('Embedding_Scores.xlsx', index=False)
