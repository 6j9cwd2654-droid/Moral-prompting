import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from gensim.utils import tokenize
from scipy.spatial.distance import cosine

# Modell laden
MODEL_PATH = 'german.model'
model = KeyedVectors.load_word2vec_format(MODEL_PATH, binary=True)

# Einlesen
dict_df = pd.read_excel('mfd_de_referenz.xlsx')
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
dims = ['Care', 'Fairness', 'Loyalty', 'Authority', 'Purity']
dictionaries = {}
for dim in dims:
    terms = dict_df[dict_df[dim] == 1]['match_pattern'].tolist()
    dictionaries[dim] = [t.rstrip('*') for t in terms]

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
for dim in dims:
    dict_centroids[dim] = calculate_centroid(dictionaries[dim], model)

# Coverage-Diagnostik: Wie viele Dictionary-Wörter sind im Modell?
print("=== Coverage pro Dimension ===")
for dim in dims:
    terms = dictionaries[dim]
    gefunden = []
    fehlend = []
    for term in terms:
        normalized = normalize_umlauts(term.lower())
        tokens = list(tokenize(normalized, lower=True))
        # ein Term gilt als gefunden wenn mindestens ein Token im Modell ist
        if any(tok in model for tok in tokens):
            gefunden.append(term)
        else:
            fehlend.append(term)
    total = len(terms)
    n_gefunden = len(gefunden)
    prozent = (n_gefunden / total * 100) if total > 0 else 0
    print(f"{dim}: {n_gefunden}/{total} gefunden ({prozent:.1f}%)")
    print(f"   Fehlend: {', '.join(fehlend[:10])}{' ...' if len(fehlend) > 10 else ''}")
    print()

# Cosine Similarity zwischen Text-Centroid und Dictionary-Centroids
def score_embedding(text, model, dict_centroids):
    text_centroid = calculate_centroid(text, model)
    if text_centroid is None:
        return {dim: None for dim in dims}
    scores = {}
    for dim in dims:
        if dict_centroids[dim] is not None:
            # cosine distance gibt 1 - similarity, daher 1 - cosine(...)
            scores[dim] = 1 - cosine(text_centroid, dict_centroids[dim])
        else:
            scores[dim] = None
    return scores

# Scores pro VP berechnen
results = data['Aggregiert'].apply(lambda t: score_embedding(t, model, dict_centroids))
for dim in dims:
    data[f'emb_{dim.lower()}'] = results.apply(lambda x: x[dim])

# Ergebnisse speichern
data.to_excel('Embedding_Scores.xlsx', index=False)
