import bisect
import numpy as np
from gensim.utils import tokenize
from scipy.spatial.distance import cosine

from config import DIMS
from text_utils import normalize_umlauts


def build_vocab_index(model):
    """
    Baut einen Index für case-insensitive und Prefix-Lookups im Modell-Vokabular.
    Wird nur einmal pro Modell aufgerufen, nicht pro VP oder Dictionary-Term.
    """
    pairs = sorted((word.lower(), word) for word in model.key_to_index)
    lower_to_key = {}
    for lower, original in pairs:
        lower_to_key.setdefault(lower, original)  # ersten Treffer behalten
    sorted_lower_keys = [p[0] for p in pairs]
    return lower_to_key, sorted_lower_keys, pairs


def get_vector_case_insensitive(word, model, vocab_index):
    """Sucht ein Wort unabhängig von Groß-/Kleinschreibung im Modell."""
    lower_to_key, _, _ = vocab_index
    key = lower_to_key.get(word.lower())
    return model[key] if key is not None else None


def get_vectors_for_prefix(stem, model, vocab_index):
    """Findet alle Vektoren von Wörtern die mit `stem` beginnen (case-insensitive)."""
    _, sorted_lower_keys, sorted_pairs = vocab_index
    stem = stem.lower()
    start = bisect.bisect_left(sorted_lower_keys, stem)
    end = bisect.bisect_left(sorted_lower_keys, stem + '\uffff')
    return [model[original] for _, original in sorted_pairs[start:end]]


def calculate_dict_centroid(patterns, model, vocab_index):
    """Centroid für ein Dictionary (Liste von Termen, ggf. mit Wildcard '*')."""
    vectors = []
    for pattern in patterns:
        if pattern.endswith('*'):
            stem = normalize_umlauts(pattern.rstrip('*'))
            vectors += get_vectors_for_prefix(stem, model, vocab_index)
        else:
            v = get_vector_case_insensitive(normalize_umlauts(pattern), model, vocab_index)
            if v is not None:
                vectors.append(v)
    if not vectors:
        return None
    return np.mean(vectors, axis=0)


def calculate_text_centroid(text, model, vocab_index):
    """Centroid für einen Prompt-Text einer VP."""
    if text is None or (isinstance(text, float)):
        raise ValueError("Leerer Text für eine VP — bitte prüfen")
    text = normalize_umlauts(text)
    tokens = list(tokenize(text, lower=True))
    vectors = []
    for token in tokens:
        v = get_vector_case_insensitive(token, model, vocab_index)
        if v is not None:
            vectors.append(v)
    if not vectors:
        return None
    return np.mean(vectors, axis=0)


def embedding_scores(text, model, dict_centroids, vocab_index):
    """Cosine-Similarity eines Texts zu den Dimensions-Centroiden (NaN, wenn nicht berechenbar)."""
    nan_scores = {dim: np.nan for dim in DIMS}
    if text is None:
        return nan_scores
    text_centroid = calculate_text_centroid(text, model, vocab_index)
    if text_centroid is None:
        return nan_scores
    return {
        dim: (1 - cosine(text_centroid, dict_centroids[dim])) if dict_centroids[dim] is not None else np.nan
        for dim in DIMS
    }
