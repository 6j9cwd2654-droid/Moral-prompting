import numpy as np
from gensim.models import KeyedVectors
from dictionary_utils import load_dictionaries, DIMS
from embedding_utils import build_vocab_index, get_vectors_for_prefix, get_vector_case_insensitive, normalize_umlauts

MODEL_PATH = 'german.model'
model = KeyedVectors.load_word2vec_format(MODEL_PATH, binary=True)

vocab_index = build_vocab_index(model)
dictionaries = load_dictionaries()

print("=== Coverage pro Dimension (mit Prefix-Matching) ===")
for dim in DIMS:
    terms = dictionaries[dim]
    gefunden = []
    fehlend = []
    for pattern in terms:
        if pattern.endswith('*'):
            stem = normalize_umlauts(pattern.rstrip('*'))
            vectors = get_vectors_for_prefix(stem, model, vocab_index)
            if vectors:
                gefunden.append(pattern)
            else:
                fehlend.append(pattern)
        else:
            v = get_vector_case_insensitive(normalize_umlauts(pattern), model, vocab_index)
            if v is not None:
                gefunden.append(pattern)
            else:
                fehlend.append(pattern)

    total = len(terms)
    n_gefunden = len(gefunden)
    prozent = (n_gefunden / total * 100) if total > 0 else 0
    print(f"{dim}: {n_gefunden}/{total} gefunden ({prozent:.1f}%)")
    print(f"   Fehlend: {', '.join(fehlend[:10])}{' ...' if len(fehlend) > 10 else ''}")
    print()