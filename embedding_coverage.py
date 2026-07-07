import numpy as np
from gensim.models import KeyedVectors
from gensim.utils import tokenize
from dictionary_utils import load_dictionaries, DIMS

MODEL_PATH = 'german.model'
model = KeyedVectors.load_word2vec_format(MODEL_PATH, binary=True)

def normalize_umlauts(text):
    text = text.replace('ä', 'ae')
    text = text.replace('ö', 'oe')
    text = text.replace('ü', 'ue')
    text = text.replace('ß', 'ss')
    return text

raw_dictionaries = load_dictionaries()
dictionaries = {dim: [t.rstrip('*') for t in raw_dictionaries[dim]] for dim in DIMS}

print("=== Coverage pro Dimension ===")
for dim in DIMS:
    terms = dictionaries[dim]
    gefunden = []
    fehlend = []
    for term in terms:
        normalized = normalize_umlauts(term.lower())
        tokens = list(tokenize(normalized, lower=True))
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