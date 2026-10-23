"""Diagnose: Wie viele Wörterbuch-Einträge findet das Word2Vec-Modell (inkl. Präfix-Matching)?"""

from gensim.models import KeyedVectors

import config
from config import DIMS
from dictionary_utils import load_dictionaries
from embedding_utils import build_vocab_index, get_vectors_for_prefix, get_vector_case_insensitive
from text_utils import normalize_umlauts


def is_covered(pattern, model, vocab_index):
    if pattern.endswith("*"):
        stem = normalize_umlauts(pattern.rstrip("*"))
        return bool(get_vectors_for_prefix(stem, model, vocab_index))
    return get_vector_case_insensitive(normalize_umlauts(pattern), model, vocab_index) is not None


def main():
    model = KeyedVectors.load_word2vec_format(str(config.MODEL_PATH), binary=True)
    vocab_index = build_vocab_index(model)
    dictionaries = load_dictionaries()

    print("=== Coverage pro Dimension (mit Prefix-Matching) ===")
    for dim in DIMS:
        terms = dictionaries[dim]
        missing = [p for p in terms if not is_covered(p, model, vocab_index)]
        found = len(terms) - len(missing)
        pct = found / len(terms) * 100 if terms else 0
        print(f"{dim}: {found}/{len(terms)} gefunden ({pct:.1f}%)")
        print(f"   Fehlend: {', '.join(missing[:10])}{' ...' if len(missing) > 10 else ''}\n")


if __name__ == "__main__":
    main()
