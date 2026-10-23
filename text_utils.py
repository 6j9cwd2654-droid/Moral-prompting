"""Textaufbereitung, die von Wordcounting und Embedding gemeinsam genutzt wird."""

import re

import pandas as pd


def normalize_umlauts(text):
    """ä/ö/ü/ß -> ae/oe/ue/ss (Wortlisten und das Word2Vec-Modell nutzen diese Schreibweise)."""
    return (text.replace("ä", "ae").replace("ö", "oe")
                .replace("ü", "ue").replace("ß", "ss"))


def preprocess_text(text):
    """Kleinschreibung, Umlaute normalisieren, Satzzeichen entfernen, in Wörter splitten."""
    text = normalize_umlauts(text.lower())
    text = re.sub(r"[^\w\s]", "", text)
    return text.split()


def build_topic_text(row, cols):
    """Fügt die vorhandenen Prompt-Felder eines Themas zusammen; None, wenn alle leer sind."""
    parts = [str(row[c]).strip() for c in cols if pd.notna(row[c]) and str(row[c]).strip()]
    return " ".join(parts) if parts else None
