"""Statistische Hilfsfunktionen, die von mehreren Analysen gemeinsam genutzt werden."""

import numpy as np
from scipy import stats


def holm_bonferroni(pvalues):
    """Holm-Bonferroni-korrigierte p-Werte (gleiche Reihenfolge wie die Eingabe)."""
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues)
    m = len(pvalues)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        running_max = max(running_max, (m - rank) * pvalues[idx])
        adjusted[idx] = min(running_max, 1.0)
    return adjusted


def fisher_z_average(r_values):
    """Mittelt Korrelationen korrekt über Fisher-Z; NaN-Werte werden ignoriert."""
    r_values = [r for r in r_values if not np.isnan(r)]
    if not r_values:
        return np.nan
    z_values = np.arctanh(np.clip(r_values, -0.999999, 0.999999))
    return np.tanh(np.mean(z_values))


def profile_correlation(raw_vec, score_vec, mfq_vec):
    """Pearson-r zwischen MFQ-Profil und (z-transformiertem) Sprach-Profil einer Person.

    Ist das Rohsignal fehlend oder in allen Dimensionen identisch (i.d.R. überall
    0 Treffer), hat es keine Varianz. Die z-Transformation würde daraus trotzdem
    einen scheinbar variierenden Vektor erzeugen, der nur die unterschiedlichen
    Wortlisten-Längen je Dimension widerspiegelt. Die Korrelation ist dann nicht
    definiert (NaN) statt ein reines Transformationsartefakt zu sein.
    """
    raw_vec = np.asarray(raw_vec, dtype=float)
    if np.isnan(raw_vec).any() or np.ptp(raw_vec) == 0:
        return np.nan
    r, _ = stats.pearsonr(np.asarray(mfq_vec, dtype=float), np.asarray(score_vec, dtype=float))
    return r
