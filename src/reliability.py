"""Übereinstimmungsmaße für Mehrfachbewertungen."""

import numpy as np


def icc_two_way(data):
    """Intraklassenkorrelation ICC(2,1) und ICC(2,k) nach Shrout und Fleiss.

    data: n Objekte (Zeilen) x k Bewertungen (Spalten, z. B. Läufe). Zeilen mit fehlenden Werten
    müssen vorher entfernt werden. Gibt (icc_einzel, icc_mittel) zurück; NaN, wenn keine Varianz vorliegt.
    """
    x = np.asarray(data, dtype=float)
    n, k = x.shape
    grand = x.mean()
    ss_rows = k * ((x.mean(axis=1) - grand) ** 2).sum()
    ss_cols = n * ((x.mean(axis=0) - grand) ** 2).sum()
    ss_error = ((x - grand) ** 2).sum() - ss_rows - ss_cols
    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))
    denom_single = ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n
    denom_mean = ms_rows + (ms_cols - ms_error) / n
    if denom_single <= 0 or denom_mean <= 0:
        return np.nan, np.nan
    return (ms_rows - ms_error) / denom_single, (ms_rows - ms_error) / denom_mean
