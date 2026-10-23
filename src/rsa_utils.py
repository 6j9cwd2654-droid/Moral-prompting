"""Bausteine für die Representational Similarity Analysis (RSA) mit Mantel-Test."""

import numpy as np


def zscore_columns(X):
    """Spaltenweise z-Transformation (ddof=0)."""
    X = np.asarray(X, dtype=float)
    return (X - X.mean(axis=0)) / X.std(axis=0)


def center_rows(X):
    """Zieht von jeder Person den Mittelwert über ihre Dimensionen ab (entfernt das Gesamtniveau,
    es bleibt nur das Muster)."""
    X = np.asarray(X, dtype=float)
    return X - X.mean(axis=1, keepdims=True)


def euclidean_distance_matrix(X):
    """n x n Matrix der euklidischen Abstände zwischen allen Zeilen von X (n Personen x k Dimensionen)."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    diff = X[:, None, :] - X[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))


def abs_difference_matrix(values):
    """n x n Matrix der absoluten Differenzen eines einzelnen Merkmals (z. B. Alter, Textlänge)."""
    values = np.asarray(values, dtype=float)
    return np.abs(values[:, None] - values[None, :])


def mismatch_matrix(labels):
    """n x n Matrix: 1, wenn zwei Personen unterschiedliche Kategorien haben, sonst 0."""
    labels = np.asarray(labels)
    return (labels[:, None] != labels[None, :]).astype(float)


def _residualize(vec, covariate_columns):
    """Residuen von vec nach linearer Regression auf [1, Kovariaten]."""
    design = np.column_stack([np.ones(len(vec))] + list(covariate_columns))
    beta, *_ = np.linalg.lstsq(design, vec, rcond=None)
    return vec - design @ beta


def mantel_test(dist_x, dist_y, covariates=None, n_perm=10000, seed=42, return_null=False):
    """Mantel-Test (optional partiell) zwischen zwei Distanzmatrizen.

    Teststatistik: Pearson-Korrelation der Matrixeinträge unterhalb der Diagonale
    (bei Kovariaten: der Residuen nach Herausrechnen der Kovariaten-Distanzen).
    Signifikanz per Permutationstest: die Personenzuordnung der x-Matrix (bzw. ihrer
    Residuen, Smouse et al. 1986) wird zufällig gemischt, weil die Matrixeinträge nicht
    unabhängig sind (jede Person steckt in n-1 Paaren).

    Rückgabe: dict mit r, p_one_sided (Hypothese r > 0), p_two_sided, n_pairs.
    """
    dist_x = np.asarray(dist_x, dtype=float)
    dist_y = np.asarray(dist_y, dtype=float)
    n = dist_x.shape[0]
    i, j = np.triu_indices(n, k=1)

    x, y = dist_x[i, j], dist_y[i, j]
    if covariates:
        cov_cols = [np.asarray(c, dtype=float)[i, j] for c in covariates]
        x_res, y_res = _residualize(x, cov_cols), _residualize(y, cov_cols)
    else:
        x_res, y_res = x - x.mean(), y - y.mean()

    # Residuen wieder als symmetrische Matrix, damit Personen-Labels permutiert werden können
    res_matrix = np.zeros((n, n))
    res_matrix[i, j] = x_res
    res_matrix[j, i] = x_res

    y_c = y_res - y_res.mean()
    y_std = y_c / np.sqrt((y_c ** 2).mean())
    n_pairs = len(y_c)

    def corr_with_y(vec):
        v = vec - vec.mean()
        return float(v @ y_std / (np.sqrt(v @ v) * np.sqrt(n_pairs)))

    r_obs = corr_with_y(x_res)

    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for k in range(n_perm):
        perm = rng.permutation(n)
        null[k] = corr_with_y(res_matrix[perm[i], perm[j]])

    result = {
        "r": r_obs,
        "p_one_sided": (1 + np.sum(null >= r_obs)) / (n_perm + 1),
        "p_two_sided": (1 + np.sum(np.abs(null) >= abs(r_obs))) / (n_perm + 1),
        "n_pairs": n_pairs,
    }
    if return_null:
        result["null"] = null
    return result
