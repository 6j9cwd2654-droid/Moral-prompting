import numpy as np
import pytest
from scipy import stats

from src.rsa_utils import (abs_difference_matrix, center_rows, euclidean_distance_matrix,
                           mantel_test, mismatch_matrix, zscore_columns)


def make_profiles(n=60, k=5, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n, k))
    other = base + noise * rng.normal(size=(n, k))
    return base, other


def test_distance_matrix_is_symmetric_with_zero_diagonal():
    X = np.random.default_rng(1).normal(size=(10, 5))
    D = euclidean_distance_matrix(X)
    assert np.allclose(D, D.T)
    assert np.allclose(np.diag(D), 0)
    assert D[0, 1] == pytest.approx(np.linalg.norm(X[0] - X[1]))


def test_mantel_r_equals_pearson_of_pair_vectors():
    a, b = make_profiles(noise=1.0)
    Dx, Dy = euclidean_distance_matrix(a), euclidean_distance_matrix(b)
    i, j = np.triu_indices(len(a), 1)
    res = mantel_test(Dx, Dy, n_perm=50)
    assert res["r"] == pytest.approx(stats.pearsonr(Dx[i, j], Dy[i, j])[0])
    assert res["n_pairs"] == len(a) * (len(a) - 1) // 2


def test_identical_structure_is_significant():
    a, _ = make_profiles()
    D = euclidean_distance_matrix(a)
    res = mantel_test(D, D, n_perm=500)
    assert res["r"] == pytest.approx(1.0)
    assert res["p_one_sided"] < 0.01


def test_independent_data_is_not_systematically_significant():
    # Unter H0 sind p-Werte (annähernd) gleichverteilt: wenige kleine, Mittelwert um 0.5
    pvals = []
    for seed in range(40):
        a, _ = make_profiles(n=30, seed=seed)
        b, _ = make_profiles(n=30, seed=1000 + seed)
        pvals.append(mantel_test(euclidean_distance_matrix(a), euclidean_distance_matrix(b),
                                 n_perm=200, seed=seed)["p_one_sided"])
    assert 0.35 < np.mean(pvals) < 0.65
    assert np.mean(np.array(pvals) < 0.05) < 0.2


def test_permutation_test_is_reproducible_with_seed():
    a, b = make_profiles(noise=2.0)
    Dx, Dy = euclidean_distance_matrix(a), euclidean_distance_matrix(b)
    assert mantel_test(Dx, Dy, n_perm=100, seed=7) == mantel_test(Dx, Dy, n_perm=100, seed=7)


def test_partial_mantel_removes_confounded_association():
    # x und y hängen nur über das gemeinsame Merkmal 'z' zusammen
    rng = np.random.default_rng(3)
    n = 80
    z = rng.normal(size=n)
    x = (z + 0.3 * rng.normal(size=n))[:, None]
    y = (z + 0.3 * rng.normal(size=n))[:, None]
    Dx, Dy, Dz = (euclidean_distance_matrix(v) for v in (x, y, z[:, None]))
    plain = mantel_test(Dx, Dy, n_perm=300)
    partial = mantel_test(Dx, Dy, covariates=[Dz], n_perm=300)
    assert plain["r"] > 0.5
    assert abs(partial["r"]) < plain["r"] / 2
    assert partial["p_one_sided"] > 0.01


def test_center_rows_removes_overall_level_only():
    X = np.array([[1.0, 2.0, 3.0], [11.0, 12.0, 13.0]])
    centered = center_rows(X)
    assert np.allclose(centered[0], centered[1])
    assert np.allclose(centered.mean(axis=1), 0)


def test_zscore_columns_and_helper_matrices():
    X = np.random.default_rng(5).normal(5, 3, size=(40, 5))
    Z = zscore_columns(X)
    assert np.allclose(Z.mean(axis=0), 0) and np.allclose(Z.std(axis=0), 1)
    assert abs_difference_matrix([1, 4])[0, 1] == 3
    assert mismatch_matrix(["a", "b", "a"]).tolist() == [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
