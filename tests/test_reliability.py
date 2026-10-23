import numpy as np
import pytest

from src.reliability import icc_two_way


def test_identical_ratings_give_icc_one():
    base = np.random.default_rng(0).normal(size=(40, 1))
    single, mean = icc_two_way(np.repeat(base, 3, axis=1))
    assert single == pytest.approx(1.0) and mean == pytest.approx(1.0)


def test_pure_noise_gives_icc_near_zero():
    data = np.random.default_rng(1).normal(size=(200, 3))
    single, _ = icc_two_way(data)
    assert abs(single) < 0.15


def test_icc_matches_known_variance_ratio():
    # wahre Varianz 1, Fehlervarianz 1 -> ICC(2,1) ist etwa 0.5, ICC(2,3) etwa 0.75
    rng = np.random.default_rng(2)
    truth = rng.normal(size=(2000, 1))
    data = truth + rng.normal(size=(2000, 3))
    single, mean = icc_two_way(data)
    assert single == pytest.approx(0.5, abs=0.05)
    assert mean == pytest.approx(0.75, abs=0.05)


def test_constant_data_returns_nan():
    single, mean = icc_two_way(np.ones((10, 3)))
    assert np.isnan(single) and np.isnan(mean)
