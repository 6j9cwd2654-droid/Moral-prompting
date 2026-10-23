import numpy as np
import pytest

from src.stats_utils import fisher_z_average, holm_bonferroni, profile_correlation


def test_holm_matches_textbook_example():
    # sortiert 0.01, 0.02, 0.04 -> *3, *2, *1, danach monoton und gedeckelt bei 1
    assert holm_bonferroni([0.04, 0.01, 0.02]) == pytest.approx([0.04, 0.03, 0.04])
    assert holm_bonferroni([0.5, 0.9]) == pytest.approx([1.0, 1.0])


def test_fisher_z_average_ignores_nan_and_is_symmetric():
    assert fisher_z_average([0.3, -0.3]) == pytest.approx(0.0)
    assert fisher_z_average([0.5, np.nan]) == pytest.approx(0.5)
    assert np.isnan(fisher_z_average([np.nan, np.nan]))


def test_profile_correlation_is_nan_for_variance_free_raw_signal():
    mfq = [5, 5, 3, 3, 2]
    assert np.isnan(profile_correlation([0, 0, 0, 0, 0], [-1, 1, 0, 0.5, 2], mfq))
    assert np.isnan(profile_correlation([1, np.nan, 2, 3, 4], [1, 2, 3, 4, 5], mfq))


def test_profile_correlation_perfect_match():
    mfq = [1, 2, 3, 4, 5]
    assert profile_correlation([2, 4, 6, 8, 10], [1, 2, 3, 4, 5], mfq) == pytest.approx(1.0)
