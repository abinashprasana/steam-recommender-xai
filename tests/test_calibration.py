import os
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calibration import (kl_miscalibration, user_genre_distribution,
                         list_genre_distribution, calibrated_rerank)
from explain import refit_user_factor, _rank_of, render_explanation


@pytest.fixture
def games():
    return pd.DataFrame(
        {'app_name': ['A', 'B', 'C', 'D', 'E', 'F'],
         'genres': [['Action'], ['Action'], ['Action'],
                    ['Strategy'], ['Strategy'], ['Racing']]},
        index=[0, 1, 2, 3, 4, 5]
    )


def test_kl_is_zero_for_identical_distributions():
    p = {'Action': 0.7, 'Strategy': 0.3}
    assert kl_miscalibration(p, p) == pytest.approx(0.0, abs=1e-9)


def test_kl_rises_as_distributions_diverge():
    p = {'Action': 0.5, 'Strategy': 0.5}
    close = kl_miscalibration(p, {'Action': 0.6, 'Strategy': 0.4})
    far = kl_miscalibration(p, {'Action': 0.95, 'Strategy': 0.05})
    worst = kl_miscalibration(p, {'Action': 1.0})
    assert 0 < close < far < worst


def test_distributions_normalise_despite_genreless_items():
    """Games with no genres must not inflate the denominator. If they do, p and q
    sum to less than 1 and KL can come out negative, which is meaningless."""
    g = pd.DataFrame({'app_name': ['A', 'B', 'C'],
                      'genres': [['Action'], [], ['Strategy']]},
                     index=[0, 1, 2])
    p = user_genre_distribution([0, 1, 2], g)
    assert sum(p.values()) == pytest.approx(1.0)
    q = list_genre_distribution([0, 1, 2], g)
    assert sum(q.values()) == pytest.approx(1.0)


def test_kl_is_never_negative_on_real_shaped_input():
    g = pd.DataFrame({'app_name': ['A', 'B', 'C', 'D'],
                      'genres': [['Action'], [], ['Strategy'], ['Action', 'Indie']]},
                     index=[0, 1, 2, 3])
    p = user_genre_distribution([0, 1, 3], g)
    for lst in ([0], [2], [0, 2], [1, 2, 3], [0, 1, 2, 3]):
        assert kl_miscalibration(p, list_genre_distribution(lst, g)) >= -1e-12


def test_kl_is_finite_when_a_genre_is_missing():
    """Without Steck's smoothing this is infinite and the objective is unusable."""
    val = kl_miscalibration({'Action': 0.5, 'Strategy': 0.5}, {'Action': 1.0})
    assert np.isfinite(val)


def test_user_distribution_matches_history_mix(games):
    # 3 Action items, 1 Strategy item -> 75/25
    dist = user_genre_distribution([0, 1, 2, 3], games)
    assert dist['Action'] == pytest.approx(0.75)
    assert dist['Strategy'] == pytest.approx(0.25)


def test_list_distribution_matches_list_mix(games):
    dist = list_genre_distribution([0, 3], games)
    assert dist['Action'] == pytest.approx(0.5)
    assert dist['Strategy'] == pytest.approx(0.5)


def test_lambda_zero_is_pure_relevance(games):
    scores = np.array([9.0, 8.0, 7.0, 1.0, 0.5, 0.1])
    out = calibrated_rerank(scores, [0, 3], games, n=3, lam=0.0)
    assert out[:3] == [0, 1, 2]


def test_calibration_pulls_in_underrepresented_genre(games):
    """User history is half Strategy, but relevance favours Action exclusively.
    Turning lambda up should surface a Strategy title that lambda=0 misses."""
    scores = np.array([9.0, 8.0, 7.0, 1.0, 0.9, 0.1])
    history = [0, 3]

    greedy = calibrated_rerank(scores, history, games, n=2, lam=0.0)
    calibrated = calibrated_rerank(scores, history, games, n=2, lam=0.9)

    strategy = {3, 4}
    assert not (set(greedy) & strategy)
    assert set(calibrated) & strategy


def test_refit_user_factor_prefers_its_positives():
    """A refit vector should score the items it was fit on above unrelated ones."""
    rng = np.random.RandomState(0)
    item_factors = rng.randn(20, 8).astype(np.float32) * 0.5
    u = refit_user_factor(item_factors, [3, 4], n_items=20, steps=800, seed=1)
    scores = u @ item_factors.T
    assert scores[[3, 4]].mean() > np.median(scores)


def test_rank_of_masks_excluded_items():
    scores = np.array([5.0, 4.0, 3.0, 2.0])
    assert _rank_of(scores, 2, exclude=[]) == 2
    assert _rank_of(scores, 2, exclude=[0, 1]) == 0


def test_render_explanation_names_the_supporting_items():
    loo = {
        'target_name': 'Hollow Knight',
        'influences': [
            {'name': 'Ori', 'rank_delta': 5, 'score_delta': 0.2},
            {'name': 'Celeste', 'rank_delta': 3, 'score_delta': 0.1},
            {'name': 'Noise', 'rank_delta': 0, 'score_delta': 0.0},
        ],
    }
    text = render_explanation(loo)
    assert 'Hollow Knight' in text
    assert 'Ori' in text and 'Celeste' in text
    assert 'Noise' not in text


def test_render_explanation_handles_no_supporters():
    loo = {'target_name': 'X', 'influences': [{'name': 'Y', 'rank_delta': -1, 'score_delta': 0}]}
    assert 'no ' in render_explanation(loo).lower()
