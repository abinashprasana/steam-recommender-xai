"""The PyTorch port is only trustworthy if it agrees with the numpy reference.

These run on a small synthetic dataset with planted structure so they stay fast
enough for the normal test loop. The full-scale parity check against real data
lives in train.py, which compares NDCG@10 between the two implementations.
"""
import os
import sys
import numpy as np
import pytest
from scipy.sparse import csr_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recommender import BayesianPersonalizedRanking

torch = pytest.importorskip("torch")
from recommender_torch import BPRTorch


def planted_csr(n_users=60, n_items=40, seed=0):
    """Two user groups with disjoint item preferences -- recoverable structure."""
    rng = np.random.RandomState(seed)
    rows, cols = [], []
    for u in range(n_users):
        group = u % 2
        pool = range(0, n_items // 2) if group == 0 else range(n_items // 2, n_items)
        for i in rng.choice(list(pool), size=6, replace=False):
            rows.append(u)
            cols.append(i)
    return csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                      shape=(n_users, n_items))


def group_hit_rate(model, csr, n_items, k=10):
    """Share of top-k that lands in the user's own group's item block."""
    n_users = csr.shape[0]
    half = n_items // 2
    total, good = 0, 0
    for u in range(n_users):
        seen = set(csr.indices[csr.indptr[u]:csr.indptr[u + 1]].tolist())
        scores = np.asarray(model.score_all(u), dtype=np.float64).copy()
        scores[list(seen)] = -np.inf
        top = np.argsort(-scores)[:k]
        in_group = [i for i in top if (i < half) == (u % 2 == 0)]
        good += len(in_group)
        total += len(top)
    return good / total


def test_torch_recovers_planted_structure():
    csr = planted_csr()
    m = BPRTorch(60, 40, factors=16, iterations=60, random_state=0).fit(csr, verbose=False)
    assert group_hit_rate(m, csr, 40) > 0.75


def test_torch_and_numpy_agree_on_planted_structure():
    """Both implementations should recover the same structure. They will not be
    numerically identical -- different optimisers, different sampling order -- so
    the assertion is on behaviour, not on weights."""
    csr = planted_csr()

    np_model = BayesianPersonalizedRanking(factors=16, iterations=60, random_state=0).fit(csr)
    pt_model = BPRTorch(60, 40, factors=16, iterations=60, random_state=0).fit(csr, verbose=False)

    np_rate = group_hit_rate(np_model, csr, 40)
    pt_rate = group_hit_rate(pt_model, csr, 40)

    assert abs(np_rate - pt_rate) < 0.20, f"numpy={np_rate:.3f} torch={pt_rate:.3f}"


def test_score_all_shape_and_factor_accessors():
    csr = planted_csr()
    m = BPRTorch(60, 40, factors=16, iterations=5, random_state=0).fit(csr, verbose=False)
    assert m.score_all(0).shape == (40,)
    assert m.user_factors.shape == (60, 16)
    assert m.item_factors.shape == (40, 16)
