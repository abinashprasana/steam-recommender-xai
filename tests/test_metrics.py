import os
import sys
import numpy as np
import pytest
from scipy.sparse import csr_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation import ndcg_at_k, map_at_k, recall_at_k, hit_rate_at_k, gini
from baselines import EASE, PopularityRecommender, ItemKNN


def test_perfect_ranking_scores_one():
    assert ndcg_at_k([1, 2, 3, 9, 8], {1, 2, 3}, k=5) == pytest.approx(1.0)


def test_empty_relevant_is_zero():
    assert ndcg_at_k([1, 2, 3], set(), k=3) == 0.0


def test_inverted_ranking_scores_low():
    ranked = list(range(20))
    relevant = {17, 18, 19}
    assert ndcg_at_k(ranked, relevant, k=10) == 0.0


def test_idcg_uses_relevant_count_not_retrieved_hits():
    """The bug this catches: IDCG built from retrieved hits instead of the true
    relevant count. A model that surfaces 1 of 5 relevant items at rank 5 should
    NOT score the same as one that surfaces the only relevant item at rank 5."""
    ranked = [90, 91, 92, 93, 1]

    one_of_five = ndcg_at_k(ranked, {1, 2, 3, 4, 5}, k=5)
    one_of_one = ndcg_at_k(ranked, {1}, k=5)

    assert one_of_five < one_of_one
    # Under the old implementation both collapsed to the same value.
    assert one_of_one == pytest.approx(1.0 / np.log2(6))


def test_map_rewards_earlier_hits():
    early = map_at_k([1, 50, 51, 52], {1}, k=4)
    late = map_at_k([50, 51, 52, 1], {1}, k=4)
    assert early > late


def test_recall_and_hitrate():
    assert recall_at_k([1, 2, 99], {1, 2, 3, 4}, k=3) == pytest.approx(0.5)
    assert hit_rate_at_k([1, 99], {1}, k=2) == 1.0
    assert hit_rate_at_k([98, 99], {1}, k=2) == 0.0


def test_gini_uniform_is_low_concentrated_is_high():
    uniform = gini({i: 10 for i in range(50)}, 50)
    concentrated = gini({0: 500}, 50)
    assert uniform < 0.05
    assert concentrated > 0.9


def _toy_csr():
    # 4 users, 5 items. Users 0/1 share taste; users 2/3 share a different taste.
    rows = [0, 0, 1, 1, 2, 2, 3, 3]
    cols = [0, 1, 0, 1, 3, 4, 3, 4]
    return csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(4, 5))


def test_ease_closed_form_shape_and_zero_diagonal():
    m = EASE(reg=1.0).fit(_toy_csr())
    assert m.B.shape == (5, 5)
    assert np.allclose(np.diag(m.B), 0.0), "EASE must not let an item explain itself"


def test_ease_scores_co_occurring_items():
    """Item 1 co-occurs with item 0, so a user who has item 0 should score item 1
    above the items they have never co-occurred with."""
    m = EASE(reg=0.1).fit(_toy_csr())
    scores = m.score_all(0)
    assert scores[1] > scores[3]
    assert scores[1] > scores[4]


def test_popularity_is_identical_for_every_user():
    m = PopularityRecommender().fit(_toy_csr())
    assert np.array_equal(m.score_all(0), m.score_all(3))


def test_itemknn_zero_self_similarity():
    m = ItemKNN(topk=None).fit(_toy_csr())
    assert np.allclose(np.diag(m.sim), 0.0)
