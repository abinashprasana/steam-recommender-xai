"""Calibrated recommendation (Steck, RecSys 2018).

Replaces the ad-hoc framing of fair_recommend / diverse_recommend with the
canonical technique they were informally approximating. The premise: if a user's
history is 70% strategy and 30% racing, their recommendations should be roughly
70/30 too -- not 100% strategy because that is where the model is most confident.

The result this produces is not a single fairness number but a trade-off curve:
accuracy against miscalibration as lambda sweeps from 0 to 1.
"""
import numpy as np


def build_genre_map(games_in_use):
    """{item_idx: [genres]} lookup.

    The greedy re-ranker evaluates ~200 candidates per selected slot, so a pandas
    .loc inside that loop costs hundreds of thousands of lookups per sweep. Built
    once and passed down instead. Accepts an already-built map so callers can hand
    the same dict around.
    """
    if isinstance(games_in_use, dict):
        return games_in_use
    return {idx: (list(g) if isinstance(g, (list, tuple)) else [])
            for idx, g in games_in_use['genres'].items()}


def item_genre_distribution(genres):
    """p(g|i) -- uniform over the genres an item carries."""
    if not genres:
        return {}
    w = 1.0 / len(genres)
    return {g: w for g in genres}


def _normalise(dist):
    """Rescale to sum to 1.

    Normalising by the accumulated mass rather than the item count matters:
    a game with an empty genres list contributes nothing but would still inflate
    the denominator, leaving p and q summing to less than 1. KL between two
    sub-normalised distributions can come out negative, which is meaningless as a
    miscalibration score.
    """
    total = sum(dist.values())
    if total <= 0:
        return {}
    return {g: v / total for g, v in dist.items()}


def user_genre_distribution(item_indices, games_in_use):
    """p(g|u) -- the genre mix of a user's interaction history."""
    gmap = build_genre_map(games_in_use)
    dist = {}
    for idx in item_indices:
        for g, w in item_genre_distribution(gmap.get(idx, [])).items():
            dist[g] = dist.get(g, 0.0) + w
    return _normalise(dist)


def list_genre_distribution(item_indices, games_in_use, weights=None):
    """q(g|I) -- the genre mix of a recommendation list."""
    gmap = build_genre_map(games_in_use)
    dist = {}
    for pos, idx in enumerate(item_indices):
        w = 1.0 if weights is None else float(weights[pos])
        for g, gw in item_genre_distribution(gmap.get(idx, [])).items():
            dist[g] = dist.get(g, 0.0) + gw * w
    return _normalise(dist)


def kl_miscalibration(p, q, alpha=0.01):
    """C(p, q) = KL(p || q~), with q~ = (1-alpha)q + alpha*p.

    The smoothing is Steck's: without it a genre the user likes but the list omits
    sends the divergence to infinity, which makes the objective unusable.
    """
    if not p:
        return 0.0
    total = 0.0
    for g, pg in p.items():
        if pg <= 0:
            continue
        qg = (1 - alpha) * q.get(g, 0.0) + alpha * pg
        if qg <= 0:
            continue
        total += pg * np.log2(pg / qg)
    return float(total)


def calibrated_rerank(scores, user_history, games_in_use, n=10, lam=0.5,
                      candidate_pool=200, exclude=None):
    """Greedily build a list maximising (1-lam)*relevance - lam*miscalibration.

    Steck's argument for post-hoc re-ranking: models trained pointwise or pairwise
    cannot fold calibration into training, so it belongs in a re-ranking step.
    lam=0 reduces to pure relevance; lam=1 optimises calibration alone.
    """
    scores = np.asarray(scores, dtype=np.float64).copy()
    if exclude:
        scores[list(exclude)] = -np.inf

    gmap = build_genre_map(games_in_use)
    p_user = user_genre_distribution(user_history, gmap)
    if not p_user:
        top = np.argsort(-scores)[:n]
        return list(top)

    pool_size = min(candidate_pool, int(np.isfinite(scores).sum()))
    candidates = list(np.argsort(-scores)[:pool_size])

    finite = scores[np.isfinite(scores)]
    lo, hi = (finite.min(), finite.max()) if finite.size else (0.0, 1.0)
    span = (hi - lo) or 1.0
    rel = {c: (scores[c] - lo) / span for c in candidates}

    selected = []
    for _ in range(min(n, len(candidates))):
        best, best_obj = None, -np.inf
        for c in candidates:
            if c in selected:
                continue
            trial = selected + [c]
            q = list_genre_distribution(trial, gmap)
            obj = (1 - lam) * sum(rel[i] for i in trial) - lam * kl_miscalibration(p_user, q)
            if obj > best_obj:
                best, best_obj = c, obj
        if best is None:
            break
        selected.append(best)

    return selected


class CalibratedRecommender:
    """Wraps any scorer so calibrated lists flow through the standard harness.

    Exposes score_all() by returning a score vector whose ordering reproduces the
    calibrated ranking, so accuracy and calibration are measured on the same
    footing as every other model rather than in a separate ad-hoc script.
    """

    def __init__(self, base, games_in_use, train_by_user, n_items, lam=0.5, n=10):
        self.base = base
        self.games_in_use = build_genre_map(games_in_use)
        self.train_by_user = train_by_user
        self.n_items = n_items
        self.lam = lam
        self.n = n

    def score_all(self, user_idx):
        raw = np.asarray(self.base.score_all(user_idx), dtype=np.float64)
        history = self.train_by_user.get(user_idx, [])
        order = calibrated_rerank(raw, history, self.games_in_use,
                                  n=self.n, lam=self.lam, exclude=set(history))
        out = np.full(self.n_items, -np.inf)
        for rank, item in enumerate(order):
            out[item] = float(len(order) - rank)
        return out
