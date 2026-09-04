"""One evaluation harness that every strategy passes through.

The flaw this replaces: train.py printed "Evaluating hybrid recommender" while the
scoring loop called bpr.score_all() directly, so the reported NDCG/MAP described
plain BPR and the hybrid was never measured for ranking accuracy at all. Any model
exposing score_all(user_idx) now goes through the same code path, so the numbers
in the results table always describe the model named beside them.
"""
import numpy as np


def dcg(gains):
    return sum(g / np.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked_items, relevant, k=10):
    """NDCG@k with the ideal ranking defined by how many relevant items exist.

    The previous implementation built IDCG by sorting the retrieved hit vector,
    so a user with 5 relevant items of which the model surfaced 1 was scored
    against an ideal of "1 hit", not "5 hits". That inflates the metric whenever
    recall is incomplete -- which is most of the time.
    """
    if not relevant:
        return 0.0
    hits = [1.0 if r in relevant else 0.0 for r in ranked_items[:k]]
    ideal = [1.0] * min(len(relevant), k)
    idcg = dcg(ideal)
    return dcg(hits) / idcg if idcg > 0 else 0.0


def map_at_k(ranked_items, relevant, k=10):
    if not relevant:
        return 0.0
    hits, total = 0, 0.0
    for i, r in enumerate(ranked_items[:k]):
        if r in relevant:
            hits += 1
            total += hits / (i + 1)
    return total / min(len(relevant), k)


def recall_at_k(ranked_items, relevant, k=10):
    if not relevant:
        return 0.0
    return len(set(ranked_items[:k]) & relevant) / len(relevant)


def hit_rate_at_k(ranked_items, relevant, k=10):
    if not relevant:
        return 0.0
    return 1.0 if set(ranked_items[:k]) & relevant else 0.0


def coverage(all_recommended, n_items):
    """Share of the catalogue that appears in anyone's top-k."""
    return len(all_recommended) / n_items if n_items else 0.0


def gini(counts, n_items):
    """Concentration of recommendations across items. 0 = uniform, 1 = one item.

    Reported alongside accuracy because a model can score well while serving the
    same handful of blockbusters to everybody.
    """
    x = np.zeros(n_items, dtype=np.float64)
    for item, c in counts.items():
        if 0 <= item < n_items:
            x[item] = c
    if x.sum() == 0:
        return 0.0
    x = np.sort(x)
    n = len(x)
    index = np.arange(1, n + 1)
    return float((2 * (index * x).sum()) / (n * x.sum()) - (n + 1) / n)


def select_users(test_relevant, max_users=None, seed=42):
    """Pick the evaluation cohort once, so every model is scored on the same users.

    This matters for more than fairness of the accuracy columns: Coverage counts
    distinct recommended items, so it rises purely with the number of users
    evaluated. Scoring one model on 200 users and another on 1500 makes their
    Coverage figures incomparable even though they sit in the same column.
    """
    uids = sorted(test_relevant.keys())
    if max_users is not None and len(uids) > max_users:
        rng = np.random.RandomState(seed)
        uids = sorted(rng.choice(uids, size=max_users, replace=False).tolist())
    return uids


def evaluate_model(scorer, train_by_user, test_relevant, n_items, k=10,
                   mode='full', n_negatives=99, seed=42, max_users=None,
                   users=None):
    """Evaluate any object exposing score_all(user_idx).

    mode='full'    -- rank the entire catalogue with training items masked. This is
                      the honest protocol and is tractable here (n_items ~ 2.8k).
    mode='sampled' -- rank relevant items against n_negatives sampled negatives.
                      Kept only so the optimism of the original protocol can be
                      quantified rather than assumed.
    """
    rng = np.random.RandomState(seed)
    uids = users if users is not None else select_users(test_relevant, max_users, seed)

    ndcgs, maps, recalls, hits = [], [], [], []
    rec_counts, rec_set = {}, set()

    for uid in uids:
        relevant = test_relevant[uid]
        if not relevant:
            continue
        seen = set(train_by_user.get(uid, []))
        scores = np.asarray(scorer.score_all(uid), dtype=np.float64).copy()

        if mode == 'full':
            scores[list(seen)] = -np.inf
            top = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
            ranked = list(top[np.argsort(-scores[top])])
        else:
            pool_excl = seen | relevant
            available = np.setdiff1d(np.arange(n_items), np.fromiter(pool_excl, int),
                                     assume_unique=False)
            if len(available) < n_negatives:
                continue
            negs = rng.choice(available, size=n_negatives, replace=False)
            pool = np.concatenate([np.fromiter(relevant, int), negs])
            ranked = list(pool[np.argsort(-scores[pool])])

        ndcgs.append(ndcg_at_k(ranked, relevant, k))
        maps.append(map_at_k(ranked, relevant, k))
        recalls.append(recall_at_k(ranked, relevant, k))
        hits.append(hit_rate_at_k(ranked, relevant, k))

        for it in ranked[:k]:
            rec_counts[it] = rec_counts.get(it, 0) + 1
            rec_set.add(it)

    return {
        f'NDCG@{k}': float(np.mean(ndcgs)) if ndcgs else 0.0,
        f'MAP@{k}': float(np.mean(maps)) if maps else 0.0,
        f'Recall@{k}': float(np.mean(recalls)) if recalls else 0.0,
        f'HitRate@{k}': float(np.mean(hits)) if hits else 0.0,
        'Coverage': coverage(rec_set, n_items),
        'Gini': gini(rec_counts, n_items),
        'n_users_evaluated': len(ndcgs),
    }


def mean_ci(values, alpha=0.05):
    """Mean with a 95% confidence interval across seeds.

    Uses the t quantile rather than 1.96. With 3 seeds the correct multiplier is
    4.30, not 1.96 -- reporting the normal quantile on a handful of runs would
    understate the interval by more than half, which is exactly the kind of
    overconfidence this project is meant to avoid.
    """
    a = np.asarray(values, dtype=np.float64)
    if len(a) < 2:
        return (float(a.mean()) if len(a) else 0.0), 0.0
    from scipy import stats
    t = float(stats.t.ppf(1 - alpha / 2, len(a) - 1))
    return float(a.mean()), float(t * a.std(ddof=1) / np.sqrt(len(a)))
