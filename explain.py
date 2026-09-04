"""Counterfactual explanations of the recommender that actually ranks.

What this replaces: a RandomForest surrogate trained on 22 item features with no
user representation at all, explained with SHAP/LIME. That surrogate could never
answer "why was this recommended to *me*" -- it had no notion of a user, and it
was not the model doing the ranking.

What this does instead: attributes a recommendation to the user's own interaction
history. "You were recommended Hollow Knight because you played Ori, Celeste and
Dead Cells -- without Celeste and Dead Cells, it would not have been recommended."

The key detail is the refit. Removing an interaction and re-scoring with the
user's *original* latent vector would measure nothing, because that vector still
encodes the removed item. Every counterfactual here re-fits the user's factor with
the item factors held fixed, using the same BPR update the model was trained with.
Without that, the explanation is not faithful and the whole exercise is theatre.
"""
import numpy as np


def refit_user_factor(item_factors, pos_items, n_items, factors=None,
                      lr=0.05, reg=0.01, steps=400, seed=42):
    """Re-learn one user's latent vector from a given positive set.

    Item factors are frozen, so this is a small convex-ish problem that converges
    in milliseconds -- which is what makes leave-one-out over a user's whole
    history practical.
    """
    rng = np.random.RandomState(seed)
    pos = list(pos_items)
    if not pos:
        return np.zeros(factors or item_factors.shape[1], dtype=np.float32)

    d = factors or item_factors.shape[1]
    u = (rng.randn(d) * 0.01).astype(np.float32)
    pos_set = set(pos)

    for _ in range(steps):
        i = pos[rng.randint(len(pos))]
        j = rng.randint(n_items)
        while j in pos_set:
            j = rng.randint(n_items)
        vi, vj = item_factors[i], item_factors[j]
        x = float(u @ vi - u @ vj)
        g = 1.0 - 1.0 / (1.0 + np.exp(-x))
        u = u + lr * (g * (vi - vj) - reg * u)

    return u


def _rank_of(scores, target, exclude):
    """0-based rank of target once the user's known items are masked out."""
    s = np.asarray(scores, dtype=np.float64).copy()
    if exclude:
        s[list(exclude)] = -np.inf
    return int((s > s[target]).sum())


class CounterfactualExplainer:
    """Leave-one-out and ACCENT-style explanations for a BPR-style model."""

    def __init__(self, item_factors, n_items, train_by_user, games_in_use,
                 lr=0.05, reg=0.01, steps=400, seed=42):
        self.item_factors = np.asarray(item_factors)
        self.n_items = n_items
        self.train_by_user = train_by_user
        self.games_in_use = games_in_use
        self.lr, self.reg, self.steps, self.seed = lr, reg, steps, seed

    def _score(self, pos_items):
        u = refit_user_factor(self.item_factors, pos_items, self.n_items,
                              lr=self.lr, reg=self.reg, steps=self.steps, seed=self.seed)
        return u @ self.item_factors.T

    def name_of(self, item_idx):
        if item_idx in self.games_in_use.index:
            row = self.games_in_use.loc[item_idx]
            return row.get('app_name') or row.get('title') or f'item {item_idx}'
        return f'item {item_idx}'

    def leave_one_out(self, user_idx, target_item=None, top_n=5):
        """Rank the user's history by how much each item props up the target.

        Both the baseline and every counterfactual are refit, so the comparison is
        like-for-like rather than trained-vector vs refit-vector.
        """
        history = list(self.train_by_user.get(user_idx, []))
        if len(history) < 2:
            return None

        base_scores = self._score(history)
        if target_item is None:
            masked = base_scores.copy()
            masked[history] = -np.inf
            target_item = int(np.argmax(masked))

        base_rank = _rank_of(base_scores, target_item, history)

        influences = []
        for held in history:
            reduced = [h for h in history if h != held]
            if not reduced:
                continue
            scores = self._score(reduced)
            rank = _rank_of(scores, target_item, reduced)
            influences.append({
                'item_idx': held,
                'name': self.name_of(held),
                'rank_delta': rank - base_rank,
                'score_delta': float(base_scores[target_item] - scores[target_item]),
            })

        influences.sort(key=lambda d: (d['rank_delta'], d['score_delta']), reverse=True)
        return {
            'user_idx': user_idx,
            'target_item': target_item,
            'target_name': self.name_of(target_item),
            'base_rank': base_rank,
            'influences': influences[:top_n],
            'all_influences': influences,
        }

    def accent_minimal_set(self, user_idx, target_item=None, k=10, max_set=5):
        """Smallest set of the user's own actions whose removal drops the target out of top-k.

        Greedy expansion over the leave-one-out influence ranking, in the spirit of
        ACCENT (Tran et al., SIGIR 2021), which extends influence-based explanation
        from a single item to a set.
        """
        loo = self.leave_one_out(user_idx, target_item)
        if loo is None:
            return None

        target = loo['target_item']
        history = list(self.train_by_user.get(user_idx, []))
        ordered = [d['item_idx'] for d in loo['all_influences']]

        # Never propose removing a user's entire history: there would be nothing
        # left to refit, so the counterfactual could not be verified at all.
        max_set = max(1, min(max_set, len(history) - 1))

        removed = []
        for cand in ordered[:max_set]:
            removed.append(cand)
            reduced = [h for h in history if h not in removed]
            if not reduced:
                break
            scores = self._score(reduced)
            if _rank_of(scores, target, reduced) >= k:
                return {
                    'user_idx': user_idx,
                    'target_item': target,
                    'target_name': self.name_of(target),
                    'removed': removed,
                    'removed_names': [self.name_of(i) for i in removed],
                    'found': True,
                    'k': k,
                }

        return {
            'user_idx': user_idx,
            'target_item': target,
            'target_name': self.name_of(target),
            'removed': removed,
            'removed_names': [self.name_of(i) for i in removed],
            'found': False,
            'k': k,
        }

    def verify(self, user_idx, target_item, removed, k=10, trials=3):
        """Does the counterfactual claim actually hold when carried out?

        Re-fits across several seeds because the refit is stochastic; a claim that
        only survives one lucky seed is not a claim worth publishing.
        """
        history = list(self.train_by_user.get(user_idx, []))
        reduced = [h for h in history if h not in set(removed)]
        if not reduced:
            return None

        held = 0
        for t in range(trials):
            u = refit_user_factor(self.item_factors, reduced, self.n_items,
                                  lr=self.lr, reg=self.reg, steps=self.steps,
                                  seed=self.seed + t)
            if _rank_of(u @ self.item_factors.T, target_item, reduced) >= k:
                held += 1
        return held / trials


def render_explanation(loo, max_reasons=3):
    """Plain-language rendering. Deterministic and testable -- no LLM involved."""
    if not loo:
        return "Not enough history to explain this recommendation."

    supporters = [d for d in loo['influences'] if d['rank_delta'] > 0][:max_reasons]
    if not supporters:
        return (f"{loo['target_name']} is recommended by the overall model, but no "
                f"single item in your history is individually responsible for it.")

    names = [d['name'] for d in supporters]
    if len(names) == 1:
        because = names[0]
    else:
        because = ', '.join(names[:-1]) + f' and {names[-1]}'
    return f"{loo['target_name']} was recommended because you played {because}."


def faithfulness_report(explainer, user_idxs, k=10, max_set=3):
    """Share of counterfactual claims that hold when actually carried out.

    This is the number that answers "how do you know your explanation reflects the
    model?" -- the question most explainability write-ups cannot answer at all.
    """
    rates, records = [], []
    for uid in user_idxs:
        acc = explainer.accent_minimal_set(uid, k=k, max_set=max_set)
        if not acc or not acc['removed']:
            continue
        rate = explainer.verify(uid, acc['target_item'], acc['removed'], k=k)
        if rate is None:
            continue
        rates.append(rate)
        records.append({
            'user_idx': uid,
            'target': acc['target_name'],
            'removed': acc['removed_names'],
            'displaced': acc['found'],
            'verified_rate': rate,
        })

    # Reported with an interval because this estimate moves a lot on small
    # samples: the same method scored 52% on one set of 25 users and 64% on
    # another. A bare percentage over a couple of dozen explanations would imply
    # far more precision than the measurement supports.
    from evaluation import mean_ci
    mean, ci = mean_ci(rates) if rates else (0.0, 0.0)
    return {
        'n_explanations_checked': len(rates),
        'mean_faithfulness': mean,
        'faithfulness_ci': ci,
        'records': records,
    }
