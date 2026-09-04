import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

class BayesianPersonalizedRanking:
    def __init__(self, factors=64, learning_rate=0.01,
                 regularization=0.01, iterations=50, random_state=42):
        self.random_state = random_state
        self.factors = factors
        self.lr = learning_rate
        self.reg = regularization
        self.iterations = iterations
        self.user_factors = None
        self.item_factors = None

    def fit(self, csr):
        rng = np.random.RandomState(self.random_state)
        n_users, n_items = csr.shape

        self.user_factors = (rng.randn(n_users, self.factors) * 0.01).astype(np.float32)
        self.item_factors = (rng.randn(n_items, self.factors) * 0.01).astype(np.float32)

        user_pos = [set(csr.indices[csr.indptr[u]:csr.indptr[u+1]])
                    for u in range(n_users)]
        active = [u for u in range(n_users) if user_pos[u]]

        for _ in tqdm(range(self.iterations), desc="BPR"):
            sampled = rng.choice(active, size=len(active) * 10)
            for u in sampled:
                pos = user_pos[u]
                i = rng.choice(list(pos))
                j = rng.randint(0, n_items)
                while j in pos:
                    j = rng.randint(0, n_items)

                uf, ii, ij = self.user_factors[u], self.item_factors[i], self.item_factors[j]
                x = float(uf @ ii - uf @ ij)
                g = 1.0 - 1.0 / (1.0 + np.exp(-x))

                uf0, ii0, ij0 = uf.copy(), ii.copy(), ij.copy()
                self.user_factors[u] += self.lr * (g * (ii0 - ij0) - self.reg * uf0)
                self.item_factors[i] += self.lr * (g * uf0 - self.reg * ii0)
                self.item_factors[j] += self.lr * (-g * uf0 - self.reg * ij0)
        return self

    def score_all(self, user_idx):
        return self.user_factors[user_idx] @ self.item_factors.T

def cold_start_recommend(games_in_use, preferred_genres, n=10):
    """Genre-filtered, sentiment-ranked picks for a user with no history.

    Module-level so the web app can serve the cold-start path without building a
    full HybridRecommender (which needs a trained model and a training split).
    This code path existed for a long time but was never called by anything.
    """
    if not preferred_genres:
        return None
    mask = games_in_use['genres'].apply(
        lambda g: any(genre in g for genre in preferred_genres)
    )
    candidates = games_in_use[mask].copy()
    if candidates.empty:
        return None
    return (candidates.sort_values('sentiment_score', ascending=False)
            .head(n)[['app_name', 'genres', 'sentiment', 'price']])


def norm(v):
    out = np.zeros(len(v), dtype=np.float32)
    mask = np.isfinite(v)
    if mask.any():
        lo = v[mask].min()
        hi = v[mask].max()
        out[mask] = (v[mask] - lo) / (hi - lo + 1e-9)
    return out

class HybridRecommender:
    def __init__(self, bpr_model, item_feat_df, train_df, games_in_use, n_users, n_items,
                 alpha=0.7):
        self.bpr = bpr_model
        self.alpha = alpha
        self.item_feat_df = item_feat_df
        self.train_by_user = train_df.groupby('user_idx')['item_idx'].apply(list).to_dict()
        self.games_in_use = games_in_use
        self.n_users = n_users
        self.n_items = n_items
        
        feat_scaled = StandardScaler().fit_transform(item_feat_df.values.astype(np.float32))
        self.sim_matrix = cosine_similarity(feat_scaled)

    def content_recommend(self, seed_idx, n=10, exclude=None):
        if seed_idx >= self.sim_matrix.shape[0]:
            return []
        sims = self.sim_matrix[seed_idx].copy()
        sims[seed_idx] = -1
        if exclude:
            for idx in exclude:
                if idx < len(sims):
                    sims[idx] = -1
        return list(np.argsort(sims)[::-1][:n])

    def content_scores(self, user_idx, n=10):
        """Content-only score vector, seeded from the user's history.

        The seed is the user's highest-indexed item rather than an arbitrary one
        pulled out of a set: `list(seen)[-1]` depended on set hash ordering, so
        the content half of the blend rested on an effectively random choice and
        was not reproducible across Python versions.
        """
        seen = set(self.train_by_user.get(user_idx, []))
        cb_sc = np.zeros(self.n_items)
        if seen:
            seed = max(seen)
            for rank, idx in enumerate(self.content_recommend(seed, n=n * 2, exclude=list(seen))):
                cb_sc[idx] = 1.0 / (rank + 1)
        return cb_sc

    def combined_scores(self, user_idx, n=10, alpha=None):
        """The blended score vector that hybrid_recommend ranks.

        Exposed separately so the hybrid can be evaluated through the same harness
        as every other model. Previously only the ranked list was reachable, which
        is why the reported metrics silently described plain BPR instead.
        """
        alpha = self.alpha if alpha is None else alpha
        seen = set(self.train_by_user.get(user_idx, []))

        bpr_sc = self.bpr.score_all(user_idx).copy()
        for s in seen:
            if s < len(bpr_sc):
                bpr_sc[s] = -np.inf

        cb_sc = self.content_scores(user_idx, n=n)
        combined = alpha * norm(bpr_sc) + (1 - alpha) * norm(cb_sc)
        for s in seen:
            if s < len(combined):
                combined[s] = -np.inf
        return combined

    def score_all(self, user_idx):
        """Harness-compatible interface, matching BayesianPersonalizedRanking."""
        return self.combined_scores(user_idx)

    def hybrid_recommend(self, user_idx, n=10, alpha=0.7):
        combined = self.combined_scores(user_idx, n=n, alpha=alpha)
        return list(np.argsort(combined)[::-1][:n])

    def cold_start(self, preferred_genres, n=10):
        return cold_start_recommend(self.games_in_use, preferred_genres, n)

    def fair_recommend(self, user_idx, item_pop, max_pop, n=10, alpha=0.7, beta=0.8):
        seen = set(self.train_by_user.get(user_idx, []))
        rec_score = self.combined_scores(user_idx, n=n, alpha=alpha)
        inv_pop = np.array([1.0 - item_pop.get(i, 0) / max_pop for i in range(self.n_items)])
        fair_sc = beta * rec_score + (1 - beta) * inv_pop

        for s in seen:
            if s < len(fair_sc):
                fair_sc[s] = -np.inf
        return list(np.argsort(fair_sc)[::-1][:n])

    def diverse_recommend(self, user_idx, n=10, alpha=0.7, max_per_genre=3):
        combined = self.combined_scores(user_idx, n=n, alpha=alpha)
        pool = np.argsort(combined)[::-1][:n * 5]
        selected = []
        gcounts = {}

        for idx in pool:
            if len(selected) >= n:
                break
            genres = self.games_in_use.loc[idx, 'genres'] if idx in self.games_in_use.index else []
            if not any(gcounts.get(g, 0) >= max_per_genre for g in genres):
                selected.append(idx)
                for g in genres:
                    gcounts[g] = gcounts.get(g, 0) + 1
        return selected
