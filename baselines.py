"""Reference models the project was missing.

Without a baseline, NDCG@10 = 0.52 is an uninterpretable number -- it could be
excellent or it could be worse than recommending the same ten popular games to
everybody. Each model here exposes the same `score_all(user_idx)` signature as
BayesianPersonalizedRanking (recommender.py) so they drop into the shared
evaluation harness unchanged.
"""
import numpy as np
from scipy.sparse import csr_matrix


def build_csr(train_df, n_users, n_items):
    """Binary user-item interaction matrix from a training split."""
    return csr_matrix(
        (np.ones(len(train_df), dtype=np.float32),
         (train_df['user_idx'].values, train_df['item_idx'].values)),
        shape=(n_users, n_items)
    )


class PopularityRecommender:
    """Recommends the globally most-interacted items, identically for everyone.

    The single most important baseline: any personalized model that cannot beat
    it is not delivering personalization, whatever its absolute NDCG looks like.
    """

    def __init__(self):
        self.scores = None

    def fit(self, csr):
        self.scores = np.asarray(csr.sum(axis=0)).ravel().astype(np.float32)
        return self

    def score_all(self, user_idx):
        return self.scores.copy()


class ItemKNN:
    """Item-item collaborative filtering by cosine similarity over co-occurrence.

    Distinct from HybridRecommender.sim_matrix (recommender.py), which is cosine
    over the 22 content features. This one uses the interaction matrix, so it is a
    collaborative model rather than a content one -- both belong in the comparison.
    """

    def __init__(self, topk=200):
        self.topk = topk
        self.sim = None
        self.csr = None

    def fit(self, csr):
        self.csr = csr.tocsr()
        X = self.csr.toarray().astype(np.float32)
        norms = np.linalg.norm(X, axis=0)
        norms[norms == 0] = 1e-9
        Xn = X / norms
        sim = Xn.T @ Xn
        np.fill_diagonal(sim, 0.0)

        if self.topk and self.topk < sim.shape[1]:
            # Keep only the topk neighbours per item; the tail is mostly noise and
            # dropping it is standard practice for item-kNN.
            cut = np.partition(sim, -self.topk, axis=1)[:, -self.topk][:, None]
            sim[sim < cut] = 0.0

        self.sim = sim
        return self

    def score_all(self, user_idx):
        return self.csr[user_idx].toarray().ravel() @ self.sim


class EASE:
    """Embarrassingly Shallow Autoencoder (Steck, WWW 2019).

    Closed-form item-item linear model -- no iterative training, no learning rate,
    no epochs. Solves for B in one matrix inverse and routinely outperforms far
    more complex models on sparse implicit-feedback data, which is exactly what
    this dataset is (0.17% density).
    """

    def __init__(self, reg=250.0):
        self.reg = reg
        self.B = None
        self.csr = None

    def fit(self, csr):
        self.csr = csr.tocsr()
        X = self.csr.toarray().astype(np.float64)

        G = X.T @ X
        idx = np.diag_indices(G.shape[0])
        G[idx] += self.reg
        P = np.linalg.inv(G)

        B = -P / np.diag(P)[None, :]
        B[idx] = 0.0

        self.B = B.astype(np.float32)
        return self

    def score_all(self, user_idx):
        return self.csr[user_idx].toarray().ravel() @ self.B
