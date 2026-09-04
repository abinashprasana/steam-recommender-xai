"""BPR in PyTorch -- batched, and differentiable.

Two reasons this exists. First speed: the numpy implementation in recommender.py
walks one (user, positive, negative) triplet at a time in pure Python, which is
the dominant cost of the whole pipeline. Batching it collapses that to seconds.

Second, and more importantly: differentiability. Gradient-based explanation
methods -- influence functions, and LXR (Barkan et al., WWW 2024) -- need to
backpropagate through the recommender. The numpy version cannot support them.

recommender.py is kept as the reference implementation. This port is only
trustworthy insofar as it agrees with it, so parity is checked rather than
assumed (see tests/test_torch_parity.py).
"""
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


class BPRTorch(nn.Module):
    """Matrix factorization trained with the pairwise BPR objective.

    Exposes score_all(user_idx) so it drops into the same evaluation harness,
    explainer, and re-rankers as every other model in the project.
    """

    # Same 50 epochs as the NumPy class, so the two are directly comparable. The
    # batch size does matter: at 8192 this took only ~400 optimiser steps across
    # the whole run and scored at chance. 1024 restores enough updates to reach
    # parity in the same number of epochs. Parity against NumPy is what tuned it.
    def __init__(self, n_users, n_items, factors=64, learning_rate=0.01,
                 regularization=0.01, iterations=50, batch_size=1024,
                 random_state=42, device='cpu'):
        super().__init__()
        self.n_users, self.n_items, self.factors = n_users, n_items, factors
        self.lr, self.reg, self.iterations = learning_rate, regularization, iterations
        self.batch_size, self.random_state = batch_size, random_state
        self.device = torch.device(device)

        torch.manual_seed(random_state)
        self.user_emb = nn.Embedding(n_users, factors)
        self.item_emb = nn.Embedding(n_items, factors)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        self.to(self.device)

    def forward(self, u, i, j):
        ue, ie, je = self.user_emb(u), self.item_emb(i), self.item_emb(j)
        return (ue * ie).sum(1) - (ue * je).sum(1), (ue, ie, je)

    def _sample(self, rng, indptr, indices, active, pos_codes, n):
        """Vectorised triplet sampling.

        Both the positive draw and the negative rejection are done with array
        operations. A Python loop here would cost more than the batching saves --
        the first version of this method resampled negatives one at a time and was
        slower than the NumPy implementation it was meant to replace.
        """
        users = rng.choice(active, size=n)
        starts = indptr[users]
        lengths = indptr[users + 1] - starts
        pos = indices[starts + (rng.random_sample(n) * lengths).astype(np.int64)]

        neg = rng.randint(0, self.n_items, size=n)
        # Density is ~0.2%, so collisions are rare and a few vectorised passes
        # clear them; any survivors are left rather than looped over forever.
        for _ in range(4):
            clash = np.isin(users.astype(np.int64) * self.n_items + neg, pos_codes)
            if not clash.any():
                break
            neg[clash] = rng.randint(0, self.n_items, size=int(clash.sum()))

        return users, pos.astype(np.int64), neg.astype(np.int64)

    def fit(self, csr, verbose=True):
        rng = np.random.RandomState(self.random_state)
        csr = csr.tocsr()
        indptr, indices = csr.indptr, csr.indices

        active = np.array([u for u in range(self.n_users)
                           if indptr[u + 1] > indptr[u]], dtype=np.int64)
        if len(active) == 0:
            return self

        # Sorted codes for O(log n) membership testing during negative rejection.
        rows = np.repeat(np.arange(self.n_users, dtype=np.int64), np.diff(indptr))
        pos_codes = np.sort(rows * self.n_items + indices.astype(np.int64))

        n_samples = len(active) * 10
        # No weight_decay here. nn.Embedding produces DENSE gradients, so the
        # optimiser's built-in decay would regularise every row on every step,
        # including the ~99.9% not in the batch -- Adam normalises that decay term
        # and walks untouched rows toward zero at roughly lr per step, which wiped
        # out the table and left the model scoring at chance. The NumPy reference
        # penalises only the sampled u/i/j, so the L2 is applied per batch below.
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)

        loop = tqdm(range(self.iterations), desc="BPR(torch)") if verbose else range(self.iterations)
        for _ in loop:
            users, pos, neg = self._sample(rng, indptr, indices, active, pos_codes, n_samples)
            perm = rng.permutation(n_samples)
            users, pos, neg = users[perm], pos[perm], neg[perm]

            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                u = torch.as_tensor(users[start:end], dtype=torch.long, device=self.device)
                i = torch.as_tensor(pos[start:end], dtype=torch.long, device=self.device)
                j = torch.as_tensor(neg[start:end], dtype=torch.long, device=self.device)

                opt.zero_grad()
                diff, (ue, ie, je) = self(u, i, j)
                l2 = (ue.pow(2).sum() + ie.pow(2).sum() + je.pow(2).sum()) / u.shape[0]
                loss = -nn.functional.logsigmoid(diff).mean() + self.reg * l2
                loss.backward()
                opt.step()

        return self

    @property
    def user_factors(self):
        return self.user_emb.weight.detach().cpu().numpy()

    @property
    def item_factors(self):
        return self.item_emb.weight.detach().cpu().numpy()

    def score_all(self, user_idx):
        with torch.no_grad():
            u = torch.as_tensor([user_idx], dtype=torch.long, device=self.device)
            return (self.user_emb(u) @ self.item_emb.weight.T).squeeze(0).cpu().numpy()
