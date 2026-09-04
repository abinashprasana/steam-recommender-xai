"""Save and reload everything needed to score a user without retraining.

Until this existed the BPR factors were discarded when train.py exited, so the
Flask app could never produce a recommendation for anyone -- it could only render
plots generated during training. Live inference, counterfactual explanation, and
the interactive routes all depend on this module.
"""
import os
import json
import joblib

ARTIFACT_DIR = 'artifacts'
ARTIFACT_PATH = os.path.join(ARTIFACT_DIR, 'model.joblib')

def save_artifacts(bpr, bundle, path=ARTIFACT_PATH):
    """Persist the trained model plus the lookups needed to interpret its output."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    payload = {
        'user_factors': bpr.user_factors,
        'item_factors': bpr.item_factors,
        'factors': bpr.factors,
        'games_in_use': bundle.games_in_use,
        'item_feat_df': bundle.item_feat_df,
        'feature_cols': bundle.feature_cols,
        'n_users': bundle.n_users,
        'n_items': bundle.n_items,
        'user_enc': bundle.user_enc,
        'item_enc': bundle.item_enc,
        'train_by_user': bundle.train_df.groupby('user_idx')['item_idx'].apply(list).to_dict(),
    }
    joblib.dump(payload, path, compress=3)
    return path

def load_artifacts(path=ARTIFACT_PATH):
    """Load a saved model. Raises FileNotFoundError if train.py has not been run."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No artifacts at {path}. Run 'python train.py' first."
        )
    return joblib.load(path)

def artifacts_exist(path=ARTIFACT_PATH):
    return os.path.exists(path)


PROJECTION_PATH = os.path.join('static', 'embedding.json')


def dump_embedding_projection(item_factors, games_in_use, path=PROJECTION_PATH,
                              n_components=3):
    """Project the learned item factors to 3D for the site's hero.

    The hero renders the real catalogue rather than generic particles: points
    that sit near each other are games the model considers similar, so the
    object on screen is the model. Three components rather than two because the
    hero orbits, and structure that overlaps in a flat projection separates once
    you can turn it.
    """
    import numpy as np
    from sklearn.decomposition import PCA

    F = np.asarray(item_factors, dtype=np.float64)
    xyz = PCA(n_components=n_components, random_state=42).fit_transform(F)

    # PCA piles most items near the centroid, which renders as an opaque blob
    # rather than a field. Spreading radially (r -> r**0.55 about the centre)
    # opens the middle out while preserving each point's direction and its
    # distance ordering, so neighbours on screen are still neighbours in factor
    # space. Done on the full n-sphere so it holds in 3D too.
    xyz = xyz - xyz.mean(axis=0)
    r = np.linalg.norm(xyz, axis=1, keepdims=True)
    unit = xyz / np.maximum(r, 1e-9)
    r_scaled = (r / (np.percentile(r, 99) + 1e-9)) ** 0.55
    xyz = unit * r_scaled
    xyz = np.clip(xyz / (np.percentile(np.abs(xyz), 99) + 1e-9), -1.15, 1.15)

    points = []
    for idx in range(F.shape[0]):
        name = ''
        if idx in games_in_use.index:
            row = games_in_use.loc[idx]
            name = row.get('app_name') or row.get('title') or ''
        points.append([*(round(float(v), 4) for v in xyz[idx]), name])

    edges, phases = _similarity_graph(F, xyz)

    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'dims': n_components, 'points': points,
                   'edges': edges, 'phases': phases},
                  f, ensure_ascii=False, separators=(',', ':'))
    return path


def _similarity_graph(F, xyz, k=4, max_len=0.5):
    """The model's own nearest-neighbour graph, filtered so it can be drawn.

    Neighbours are found in the full 64-dimension factor space, so every edge is
    a real statement about what the recommender considers similar. Measured
    against a projected cloud diameter of ~2.9, though, raw 64-dim neighbours
    have a median projected length of 0.357 and a maximum of 1.496 -- half of
    them would cross the cloud and render as noise. Edges the projection could
    not keep close are therefore dropped rather than drawn misleadingly.

    Returns a flat index array and a per-edge phase, so pulses sweep outward
    through the graph instead of every edge flashing at once.
    """
    import numpy as np
    from collections import deque
    from sklearn.neighbors import NearestNeighbors

    n = F.shape[0]
    nn = NearestNeighbors(n_neighbors=min(k + 1, n)).fit(F)
    _, idx = nn.kneighbors(F)

    pairs = set()
    for i, row in enumerate(idx):
        for j in row[1:]:                      # row[0] is the point itself
            a, b = (i, int(j)) if i < j else (int(j), i)
            if np.linalg.norm(xyz[a] - xyz[b]) <= max_len:
                pairs.add((a, b))
    pairs = sorted(pairs)

    # Breadth-first depth from a few spread-out seeds. Phase comes from depth, so
    # light appears to propagate through the network rather than blink.
    adj = [[] for _ in range(n)]
    for a, b in pairs:
        adj[a].append(b)
        adj[b].append(a)

    depth = [-1] * n
    seeds = [int(np.argmin(xyz[:, 0])), int(np.argmax(xyz[:, 0])), int(np.argmax(xyz[:, 1]))]
    q = deque()
    for s in seeds:
        if depth[s] == -1:
            depth[s] = 0
            q.append(s)
    while q:
        u = q.popleft()
        for v in adj[u]:
            if depth[v] == -1:
                depth[v] = depth[u] + 1
                q.append(v)

    reached = [d for d in depth if d >= 0]
    span = max(reached) if reached else 1
    flat, phase = [], []
    for a, b in pairs:
        flat.extend((a, b))
        da, db = depth[a], depth[b]
        d = min(x for x in (da, db) if x >= 0) if max(da, db) >= 0 else 0
        # Normalised across the graph's actual depth, times three so several
        # wavefronts travel at once. A fixed modulus left the phases bunched in
        # the first 40% of the cycle, because the graph is only ~10 layers deep.
        phase.append(round(((d / max(span, 1)) * 3.0) % 1.0, 3))

    return flat, phase
