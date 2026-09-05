"""Flask backend: JSON API plus static host for the built front end.

Previously this rendered Jinja templates around stored PNGs. The UI now lives in
frontend/ as a Vite bundle; this file's job is to expose the model over JSON and
serve that bundle. Live inference and counterfactual explanation stay server-side
because they need the trained factors -- everything here reuses the existing
modules rather than reimplementing scoring.
"""
import os
import io
import gzip
import json
import hashlib
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

from persistence import load_artifacts, artifacts_exist
from recommender import cold_start_recommend
from explain import CounterfactualExplainer, render_explanation

DIST_DIR = os.path.join('frontend', 'dist')

# Stars of celestial navigation: the fixed points you take a bearing from, which
# is the same idea the product is named for. The dataset ships anonymised numeric
# ids like 111222333444555666888; showing those asks a reader to pick between
# meaningless digit strings. Each id is mapped to a star deterministically, so the
# same player is always the same name without inventing or storing anything about
# them. There are 55 here, and the collision walk below depends on that count
# being whatever len(NAV_STARS) actually is rather than on a number in a comment.
NAV_STARS = [
    'Acamar', 'Achernar', 'Acrux', 'Adhara', 'Aldebaran', 'Alioth', 'Alkaid',
    'Alnilam', 'Alphard', 'Alphecca', 'Alpheratz', 'Altair', 'Ankaa', 'Antares',
    'Arcturus', 'Atria', 'Avior', 'Bellatrix', 'Betelgeuse', 'Canopus', 'Capella',
    'Deneb', 'Denebola', 'Diphda', 'Dubhe', 'Elnath', 'Eltanin', 'Enif',
    'Fomalhaut', 'Gacrux', 'Gienah', 'Hadar', 'Hamal', 'Kochab', 'Markab',
    'Menkar', 'Menkent', 'Miaplacidus', 'Mirfak', 'Nunki', 'Peacock', 'Polaris',
    'Pollux', 'Procyon', 'Rasalhague', 'Regulus', 'Rigel', 'Sabik', 'Schedar',
    'Shaula', 'Sirius', 'Spica', 'Suhail', 'Vega', 'Zubenelgenubi',
]


def star_name(user_id: str) -> str:
    """Stable pseudonym for a player id. Same id always yields the same star."""
    h = hashlib.sha1(str(user_id).encode('utf-8')).hexdigest()
    return NAV_STARS[int(h[:8], 16) % len(NAV_STARS)]

app = Flask(__name__, static_folder=None)

_state = {'loaded': False}

# Origins allowed to call the API cross-origin, e.g. a frontend built with
# VITE_API_BASE pointed at this host and deployed separately (Vercel serving
# the static bundle, this process serving /api elsewhere). Empty by default:
# when Flask serves both the API and the bundle from one origin, as it does
# locally and as api.ts assumes with no base set, no CORS header is needed and
# none is sent. Set to a comma-separated list, or '*' for local testing only.
_ALLOWED_ORIGINS = [o.strip() for o in os.environ.get('ALLOWED_ORIGIN', '').split(',') if o.strip()]


class LoadedRecommender:
    """Scores users from persisted factors, with no training data in memory."""

    def __init__(self, payload):
        self.user_factors = np.asarray(payload['user_factors'])
        self.item_factors = np.asarray(payload['item_factors'])
        self.n_items = int(payload['n_items'])

    def score_all(self, user_idx):
        return self.user_factors[user_idx] @ self.item_factors.T


def get_state():
    """Load artifacts once, lazily. Missing artifacts are reported, not fatal."""
    if _state['loaded']:
        return _state
    _state['loaded'] = True
    if not artifacts_exist():
        _state['error'] = "No trained model found. Run 'python train.py' first."
        return _state

    payload = load_artifacts()
    _state['model'] = LoadedRecommender(payload)
    _state['games'] = payload['games_in_use']
    _state['train_by_user'] = payload['train_by_user']
    _state['user_enc'] = payload['user_enc']
    _state['n_items'] = int(payload['n_items'])
    _state['explainer'] = CounterfactualExplainer(
        payload['item_factors'], int(payload['n_items']),
        payload['train_by_user'], payload['games_in_use'])
    return _state


def game_name(games, idx):
    if idx in games.index:
        row = games.loc[idx]
        return row.get('app_name') or row.get('title') or f'item {idx}'
    return f'item {idx}'


def game_genres(games, idx):
    if idx in games.index:
        g = games.loc[idx, 'genres']
        return list(g) if isinstance(g, (list, tuple)) else []
    return []


# ----------------------------------------------------------------- API

@app.get('/api/results')
def api_results():
    """The evaluation report written by train.py."""
    if not os.path.exists('results.json'):
        return jsonify({'error': 'results.json missing. Run python train.py.'}), 404
    with open('results.json', encoding='utf-8') as f:
        return jsonify(json.load(f))


@app.get('/api/users')
def api_users():
    """Sample players, described by what they actually play.

    The raw Steam ids are anonymised digit strings like 111222333444555666888.
    Asking someone to pick one of those is asking them to choose at random. Each
    player is labelled here by the genres that dominate their own library, which
    is information we already hold and which makes the choice mean something.
    The real id is still returned, so direct lookup keeps working.
    """
    st = get_state()
    if st.get('error'):
        return jsonify({'error': st['error']}), 503

    tbu, enc, games = st['train_by_user'], st['user_enc'], st['games']

    # Catalogue base rate per genre, computed once. Picking a player's *most
    # frequent* genre just describes the catalogue: Action sits on most of it, so
    # every player came out as "Action, Adventure". Ranking by lift over the base
    # rate surfaces what is actually distinctive about each library instead.
    base = st.get('genre_base')
    if base is None:
        base = {}
        total = 0
        for gs in games['genres']:
            if not isinstance(gs, (list, tuple)):
                continue
            total += 1
            for g in gs:
                base[g] = base.get(g, 0) + 1
        base = {g: c / max(total, 1) for g, c in base.items()}
        st['genre_base'] = base

    picks = [u for u in list(tbu.keys())[:600] if len(tbu[u]) >= 6][:12]
    out = []
    used = set()
    for u in picks:
        counts, seen = {}, 0
        for idx in tbu[u]:
            if idx not in games.index:
                continue
            genres = games.loc[idx, 'genres']
            if not isinstance(genres, (list, tuple)):
                continue
            seen += 1
            for g in genres:
                counts[g] = counts.get(g, 0) + 1

        # Lift, with a floor on the count so a single oddity does not define a
        # player, and a floor on the base rate so rare genres do not run away.
        scored = [
            (g, (c / max(seen, 1)) / max(base.get(g, 0.01), 0.02))
            for g, c in counts.items() if c >= 2
        ] or [(g, c) for g, c in counts.items()]
        scored.sort(key=lambda kv: -kv[1])

        raw_id = str(enc.inverse_transform([u])[0])
        # Collisions are possible across 55 stars, so step to the next free one
        # rather than showing the same name twice in one list.
        name = star_name(raw_id)
        if name in used:
            start = NAV_STARS.index(name)
            for k in range(1, len(NAV_STARS)):
                alt = NAV_STARS[(start + k) % len(NAV_STARS)]
                if alt not in used:
                    name = alt
                    break
        used.add(name)

        out.append({
            'id': raw_id,
            'display': name,
            'taste': ' · '.join(g for g, _ in scored[:2]),
            'history_size': len(tbu[u]),
        })

    return jsonify({'users': out})


@app.get('/api/recommend/<user_id>')
def api_recommend(user_id):
    """Top-N for a real user, each with a counterfactual explanation.

    The explanation names which of the user's own games hold the recommendation
    up, verified by re-fitting their latent vector without those games.
    """
    st = get_state()
    if st.get('error'):
        return jsonify({'error': st['error']}), 503

    games, model, tbu, enc = st['games'], st['model'], st['train_by_user'], st['user_enc']
    try:
        user_idx = int(enc.transform([user_id])[0])
    except (ValueError, KeyError):
        return jsonify({'error': f"User '{user_id}' is not in the training data."}), 404

    n = max(1, min(int(request.args.get('n', 10)), 20))
    seen = set(tbu.get(user_idx, []))
    scores = model.score_all(user_idx).copy()
    if seen:
        scores[list(seen)] = -np.inf
    top = [int(i) for i in np.argsort(-scores)[:n]]

    explainer = st['explainer']
    out = []
    for item in top:
        loo = explainer.leave_one_out(user_idx, target_item=item, top_n=3)
        influences = []
        if loo:
            for d in loo['influences']:
                if d['rank_delta'] > 0:
                    influences.append({'name': d['name'],
                                       'rank_delta': int(d['rank_delta']),
                                       'score_delta': float(d['score_delta'])})
        out.append({
            'item_idx': item,
            'name': game_name(games, item),
            'genres': game_genres(games, item),
            'explanation': render_explanation(loo) if loo else
                           'Not enough history to explain this recommendation.',
            'influences': influences,
        })

    return jsonify({
        'user_id': user_id,
        'display': star_name(user_id),
        'history': [{'idx': int(i), 'name': game_name(games, i)} for i in sorted(seen)],
        'recommendations': out,
    })


@app.get('/api/cold-start')
def api_cold_start():
    """Genre-filtered, sentiment-ranked picks for a user with no history."""
    st = get_state()
    if st.get('error'):
        return jsonify({'error': st['error']}), 503

    games_df = st['games']
    all_genres = sorted({g for gs in games_df['genres']
                         if isinstance(gs, (list, tuple)) for g in gs})
    selected = request.args.getlist('genre')
    if not selected:
        return jsonify({'available_genres': all_genres, 'games': []})

    top = cold_start_recommend(games_df, selected, n=12)
    if top is None or top.empty:
        return jsonify({'available_genres': all_genres, 'games': [],
                        'message': 'No games matched those genres.'})

    return jsonify({
        'available_genres': all_genres,
        'selected': selected,
        'games': [{'name': r['app_name'], 'genres': list(r['genres']),
                   'sentiment': r['sentiment'], 'price': r['price']}
                  for _, r in top.iterrows()],
    })


# ------------------------------------------------- static assets + SPA

@app.after_request
def cors(resp):
    """Allow a separately hosted frontend to call this API.

    A no-op unless ALLOWED_ORIGIN is set, so single-origin deployments -- the
    default, Flask serving its own bundle -- are unaffected.
    """
    if not _ALLOWED_ORIGINS:
        return resp
    origin = request.headers.get('Origin', '')
    if origin and (origin in _ALLOWED_ORIGINS or '*' in _ALLOWED_ORIGINS):
        resp.headers['Access-Control-Allow-Origin'] = origin
        resp.headers.add('Vary', 'Origin')
    return resp


@app.after_request
def compress(resp):
    """Gzip text responses.

    The embedding file is 221.8KB of JSON and Flask ships it uncompressed by
    default; gzipped it is roughly 78KB. Measured rather than assumed, and it is
    the single largest asset on the page.
    """
    accept = request.headers.get('Accept-Encoding', '')
    ctype = (resp.mimetype or '')
    if ('gzip' not in accept.lower()
            or resp.status_code < 200 or resp.status_code >= 300
            or 'Content-Encoding' in resp.headers
            or not (ctype.startswith('text/') or ctype in
                    ('application/json', 'application/javascript', 'image/svg+xml'))):
        return resp

    # send_from_directory streams with direct_passthrough set, and get_data()
    # raises on those. Static files are exactly what needs compressing here, so
    # take them off the file wrapper rather than skipping them.
    if resp.direct_passthrough:
        resp.direct_passthrough = False

    data = resp.get_data()
    if len(data) < 1024:            # below this the header costs more than it saves
        return resp

    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode='wb', compresslevel=6) as gz:
        gz.write(data)
    resp.set_data(buf.getvalue())
    resp.headers['Content-Encoding'] = 'gzip'
    resp.headers['Content-Length'] = str(len(resp.get_data()))
    resp.headers.add('Vary', 'Accept-Encoding')
    return resp


@app.get('/static/<path:filename>')
def legacy_static(filename):
    """Generated figures and the embedding projection still live in static/."""
    return send_from_directory('static', filename)


@app.get('/')
@app.get('/<path:path>')
def spa(path=''):
    """Serve the built front end, falling back to index.html for client routes."""
    if not os.path.isdir(DIST_DIR):
        return (
            "<h1>Front end not built</h1>"
            "<p>Run <code>npm install &amp;&amp; npm run build</code> in <code>frontend/</code>.</p>",
            503,
        )
    candidate = os.path.join(DIST_DIR, path)
    if path and os.path.isfile(candidate):
        return send_from_directory(DIST_DIR, path)
    return send_from_directory(DIST_DIR, 'index.html')


if __name__ == '__main__':
    app.run(debug=os.environ.get('FLASK_DEBUG') == '1',
            port=int(os.environ.get('PORT', 5000)))
