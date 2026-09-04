import os
import ast
import html
import gzip
import urllib.request
import pandas as pd
from typing import NamedTuple
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

RANDOM_SEED = 42

# Steam omits the year on reviews posted in the current year, so the yearless
# "Posted June 24." form carries an implicit scrape-year date. Derived empirically:
# yearless reviews span January-September and are exactly zero in Oct/Nov/Dec, while
# 2015 is populated in all twelve months. The scrape therefore ran in September 2016.
# Getting this wrong scatters ~17% of interactions into the middle of the timeline
# instead of the end, which would corrupt any temporal split.
SCRAPE_YEAR = 2016

def download_data():
    files = {
        "australian_user_reviews.json.gz": "https://mcauleylab.ucsd.edu/public_datasets/data/steam/australian_user_reviews.json.gz",
        "steam_games.json.gz": "https://cseweb.ucsd.edu/~wckang/steam_games.json.gz",
    }
    for fname, url in files.items():
        if not os.path.exists(fname):
            print(f"Downloading {fname} ...")
            urllib.request.urlretrieve(url, fname)
            print("  done.")
        else:
            print(f"  {fname} already present.")

def parse_posted(posted):
    """Parse Steam's 'Posted <Month> <Day>[, <Year>].' into (datetime, was_imputed).

    Returns (None, False) when the string is missing or unparseable.
    """
    if not posted:
        return None, False
    s = posted.strip()
    if s.lower().startswith('posted'):
        s = s[len('posted'):]
    s = s.strip().rstrip('.').strip()
    if not s:
        return None, False
    try:
        return datetime.strptime(s, '%B %d, %Y'), False
    except ValueError:
        pass
    try:
        return datetime.strptime(f'{s}, {SCRAPE_YEAR}', '%B %d, %Y'), True
    except ValueError:
        return None, False

def load_reviews(path):
    rows = []
    with gzip.open(path, 'rb') as f:
        for line in f:
            d = ast.literal_eval(line.decode('utf-8'))
            uid = d.get('user_id', '')
            for r in d.get('reviews', []):
                posted_date, imputed = parse_posted(r.get('posted', ''))
                rows.append({
                    'user_id': uid,
                    'item_id': str(r.get('item_id', '')),
                    'recommend': int(r.get('recommend', False)),
                    'posted_date': posted_date,
                    'date_imputed': imputed,
                    'review': r.get('review', '') or '',
                })
    return pd.DataFrame(rows)

def _unescape(value):
    """Undo HTML entities present in the source data.

    Two genre names ship pre-escaped in the dataset -- 'Animation &amp; Modeling'
    and 'Design &amp; Illustration'. Left alone they get escaped a second time on
    render and reach the page as literal '&amp;'.
    """
    if isinstance(value, str):
        return html.unescape(value)
    if isinstance(value, list):
        return [html.unescape(v) if isinstance(v, str) else v for v in value]
    return value


def load_games(path):
    rows = []
    with gzip.open(path, 'rb') as f:
        for line in f:
            d = ast.literal_eval(line.decode('utf-8'))
            rows.append({
                'item_id': str(d.get('id', '')),
                'app_name': _unescape(d.get('app_name', '')),
                'title': _unescape(d.get('title', '')),
                'genres': _unescape(d.get('genres') or []),
                'tags': _unescape(d.get('tags') or []),
                'specs': _unescape(d.get('specs') or []),
                'price': d.get('price', None),
                'early_access': int(d.get('early_access', False)),
                'sentiment': d.get('sentiment', ''),
                'developer': d.get('developer', ''),
                'publisher': d.get('publisher', ''),
                'release_date': d.get('release_date', ''),
            })
    return pd.DataFrame(rows)

def make_split(df, train_r=0.8, val_r=0.1):
    tr, va, te = [], [], []
    for _, grp in df.groupby('user_idx'):
        g = grp.sample(frac=1, random_state=RANDOM_SEED)
        n = len(g)
        t1 = int(n * train_r)
        t2 = int(n * (train_r + val_r))
        tr.append(g.iloc[:t1])
        va.append(g.iloc[t1:t2])
        te.append(g.iloc[t2:])
    return (pd.concat(tr).reset_index(drop=True),
            pd.concat(va).reset_index(drop=True),
            pd.concat(te).reset_index(drop=True))

def make_temporal_split(df, train_r=0.8, val_r=0.1):
    """Global time-ordered split: the last interactions by posting date become test.

    Unlike make_split, this never lets a model train on a user's future to predict
    their past. Rows with no parseable date are dropped rather than guessed at.
    """
    dated = df[df['posted_date'].notna()].sort_values('posted_date', kind='mergesort')
    n = len(dated)
    t1 = int(n * train_r)
    t2 = int(n * (train_r + val_r))
    return (dated.iloc[:t1].reset_index(drop=True),
            dated.iloc[t1:t2].reset_index(drop=True),
            dated.iloc[t2:].reset_index(drop=True))

TOP_GENRES = ['Indie', 'Action', 'Casual', 'Adventure', 'Strategy',
              'Simulation', 'RPG', 'Free to Play', 'Sports', 'Racing']

TOP_TAGS = ['Singleplayer', 'Multiplayer', 'Atmospheric', 'Great Soundtrack',
            'Puzzle', 'Open World', 'Story Rich', 'Shooter', 'Sci-fi', 'FPS']

FEATURE_COLS = TOP_GENRES + TOP_TAGS + ['price_norm', 'early_access']

def build_features(games_sub):
    rows = []
    for _, row in games_sub.iterrows():
        g_vec = [int(g in row['genres']) for g in TOP_GENRES]
        t_vec = [int(t in row['tags']) for t in TOP_TAGS]
        try:
            price = min(float(row['price']), 60.0) / 60.0
        except (TypeError, ValueError):
            price = 0.0
        rows.append(g_vec + t_vec + [price, int(row['early_access'])])
    return pd.DataFrame(rows, index=games_sub.index, columns=FEATURE_COLS)

class DataBundle(NamedTuple):
    """Everything downstream stages need, named rather than positional.

    The label encoders are carried because live inference has to map a real
    user_id/item_id back to the contiguous indices the models are trained on.
    """
    df_filtered: pd.DataFrame
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    games_in_use: pd.DataFrame
    item_feat_df: pd.DataFrame
    n_users: int
    n_items: int
    feature_cols: list
    user_enc: LabelEncoder
    item_enc: LabelEncoder

def prepare_data(split='random'):
    download_data()
    reviews_df = load_reviews("australian_user_reviews.json.gz")
    games_df = load_games("steam_games.json.gz")
    df = reviews_df.merge(games_df, on='item_id', how='inner')

    user_counts = df.groupby('user_id').size()
    active = user_counts[user_counts >= 3].index
    df_filtered = df[df['user_id'].isin(active)].copy()

    user_enc = LabelEncoder()
    item_enc = LabelEncoder()
    df_filtered['user_idx'] = user_enc.fit_transform(df_filtered['user_id'])
    df_filtered['item_idx'] = item_enc.fit_transform(df_filtered['item_id'])

    if split == 'temporal':
        train_df, val_df, test_df = make_temporal_split(df_filtered)
    else:
        train_df, val_df, test_df = make_split(df_filtered)

    item_ids_in_use = df_filtered[['item_idx', 'item_id']].drop_duplicates()
    games_in_use = (games_df[games_df['item_id'].isin(item_ids_in_use['item_id'])]
                    .merge(item_ids_in_use, on='item_id')
                    .set_index('item_idx')
                    .sort_index()
                    .copy())

    SENTIMENT_MAP = {
        'Overwhelmingly Positive': 5, 'Very Positive': 4,
        'Mostly Positive': 3, 'Positive': 3,
        'Mixed': 2, 'Mostly Negative': 1,
        'Negative': 1, 'Very Negative': 0, 'Overwhelmingly Negative': 0,
    }
    games_in_use['sentiment_score'] = games_in_use['sentiment'].map(SENTIMENT_MAP).fillna(2)

    item_feat_df = build_features(games_in_use)

    n_users = df_filtered['user_idx'].nunique()
    n_items = df_filtered['item_idx'].nunique()

    return DataBundle(df_filtered, train_df, val_df, test_df, games_in_use,
                      item_feat_df, n_users, n_items, FEATURE_COLS,
                      user_enc, item_enc)
