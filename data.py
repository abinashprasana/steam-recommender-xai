import os
import gzip
import urllib.request
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.preprocessing import LabelEncoder

RANDOM_SEED = 42

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

def load_reviews(path):
    rows = []
    with gzip.open(path, 'rb') as f:
        for line in f:
            d = eval(line)
            uid = d.get('user_id', '')
            for r in d.get('reviews', []):
                rows.append({
                    'user_id': uid,
                    'item_id': str(r.get('item_id', '')),
                    'recommend': int(r.get('recommend', False)),
                })
    return pd.DataFrame(rows)

def load_games(path):
    rows = []
    with gzip.open(path, 'rb') as f:
        for line in f:
            d = eval(line)
            rows.append({
                'item_id': str(d.get('id', '')),
                'app_name': d.get('app_name', ''),
                'genres': d.get('genres') or [],
                'tags': d.get('tags') or [],
                'price': d.get('price', None),
                'early_access': int(d.get('early_access', False)),
                'sentiment': d.get('sentiment', ''),
                'developer': d.get('developer', ''),
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
        except:
            price = 0.0
        rows.append(g_vec + t_vec + [price, int(row['early_access'])])
    return pd.DataFrame(rows, index=games_sub.index, columns=FEATURE_COLS)

def prepare_data():
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
    
    return df_filtered, train_df, val_df, test_df, games_in_use, item_feat_df, n_users, n_items, FEATURE_COLS
