import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as sk_split
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import csr_matrix

from data import prepare_data, RANDOM_SEED
from recommender import BayesianPersonalizedRanking, HybridRecommender
from xai import run_xai
from bias import run_bias_analysis

def ndcg_at_k(recs, relevant, k=10):
    if not relevant: return 0.0
    hits = [1 if r in relevant else 0 for r in recs[:k]]
    dcg = sum(h / np.log2(i + 2) for i, h in enumerate(hits))
    ideal = sorted(hits, reverse=True)
    idcg = sum(h / np.log2(i + 2) for i, h in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0

def map_at_k(recs, relevant, k=10):
    if not relevant: return 0.0
    hits, s = 0, 0.0
    for i, r in enumerate(recs[:k]):
        if r in relevant:
            hits += 1
            s += hits / (i + 1)
    return s / min(len(relevant), k)

def main():
    if not os.path.exists('static'):
        os.makedirs('static')

    print("Step 1: Preparing data...")
    df_filtered, train_df, val_df, test_df, games_in_use, item_feat_df, n_users, n_items, FEATURE_COLS = prepare_data()

    print("Step 2: Training BPR model...")
    train_csr = csr_matrix(
        (np.ones(len(train_df), dtype=np.float32),
         (train_df['user_idx'].values, train_df['item_idx'].values)),
        shape=(n_users, n_items)
    )
    bpr = BayesianPersonalizedRanking(factors=64, learning_rate=0.01, regularization=0.01, iterations=50)
    bpr.fit(train_csr)

    print("Step 3: Evaluating hybrid recommender...")
    hybrid_rec = HybridRecommender(bpr, item_feat_df, train_df, games_in_use, n_users, n_items)
    
    rng_eval = np.random.RandomState(RANDOM_SEED)
    test_rel = (test_df[test_df['recommend'] == 1].groupby('user_idx')['item_idx'].apply(set).to_dict())
    test_uids = list(test_rel.keys())
    
    ndcg_list, map_list = [], []
    train_by_user = train_df.groupby('user_idx')['item_idx'].apply(list).to_dict()
    print(f"Evaluating {len(test_uids)} test users...")
    for uid in tqdm(test_uids):
        relevant = test_rel[uid]
        seen = set(train_by_user.get(uid, []))
        all_unseen = list(set(range(n_items)) - seen - relevant)
        if len(all_unseen) < 99: continue
        neg_sample = rng_eval.choice(all_unseen, size=99, replace=False)
        pool = list(relevant) + list(neg_sample)
        sc = bpr.score_all(uid)[pool]
        ranked = [pool[i] for i in np.argsort(sc)[::-1]]
        ndcg_list.append(ndcg_at_k(ranked, relevant))
        map_list.append(map_at_k(ranked, relevant))

    mean_ndcg = np.mean(ndcg_list)
    mean_map = np.mean(map_list)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    vals = [mean_ndcg, mean_map]
    names = ['NDCG@10', 'MAP@10']
    bars = axes[0].bar(names, vals, color=['#4C72B0', '#55A868'], edgecolor='white', width=0.4)
    axes[0].set_title('Sampled Evaluation Metrics (k=10)')
    axes[0].set_ylabel('Score')
    axes[0].set_ylim(0, max(vals) * 1.5 if max(vals) > 0 else 0.5)
    for bar, v in zip(bars, vals):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.04, f'{v:.4f}', ha='center', fontsize=12, fontweight='bold')
    
    axes[1].hist(ndcg_list, bins=20, color='#4C72B0', edgecolor='white', alpha=0.85)
    axes[1].axvline(mean_ndcg, color='red', linestyle='--', linewidth=1.5, label=f'Mean = {mean_ndcg:.3f}')
    axes[1].set_title('Per-User NDCG@10 Distribution')
    axes[1].set_xlabel('NDCG@10')
    axes[1].set_ylabel('Number of Users')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig('static/evaluation_metrics.png')
    plt.close()

    print("Step 4: Running XAI analysis...")
    X = item_feat_df.loc[df_filtered['item_idx'].values].values.astype(np.float32)
    y = df_filtered['recommend'].values.astype(np.int32)
    X_train, X_test, y_train, y_test = sk_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)
    
    clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=RANDOM_SEED, n_jobs=-1)
    clf.fit(X_train, y_train)
    run_xai(X_train, X_test, y_train, y_test, clf, FEATURE_COLS, RANDOM_SEED)

    print("Step 5: Running Bias analysis...")
    bias_results = run_bias_analysis(df_filtered, games_in_use, test_uids, hybrid_rec)

    results = {
        "NDCG_10": float(mean_ndcg),
        "MAP_10": float(mean_map),
        "bias": bias_results
    }
    
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    print("Done! Results saved to results.json and plots to static/")

if __name__ == "__main__":
    main()
