import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

def run_bias_analysis(df_filtered, games_in_use, test_uids, hybrid_recommender, output_dir='static'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    target_genres = ['Indie', 'Action', 'Casual', 'Adventure', 'Strategy', 'Simulation', 'RPG', 'Sports', 'Racing']

    genre_ints = Counter()
    genre_games = Counter()
    for _, row in df_filtered.iterrows():
        for g in row['genres']:
            genre_ints[g] += 1
    for _, row in games_in_use.iterrows():
        for g in row['genres']:
            genre_games[g] += 1

    bias_rows = []
    for g in target_genres:
        n_g = genre_games.get(g, 1)
        n_i = genre_ints.get(g, 0)
        bias_rows.append({
            'Genre': g, 'Games': n_g, 'Interactions': n_i, 'Per Game': round(n_i / n_g, 2)
        })
    bias_df = pd.DataFrame(bias_rows).sort_values('Per Game', ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sorted_b = bias_df.sort_values('Games', ascending=False)
    x = np.arange(len(sorted_b))
    w = 0.35
    axes[0].bar(x - w/2, sorted_b['Games'], w, label='Games in catalogue', color='#2980B9', edgecolor='white')
    int_scaled = (sorted_b['Interactions'] / sorted_b['Interactions'].max() * sorted_b['Games'].max())
    axes[0].bar(x + w/2, int_scaled, w, label='Interactions (scaled)', color='#E8A838', edgecolor='white')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(sorted_b['Genre'], rotation=30, ha='right')
    axes[0].set_title('Games in Catalogue vs Interaction Volume')
    axes[0].set_ylabel('Count')
    axes[0].legend()

    by_ratio = bias_df.sort_values('Per Game', ascending=True)
    bar_colors = ['#E67E22' if g == 'Indie' else '#2980B9' for g in by_ratio['Genre']]
    axes[1].barh(by_ratio['Genre'], by_ratio['Per Game'], color=bar_colors, edgecolor='white')
    axes[1].set_title('Interactions per Game by Genre (Indie highlighted)')
    axes[1].set_xlabel('Avg Interactions per Game')
    plt.suptitle('Popularity Bias in the Steam Dataset', fontsize=13)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/popularity_bias.png')
    plt.close()

    total_games = sum(genre_games[g] for g in target_genres)
    cat_share = {g: 100 * genre_games.get(g, 0) / total_games for g in target_genres}

    rec_genre_counts = Counter()
    total_rec = 0
    for uid in test_uids[:100]:
        for idx in hybrid_recommender.hybrid_recommend(uid, n=10, alpha=0.7):
            if idx in games_in_use.index:
                for g in games_in_use.loc[idx, 'genres']:
                    if g in target_genres:
                        rec_genre_counts[g] += 1
                        total_rec += 1
    rec_share = {g: 100 * rec_genre_counts.get(g, 0) / max(total_rec, 1) for g in target_genres}

    compare_df = pd.DataFrame({
        'Genre': target_genres,
        'Catalogue (%)': [cat_share[g] for g in target_genres],
        'Recommendations (%)': [rec_share[g] for g in target_genres],
    }).sort_values('Catalogue (%)', ascending=False)

    x = np.arange(len(compare_df))
    w = 0.35
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x - w/2, compare_df['Catalogue (%)'], w, label='In Catalogue', color='#2980B9', edgecolor='white')
    ax.bar(x + w/2, compare_df['Recommendations (%)'], w, label='In Recommendations', color='#E67E22', edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(compare_df['Genre'], rotation=30, ha='right')
    ax.set_title('Catalogue vs Recommendation Genre Share (gap = bias)')
    ax.set_ylabel('Share (%)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/genre_share_bias.png')
    plt.close()

    item_pop = df_filtered.groupby('item_idx').size().to_dict()
    max_pop = max(item_pop.values()) if item_pop else 1

    sample_100 = test_uids[:100]
    betas = [1.0, 0.8, 0.6, 0.4]
    result = []
    for b in betas:
        pops = [np.mean([item_pop.get(i, 0) for i in hybrid_recommender.fair_recommend(uid, item_pop, max_pop, beta=b)]) for uid in sample_100]
        result.append({'beta': b, 'avg_popularity': np.mean(pops)})
    res_df = pd.DataFrame(result)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(res_df['beta'], res_df['avg_popularity'], marker='o', color='#4C72B0', linewidth=2, markersize=8)
    ax.set_xlabel('Beta  (1.0 = standard, lower = more fairness)')
    ax.set_ylabel('Avg Item Popularity in Recommendations')
    ax.set_title('Mitigation 1 — Popularity Drops as Beta Decreases')
    ax.invert_xaxis()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/beta_sweep.png')
    plt.close()

    def genre_entropy(rec_list):
        gc = Counter()
        for idx in rec_list:
            if idx in games_in_use.index:
                gc.update(games_in_use.loc[idx, 'genres'])
        total = sum(gc.values())
        if total == 0:
            return 0.0
        return -sum((c / total) * np.log2(c / total + 1e-9) for c in gc.values())

    std_ent = [genre_entropy(hybrid_recommender.hybrid_recommend(uid)) for uid in sample_100]
    div_ent = [genre_entropy(hybrid_recommender.diverse_recommend(uid)) for uid in sample_100]
    
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(std_ent, bins=15, alpha=0.65, label='Standard', color='#2E86AB', edgecolor='white')
    ax.hist(div_ent, bins=15, alpha=0.65, label='Diversified', color='#F18F01', edgecolor='white')
    ax.axvline(np.mean(std_ent), color='#1a5f7a', linestyle='--', linewidth=1.5, label=f'Mean std = {np.mean(std_ent):.2f}')
    ax.axvline(np.mean(div_ent), color='#9e5e00', linestyle='--', linewidth=1.5, label=f'Mean div = {np.mean(div_ent):.2f}')
    ax.set_xlabel('Genre Entropy (higher = more varied)')
    ax.set_ylabel('Number of Users')
    ax.set_title('Mitigation 2 — Diversity Constraint Shifts the Distribution Right')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/entropy_comparison.png')
    plt.close()

    return {
        "beta_sweep": result,
        "mean_std_entropy": float(np.mean(std_ent)),
        "mean_div_entropy": float(np.mean(div_ent))
    }
