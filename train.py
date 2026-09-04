"""Pipeline orchestrator.

Every model is scored through the same harness in evaluation.py, so a number in
the results table always describes the model named beside it. The previous
version printed "Evaluating hybrid recommender" while scoring plain BPR, which
made the headline metrics describe a model the report never discussed.
"""
import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotstyle
plotstyle.apply()
from sklearn.model_selection import train_test_split as sk_split
from sklearn.ensemble import RandomForestClassifier

from data import prepare_data, RANDOM_SEED
from recommender import BayesianPersonalizedRanking, HybridRecommender
from recommender_torch import BPRTorch
from baselines import build_csr, PopularityRecommender, ItemKNN, EASE
from calibration import CalibratedRecommender, user_genre_distribution, \
    list_genre_distribution, kl_miscalibration, calibrated_rerank
from evaluation import evaluate_model, mean_ci, select_users
from explain import CounterfactualExplainer, faithfulness_report, render_explanation
from persistence import save_artifacts, dump_embedding_projection, ARTIFACT_PATH
from xai import run_xai
from bias import run_bias_analysis

SEEDS = [42, 7, 13, 21, 99]
K = 10


def relevance_map(test_df):
    return test_df[test_df['recommend'] == 1].groupby('user_idx')['item_idx'].apply(set).to_dict()


def evaluate_split(bundle, seeds, eval_users, calib_users, quick=False):
    """Evaluate every model on one split. Returns {model_name: {metric: (mean, ci)}}."""
    csr = build_csr(bundle.train_df, bundle.n_users, bundle.n_items)
    train_by_user = bundle.train_df.groupby('user_idx')['item_idx'].apply(list).to_dict()
    test_rel = relevance_map(bundle.test_df)

    # One cohort for every model in this split. Coverage counts distinct
    # recommended items, so it scales with cohort size -- evaluating models on
    # different cohorts would put incomparable numbers in the same column.
    cohort = select_users(test_rel, max_users=eval_users, seed=RANDOM_SEED)
    print(f"  evaluation cohort: {len(cohort)} users (shared by all models)")

    def ev(model, mode='full'):
        return evaluate_model(model, train_by_user, test_rel, bundle.n_items,
                              k=K, mode=mode, users=cohort)

    table = {}

    # Deterministic models: closed-form or count-based, so repeated seeds would
    # produce identical numbers. Reported with a zero-width interval, not faked.
    print("  Popularity ...")
    table['Popularity'] = {m: (v, 0.0) for m, v in ev(PopularityRecommender().fit(csr)).items()}
    print("  ItemKNN ...")
    table['ItemKNN'] = {m: (v, 0.0) for m, v in ev(ItemKNN(topk=200).fit(csr)).items()}
    print("  EASE ...")
    table['EASE'] = {m: (v, 0.0) for m, v in ev(EASE(reg=250.0).fit(csr)).items()}

    # Stochastic models across seeds.
    runs = {'BPR': [], 'Hybrid': [], 'Content': []}
    last_bpr = None
    for seed in seeds:
        print(f"  BPR(torch) seed={seed} ...")
        bpr = BPRTorch(bundle.n_users, bundle.n_items, factors=64, learning_rate=0.01,
                       regularization=0.01, iterations=50, random_state=seed).fit(csr, verbose=False)
        last_bpr = bpr
        runs['BPR'].append(ev(bpr))

        hyb = HybridRecommender(bpr, bundle.item_feat_df, bundle.train_df,
                                bundle.games_in_use, bundle.n_users, bundle.n_items)
        runs['Hybrid'].append(ev(hyb))
        runs['Content'].append(ev(ContentOnly(hyb, bundle.n_items)))

    for name, rs in runs.items():
        table[name] = {m: mean_ci([r[m] for r in rs]) for m in rs[0]}

    # Calibrated re-ranking is a greedy per-user search, so it is evaluated on a
    # subsample. The sample size is reported rather than hidden.
    if not quick:
        print("  Calibrated (lambda=0.5) ...")
        hyb = HybridRecommender(last_bpr, bundle.item_feat_df, bundle.train_df,
                                bundle.games_in_use, bundle.n_users, bundle.n_items)
        cal = CalibratedRecommender(hyb, bundle.games_in_use, train_by_user,
                                    bundle.n_items, lam=0.5, n=K)
        table['Calibrated'] = {m: (v, 0.0) for m, v in ev(cal).items()}

    return table, last_bpr, train_by_user, test_rel, csr, cohort


class ContentOnly:
    """Content-based half of the hybrid, exposed on its own for comparison."""

    def __init__(self, hybrid, n_items):
        self.hybrid = hybrid
        self.n_items = n_items

    def score_all(self, user_idx):
        return self.hybrid.content_scores(user_idx, n=K)


def print_table(title, table):
    metrics = [f'NDCG@{K}', f'MAP@{K}', f'Recall@{K}', f'HitRate@{K}', 'Coverage', 'Gini']
    print(f"\n{title}")
    print('-' * 104)
    print(f"{'Model':<14}" + ''.join(f"{m:>15}" for m in metrics))
    print('-' * 104)
    for name, row in table.items():
        cells = []
        for m in metrics:
            mean, ci = row.get(m, (0.0, 0.0))
            cells.append(f"{mean:.4f}+-{ci:.3f}" if ci else f"{mean:.4f}      ")
        print(f"{name:<14}" + ''.join(f"{c:>15}" for c in cells))
    print('-' * 104)


def plot_comparison(table, path, title):
    names = list(table.keys())
    means = [table[n][f'NDCG@{K}'][0] for n in names]
    cis = [table[n][f'NDCG@{K}'][1] for n in names]
    cov = [table[n]['Coverage'][0] for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    axes[0].bar(names, means, yerr=cis, capsize=4, color=plotstyle.BLUE, edgecolor=plotstyle.BG)
    axes[0].set_ylabel(f'NDCG@{K}')
    axes[0].set_title(f'Ranking accuracy ({title})')
    axes[0].tick_params(axis='x', rotation=30)

    axes[1].bar(names, cov, color=plotstyle.AMBER, edgecolor=plotstyle.BG)
    axes[1].set_ylabel('Catalogue coverage')
    axes[1].set_title(f'Share of catalogue ever recommended ({title})')
    axes[1].tick_params(axis='x', rotation=30)

    plt.suptitle('Every model scored through the same harness', fontsize=13)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def calibration_sweep(hyb, bundle, train_by_user, test_rel, users, lams):
    """Accuracy vs miscalibration as lambda sweeps. The curve is the result."""
    out = []
    for lam in lams:
        cal = CalibratedRecommender(hyb, bundle.games_in_use, train_by_user,
                                    bundle.n_items, lam=lam, n=K)
        res = evaluate_model(cal, train_by_user, test_rel, bundle.n_items,
                             k=K, mode='full', users=users)
        mis = []
        for uid in users:
            hist = train_by_user.get(uid, [])
            if not hist:
                continue
            p = user_genre_distribution(hist, bundle.games_in_use)
            order = calibrated_rerank(hyb.score_all(uid), hist, bundle.games_in_use,
                                      n=K, lam=lam, exclude=set(hist))
            q = list_genre_distribution(order, bundle.games_in_use)
            mis.append(kl_miscalibration(p, q))
        out.append({'lambda': lam,
                    f'NDCG@{K}': res[f'NDCG@{K}'],
                    'miscalibration': float(np.mean(mis)) if mis else 0.0,
                    'coverage': res['Coverage']})
        print(f"    lambda={lam:.2f}  NDCG={res[f'NDCG@{K}']:.4f}  KL={out[-1]['miscalibration']:.4f}")
    return out


def plot_calibration(sweep, path):
    lams = [s['lambda'] for s in sweep]
    ndcg = [s[f'NDCG@{K}'] for s in sweep]
    kl = [s['miscalibration'] for s in sweep]

    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    ax1.plot(lams, ndcg, marker='o', color=plotstyle.BLUE, linewidth=2, label=f'NDCG@{K}')
    ax1.set_xlabel('lambda  (0 = pure relevance, 1 = pure calibration)')
    ax1.set_ylabel(f'NDCG@{K}', color=plotstyle.BLUE)
    ax2 = ax1.twinx()
    ax2.plot(lams, kl, marker='s', color=plotstyle.AMBER, linewidth=2, label='Miscalibration')
    ax2.set_ylabel('KL miscalibration', color=plotstyle.AMBER)
    plt.title('Accuracy vs calibration trade-off (Steck 2018)')
    fig.tight_layout()
    plt.savefig(path)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=3, help='number of seeds for stochastic models')
    ap.add_argument('--eval-users', type=int, default=1500)
    ap.add_argument('--calib-users', type=int, default=300)
    ap.add_argument('--explain-users', type=int, default=25)
    ap.add_argument('--quick', action='store_true', help='skip calibration, XAI and bias')
    ap.add_argument('--skip-xai', action='store_true')
    args = ap.parse_args()

    os.makedirs('static', exist_ok=True)
    seeds = SEEDS[:max(1, args.seeds)]
    results = {}

    # ---------- Random split (the original protocol) ----------
    print("\n=== Split: random (per-user) ===")
    bundle = prepare_data(split='random')
    print(f"  {bundle.n_users} users, {bundle.n_items} items, {len(bundle.df_filtered)} interactions")
    table, bpr, train_by_user, test_rel, csr, cohort = evaluate_split(
        bundle, seeds, args.eval_users, args.calib_users, quick=args.quick)
    print_table("RANDOM SPLIT", table)
    plot_comparison(table, 'static/model_comparison.png', 'random split')
    results['random_split'] = {k: {m: list(v) for m, v in row.items()} for k, row in table.items()}

    # Backwards-compatible keys for the existing dashboard.
    results['NDCG_10'] = table['Hybrid'][f'NDCG@{K}'][0]
    results['MAP_10'] = table['Hybrid'][f'MAP@{K}'][0]

    # ---------- Temporal split ----------
    print("\n=== Split: temporal (global time-ordered) ===")
    tbundle = prepare_data(split='temporal')
    ttable, _, _, _, _, _ = evaluate_split(tbundle, seeds, args.eval_users,
                                          args.calib_users, quick=args.quick)
    print_table("TEMPORAL SPLIT", ttable)
    plot_comparison(ttable, 'static/model_comparison_temporal.png', 'temporal split')
    results['temporal_split'] = {k: {m: list(v) for m, v in row.items()} for k, row in ttable.items()}

    # ---------- Protocol comparison: how optimistic was 99-sampled-negatives? ----------
    print("\n=== Evaluation protocol comparison ===")
    full = evaluate_model(bpr, train_by_user, test_rel, bundle.n_items, k=K,
                          mode='full', users=cohort)
    sampled = evaluate_model(bpr, train_by_user, test_rel, bundle.n_items, k=K,
                             mode='sampled', users=cohort)
    print(f"  full ranking      NDCG@{K} = {full[f'NDCG@{K}']:.4f}")
    print(f"  99 sampled negs   NDCG@{K} = {sampled[f'NDCG@{K}']:.4f}")
    results['protocol'] = {'full': full, 'sampled_99': sampled}

    # ---------- numpy vs torch parity at full scale ----------
    print("\n=== BPR implementation parity (numpy reference vs torch port) ===")
    np_bpr = BayesianPersonalizedRanking(factors=64, learning_rate=0.01,
                                         regularization=0.01, iterations=50,
                                         random_state=RANDOM_SEED).fit(csr)
    np_res = evaluate_model(np_bpr, train_by_user, test_rel, bundle.n_items, k=K,
                            mode='full', users=cohort)
    torch_mean, torch_ci = table['BPR'][f'NDCG@{K}']
    print(f"  numpy NDCG@{K} = {np_res[f'NDCG@{K}']:.4f}")
    print(f"  torch NDCG@{K} = {torch_mean:.4f} +- {torch_ci:.4f}")
    agree = abs(np_res[f'NDCG@{K}'] - torch_mean) <= max(torch_ci, 0.02)
    print(f"  parity: {'PASS' if agree else 'REVIEW'}")
    results['parity'] = {'numpy': np_res[f'NDCG@{K}'], 'torch': torch_mean,
                         'torch_ci': torch_ci, 'agree': bool(agree)}

    # Persist the torch model: it is the one that is served, explained, and
    # re-ranked. The numpy version exists to check it, not to ship.
    save_artifacts(bpr, bundle)
    print(f"  artifacts saved to {ARTIFACT_PATH}")
    proj = dump_embedding_projection(bpr.item_factors, bundle.games_in_use)
    print(f"  embedding projection saved to {proj}")

    hyb = HybridRecommender(bpr, bundle.item_feat_df, bundle.train_df,
                            bundle.games_in_use, bundle.n_users, bundle.n_items)

    if not args.quick:
        # ---------- Calibration trade-off ----------
        print("\n=== Calibration sweep (Steck 2018) ===")
        sweep = calibration_sweep(hyb, bundle, train_by_user, test_rel,
                                  cohort[:args.calib_users], [0.0, 0.25, 0.5, 0.75, 1.0])
        plot_calibration(sweep, 'static/calibration_tradeoff.png')
        results['calibration_sweep'] = sweep

    # ---------- Counterfactual explanations ----------
    print("\n=== Counterfactual explanations ===")
    explainer = CounterfactualExplainer(bpr.item_factors, bundle.n_items,
                                        train_by_user, bundle.games_in_use)
    sample_users = [u for u in list(test_rel.keys()) if len(train_by_user.get(u, [])) >= 3]
    sample_users = sample_users[:args.explain_users]

    examples = []
    for uid in sample_users[:5]:
        loo = explainer.leave_one_out(uid)
        if loo:
            # Item indices as well as names: the hero renders the catalogue as a
            # point cloud and needs to locate this explanation's games in it.
            supporters = [d for d in loo['influences'] if d['rank_delta'] > 0][:3]
            examples.append({'user_idx': int(uid), 'text': render_explanation(loo),
                             'target': loo['target_name'],
                             'target_idx': int(loo['target_item']),
                             'supporters': [d['name'] for d in loo['influences']],
                             'supporter_idx': [int(d['item_idx']) for d in supporters]})
            print(f"  {render_explanation(loo)}")

    faith = faithfulness_report(explainer, sample_users, k=K, max_set=3)
    print(f"\n  explanations checked : {faith['n_explanations_checked']}")
    print(f"  mean faithfulness    : {faith['mean_faithfulness']:.1%} "
          f"+-{faith['faithfulness_ci']:.1%}")
    results['explanations'] = {'examples': examples,
                               'n_checked': faith['n_explanations_checked'],
                               'mean_faithfulness': faith['mean_faithfulness'],
                               'faithfulness_ci': faith['faithfulness_ci']}

    # ---------- Catalogue-level attribute analysis (formerly mislabelled as XAI) ----------
    if not args.quick and not args.skip_xai:
        print("\n=== Catalogue-level attribute analysis (SHAP/LIME) ===")
        X = bundle.item_feat_df.loc[bundle.df_filtered['item_idx'].values].values.astype(np.float32)
        y = bundle.df_filtered['recommend'].values.astype(np.int32)
        X_train, X_test, y_train, y_test = sk_split(X, y, test_size=0.2,
                                                    random_state=RANDOM_SEED, stratify=y)
        clf = RandomForestClassifier(n_estimators=100, max_depth=6,
                                     random_state=RANDOM_SEED, n_jobs=-1)
        clf.fit(X_train, y_train)
        run_xai(X_train, X_test, y_train, y_test, clf, bundle.feature_cols, RANDOM_SEED)

        print("\n=== Bias analysis ===")
        test_uids = list(test_rel.keys())
        results['bias'] = run_bias_analysis(bundle.df_filtered, bundle.games_in_use,
                                            test_uids, hyb)

    with open('results.json', 'w') as f:
        json.dump(results, f, indent=4, default=float)
    print("\nDone. results.json written; plots in static/")


if __name__ == "__main__":
    main()
