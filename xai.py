import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotstyle
plotstyle.apply()
import shap
import lime
import lime.lime_tabular

def run_xai(X_train, X_test, y_train, y_test, clf, FEATURE_COLS, RANDOM_SEED=42, output_dir='static'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Computing SHAP values...")
    explainer = shap.TreeExplainer(clf)
    shap_vals = explainer.shap_values(X_test[:500])

    if isinstance(shap_vals, list):
        sv = shap_vals[1]
    elif hasattr(shap_vals, 'ndim') and shap_vals.ndim == 3:
        sv = shap_vals[:, :, 1]
    else:
        sv = shap_vals

    mean_shap = np.abs(sv).mean(axis=0)
    shap_df = pd.DataFrame({'feature': FEATURE_COLS, 'importance': mean_shap})
    shap_df = shap_df.sort_values('importance', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    bar_colors = [plotstyle.BLUE if v > shap_df['importance'].median() else plotstyle.SLATE for v in shap_df['importance']]
    ax.barh(shap_df['feature'], shap_df['importance'], color=bar_colors, edgecolor=plotstyle.BG)
    ax.axvline(shap_df['importance'].median(), color=plotstyle.AMBER, linestyle='--', linewidth=1, alpha=0.6, label='Median')
    ax.set_xlabel('Mean |SHAP value|')
    ax.set_title('SHAP — Which Features Drive Recommendations Most?')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/shap_bar.png')
    plt.close()

    plt.figure(figsize=(10, 7))
    shap.summary_plot(sv, X_test[:500], feature_names=FEATURE_COLS, show=False)
    plt.title('SHAP Beeswarm — Feature Direction and Strength')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/shap_beeswarm.png')
    plt.close()

    try:
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train,
            feature_names=FEATURE_COLS,
            class_names=['Not Recommended', 'Recommended'],
            mode='classification',
            random_state=RANDOM_SEED
        )
    except TypeError:
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train,
            feature_names=FEATURE_COLS,
            class_names=['Not Recommended', 'Recommended'],
            mode='classification'
        )

    probs = clf.predict_proba(X_test)
    pos_idxs = np.where(probs[:, 1] > 0.80)[0]
    neg_idxs = np.where(probs[:, 1] < 0.20)[0]
    pos_i = pos_idxs[0] if len(pos_idxs) > 0 else 0
    neg_i = neg_idxs[0] if len(neg_idxs) > 0 else 1

    for label_name, idx in [("Recommended", pos_i), ("Not_Recommended", neg_i)]:
        inst = X_test[idx]
        prob = probs[idx, 1]
        
        exp = lime_explainer.explain_instance(
            data_row=inst,
            predict_fn=clf.predict_proba,
            num_features=10,
            num_samples=500
        )

        exp_list = sorted(exp.as_list(), key=lambda x: x[1])
        feat_labels = [f[:35] for f, w in exp_list]
        weights = [w for f, w in exp_list]
        bar_colors = [plotstyle.BLUE if w > 0 else plotstyle.AMBER for w in weights]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(feat_labels, weights, color=bar_colors, edgecolor=plotstyle.BG)
        ax.axvline(0, color=plotstyle.FAINT, linewidth=0.8, alpha=0.5)
        ax.set_xlabel('LIME weight (positive pushes towards Recommended)')
        ax.set_title(f"LIME — Likely {label_name.replace('_', ' ')} (prob = {prob:.2f})")
        plt.tight_layout()
        plt.savefig(f'{output_dir}/lime_{label_name.lower()}.png')
        plt.close()

    inst = X_test[pos_i]
    sample_sizes = [100, 500, 1000]
    results = {}
    for n in sample_sizes:
        exp_s = lime_explainer.explain_instance(
            data_row=inst, predict_fn=clf.predict_proba, num_features=6, num_samples=n
        )
        results[n] = {(f.split('=')[0].strip()[:20] if '=' in f else f[:20]): round(w, 4) for f, w in exp_s.as_list()}

    stab_df = pd.DataFrame(results).fillna(0)
    stab_df.columns = [f'n={n}' for n in sample_sizes]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(stab_df))
    w = 0.25
    colors = [plotstyle.BLUE, plotstyle.AMBER, plotstyle.VIOLET]
    for i, col in enumerate(stab_df.columns):
        ax.bar(x + i * w, stab_df[col].abs(), w, label=col, color=colors[i], edgecolor=plotstyle.BG, alpha=0.85)
    ax.set_xticks(x + w)
    ax.set_xticklabels(stab_df.index, rotation=30, ha='right')
    ax.set_ylabel('Absolute LIME weight')
    ax.set_title('LIME Stability — same instance, three sample sizes')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/lime_stability.png')
    plt.close()

    top_shap = shap_df.sort_values('importance', ascending=False).head(8)
    exp_pos = lime_explainer.explain_instance(X_test[pos_i], clf.predict_proba, num_features=8, num_samples=500)
    lime_list = sorted(exp_pos.as_list(), key=lambda x: abs(x[1]), reverse=True)[:8]

    shap_norm = top_shap['importance'].values / (top_shap['importance'].max() + 1e-9)
    lime_vals = np.abs([w for _, w in lime_list])
    lime_norm = lime_vals / (lime_vals.max() + 1e-9)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].barh(top_shap['feature'].values[::-1], shap_norm[::-1], color=plotstyle.BLUE, edgecolor=plotstyle.BG)
    axes[0].set_title('SHAP — Top Features (Global Average)')
    axes[0].set_xlabel('Normalised Importance')

    lime_labels = [f.split('=')[0].strip()[:25] if '=' in f else f[:25] for f, _ in lime_list]
    axes[1].barh(lime_labels[::-1], lime_norm[::-1], color=plotstyle.GREEN, edgecolor=plotstyle.BG)
    axes[1].set_title('LIME — Top Features (One Instance)')
    axes[1].set_xlabel('Normalised |Weight|')

    plt.suptitle('SHAP vs LIME — Feature Rankings', fontsize=13)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/shap_vs_lime.png')
    plt.close()
