<h1 align="center">🎮 Steam Game Recommender — Explainability and Ethics</h1>

<p align="center">
  A hybrid recommender system built on real Steam data, with SHAP and LIME explainability and a full popularity bias analysis including two mitigation strategies.
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" />
  <img alt="Flask" src="https://img.shields.io/badge/Dashboard-Flask-000000?logo=flask&logoColor=white" />
  <img alt="SHAP" src="https://img.shields.io/badge/XAI-SHAP%20%2B%20LIME-FF6B6B" />
  <img alt="NDCG" src="https://img.shields.io/badge/NDCG@10-0.5203-2ea44f" />
  <img alt="MAP" src="https://img.shields.io/badge/MAP@10-0.4502-2ea44f" />
  <img alt="Status" src="https://img.shields.io/badge/Status-Completed-2ea44f" />
</p>

---

## 🔎 What This Project Is

This project builds a hybrid game recommender system on real Steam user review data and then goes a step further by asking two important questions: can we explain why the model made a recommendation, and is the system treating all games fairly?

The recommender combines Bayesian Personalised Ranking (BPR) with content-based filtering using game genres, tags and price. SHAP and LIME are applied to make the model's decisions transparent. A full popularity bias analysis then measures how much the system amplifies the dominance of already-popular games, and two mitigation strategies are tested and evaluated.

All results are presented through a designed Flask web dashboard with clear explanations for every chart.

---

## 🎥 Demo Video

<!-- Record your screen running train.py and the Flask dashboard, then upload here -->

https://github.com/user-attachments/assets/5c6a83db-2936-4112-901a-addcc85d5c54

---

## 📊 Recommender Performance

| Metric | Score |
|---|---|
| NDCG@10 | **0.5203** |
| MAP@10 | **0.4502** |

Evaluation used sampled ranking where each test item was ranked against 99 randomly sampled unseen games. This is the standard approach for sparse interaction data where full-pool ranking produces near-zero scores.

---

## 🗂️ Dataset

| Detail | Value |
|---|---|
| Source | UCSD McAuley Lab — Australian User Reviews + Steam Games |
| Users | 6,684 (filtered to 3+ reviews) |
| Games | 2,758 |
| Interactions | 31,799 |
| Sparsity | 99.83% |
| Split | 80% Train / 10% Validation / 10% Test (per user) |

---

## 🧠 Model Architecture

**BPR (Bayesian Personalised Ranking)**
64 latent factors, learning rate 0.01, regularisation 0.01, 50 iterations. Learns from implicit feedback — what users played, not explicit star ratings.

**Content-Based Filtering**
Binary vectors for top 10 genres and top 10 tags, an early access flag, and normalised price capped at $60 and scaled to [0, 1].

**Hybrid Blending**
Alpha = 0.7, meaning 70% BPR score and 30% content-based score for every recommendation.

**Cold-Start Strategy**
New users with no history are handled using genre preferences inferred from community sentiment scores from other players.

---

## 🔍 Explainability — SHAP and LIME

### SHAP (Global Explanations)

SHAP assigns each feature a contribution value using Shapley values from cooperative game theory. It answers the question of which features matter most across all predictions. The top three most important features globally were Open World, normalised price, and Singleplayer.

The most interesting finding from the beeswarm plot was around price. Price has a non-linear relationship with recommendations where both very cheap and very expensive games push predictions in different directions. This effect is only visible in the beeswarm view and would be missed looking at the bar chart alone.

### LIME (Local Explanations)

LIME explains individual predictions by fitting a simple model around one specific instance. It answers the question of why the model recommended one particular game to one particular user.

For the recommended instance with a probability of 0.87, the strongest positive signal was Early Access being absent and the strongest negative signal was Open World being present. For the not-recommended instance with a probability of 0.90, the strongest negative signal was Free to Play being present and the absence of Singleplayer also pushed against recommendation strongly.

**Stability check:** At n=100 samples Sports appeared as the top ranked feature but at n=500 it disappeared entirely and was replaced by FPS, confirming that n=100 is too small for reliable LIME explanations. n=500 is the safe minimum for stable results.

---

## ⚖️ Popularity Bias and Mitigation

### Bias Measurement

| Genre | Catalogue Share | Recommendation Share |
|---|---|---|
| Action | 24.18% | 26.36% |
| Strategy | 9.97% | 10.07% |
| Indie | Large catalogue | 9.67 avg interactions per game |

Indie games make up the largest genre in the catalogue but receive far fewer interactions per game than Action titles. The recommender amplifies this gap rather than correcting it.

### Mitigation 1 — Inverse Popularity Re-Ranking

Games with fewer interactions receive a score boost controlled by a beta parameter. Lowering beta gives less popular games a fairer chance at appearing in recommendation lists.

| Beta | Average Item Popularity |
|---|---|
| 1.0 (no change) | 600.36 |
| 0.4 (strongest correction) | 84.39 |

### Mitigation 2 — Genre Diversity Constraint

A greedy cap prevents any single genre from taking more than 3 slots in a recommendation list, guaranteeing that smaller genres get some representation regardless of their individual scores.

| Setting | Shannon Entropy |
|---|---|
| Standard recommender | 2.751 |
| With diversity constraint | 2.903 |

A gain of 0.15 bits — real but modest. The hybrid already produces some variety so the constraint has limited room to push further.

---

## 📈 Visualisations

The dashboard produces 11 charts covering recommender evaluation, SHAP and LIME explanations, popularity bias measurements and both mitigation results. All charts are viewable in the demo video above.

---

## 🗂️ Project Structure

```text
steam-recommender-xai/
├── data.py              # Data loading, filtering, splitting, feature matrix
├── recommender.py       # BPR, content-based, hybrid, cold-start, mitigations
├── xai.py               # SHAP and LIME analysis, stability check
├── bias.py              # Popularity bias measurement and both mitigations
├── train.py             # Full pipeline — saves all plots and results.json
├── app.py               # Flask dashboard
├── static/              # Generated PNG plots (created by train.py)
├── results.json         # Metrics summary (created by train.py)
└── requirements.txt
```

---

## ⚙️ Setup and Usage

```bash
# 1. Clone the repository
git clone https://github.com/abinashprasana/steam-recommender-xai.git
cd steam-recommender-xai

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the training pipeline
python train.py

# 4. Launch the Flask dashboard
python app.py
```

Then open `http://localhost:5000` in your browser.

---

## 🧪 Limitations and Future Work

- The dataset comes from Australian users only and may not reflect global Steam preferences
- LIME explanations vary with sample size,  n=500 should be treated as the minimum for any reliable result
- The diversity constraint improves entropy but has not been validated against actual user preferences,  a proper user study would be needed
- SHAP and LIME are applied to a proxy Random Forest rather than BPR directly, since BPR's 64-dimensional latent vectors have no direct feature interpretability
- Combining both mitigations would require joint parameter calibration to avoid conflicting adjustments
---

## 📌 Dataset Source

**UCSD McAuley Lab — Steam Dataset**
Available at: https://cseweb.ucsd.edu/~jmcauley/datasets.html

---

## 🙋 Author

**Abinash Prasana Selvanathan**

⭐ If you found this useful, feel free to star the repo.
