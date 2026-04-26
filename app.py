from flask import Flask, render_template_string
import json
import os

app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Steam Game Recommender</title>
    <style>
        :root {
            --bg-color: #0d1117;
            --text-color: #c9d1d9;
            --card-bg: #161b22;
            --accent-color: #58a6ff;
            --border-color: #30363d;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            margin: 0;
            padding: 0;
            line-height: 1.6;
        }
        nav {
            background-color: var(--card-bg);
            border-bottom: 1px solid var(--border-color);
            padding: 1rem 2rem;
            display: flex;
            justify-content: center;
            gap: 2rem;
            position: sticky;
            top: 0;
            z-index: 1000;
        }
        nav a {
            color: var(--text-color);
            text-decoration: none;
            font-weight: 600;
        }
        nav a:hover {
            color: var(--accent-color);
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }
        h1, h2, h3 {
            color: #fff;
        }
        .section {
            margin-bottom: 4rem;
            padding-top: 2rem;
        }
        .card {
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 2rem;
        }
        .plot-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            margin: 2rem 0;
        }
        .plot-container img {
            max-width: 100%;
            height: auto;
            border-radius: 4px;
            border: 1px solid var(--border-color);
        }
        .caption {
            margin-top: 0.5rem;
            font-size: 0.9rem;
            color: #8b949e;
            text-align: center;
        }
        .metrics-table {
            width: 100%;
            max-width: 600px;
            margin: 1rem auto;
            border-collapse: collapse;
        }
        .metrics-table th, .metrics-table td {
            border: 1px solid var(--border-color);
            padding: 0.75rem;
            text-align: center;
        }
        .metrics-table th {
            background-color: #21262d;
        }
        .flex-row {
            display: flex;
            flex-wrap: wrap;
            gap: 2rem;
            justify-content: center;
        }
        .flex-col {
            flex: 1;
            min-width: 300px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        p {
            font-size: 1.05rem;
        }
    </style>
</head>
<body>
    <nav>
        <a href="#performance">Recommender Performance</a>
        <a href="#xai">XAI Analysis</a>
        <a href="#bias">Bias & Fairness</a>
    </nav>
    <div class="container">
        <header style="text-align: center; margin-bottom: 3rem;">
            <h1>Steam Game Recommender Dashboard</h1>
            <p>Analysis of performance, explainability, and popularity bias in game recommendations.</p>
        </header>

        <!-- PERFORMANCE SECTION -->
        <div id="performance" class="section card">
            <h2>Recommender Performance</h2>
            <p>
                This section is all about how well the game recommender actually works in practice. I used a method called Bayesian Personalized Ranking which we usually just call BPR for short. This method learns from what games users have actually played instead of asking them for explicit star ratings. It basically figures out your preferences by looking at the games you chose to play over the ones you ignored. One big challenge in building this was dealing with brand new players who have not played anything yet. This is known as the cold start problem because the system has no history to learn from. To handle these new users my system asks them for their favourite genres and then looks at how positive the overall community sentiment is to recommend the best games in those categories.
            </p>
            <p>
                Before we can trust the recommender we need to measure how good its suggestions are. The NDCG at 10 score measures whether the one game we know a user actually liked appeared near the very top of their recommendation list. The MAP at 10 score measures exactly the same thing but it averages the position across all the users we tested. The distribution chart below shows how these scores are spread out across different users in the test group. You can see that some users get a perfect score of one because the system guessed exactly what they wanted while others get a zero because their unique preferences were just too hard to predict.
            </p>
            <table class="metrics-table">
                <tr>
                    <th>Metric</th>
                    <th>Score</th>
                </tr>
                <tr>
                    <td>NDCG@10</td>
                    <td>{{ "%.4f"|format(results.NDCG_10) }}</td>
                </tr>
                <tr>
                    <td>MAP@10</td>
                    <td>{{ "%.4f"|format(results.MAP_10) }}</td>
                </tr>
            </table>
            
            <div class="plot-container">
                <img src="{{ url_for('static', filename='evaluation_metrics.png') }}" alt="Evaluation Metrics">
                <div class="caption">Figure 1: NDCG and MAP scores alongside a histogram showing how individual user scores vary across the test set.</div>
            </div>
        </div>

        <!-- XAI SECTION -->
        <div id="xai" class="section card">
            <h2>Explainable AI Analysis</h2>
            <p>
                This section looks at explainable artificial intelligence which helps us understand why the model made a specific recommendation. Without these tools the machine learning model is basically a black box where we put data in and get answers out without knowing how it got there. I used two different tools called SHAP and LIME to crack open this black box. SHAP gives us a big picture view of how the model thinks overall while LIME zooms in to explain individual game recommendations.
            </p>
            
            <h3>Global Explanations</h3>
            <p>
                The first tool I used is SHAP which helps us see the global picture of what the model cares about. This bar chart shows which game features matter most on average across all the predictions the model makes. I found it really interesting that being an Open World game and the price of the game came out as the absolute strongest signals. This makes sense because players generally have strong feelings about paying full price and whether they want a massive game world to explore.
            </p>
            <div class="plot-container">
                <img src="{{ url_for('static', filename='shap_bar.png') }}" alt="SHAP Bar Chart">
                <div class="caption">Figure 2: Global feature importance calculated by SHAP showing Open World as the strongest predictor.</div>
            </div>

            <p>
                This next chart is a SHAP beeswarm plot which gives us even more detail about how these features affect the score. Every single dot you see on this chart represents one prediction made for a user. The colour of the dot shows whether the actual feature value was high or low for that specific game. When looking at price you can see a really interesting non linear relationship where both very cheap and very expensive games behave differently than average priced ones.
            </p>
            <div class="plot-container">
                <img src="{{ url_for('static', filename='shap_beeswarm.png') }}" alt="SHAP Beeswarm">
                <div class="caption">Figure 3: Beeswarm plot illustrating how the high or low value of a feature impacts the final recommendation score.</div>
            </div>

            <h3>Local Explanations</h3>
            <p>
                Now we move on to LIME which takes just one specific game recommendation and breaks down exactly why it happened. This first example looks at a game that the model was very confident the user would enjoy playing. In this case the fact that the game was not in early access pushed the score up significantly. This tells us that this particular user probably prefers finished games rather than buying into betas or incomplete projects.
            </p>
            <div class="plot-container">
                <img src="{{ url_for('static', filename='lime_recommended.png') }}" alt="LIME Recommended Instance">
                <div class="caption">Figure 4: A LIME explanation for a game that the system successfully recommended to a user.</div>
            </div>

            <p>
                This second LIME example shows the exact opposite situation where the model decided not to recommend a game. Here we can see what features dragged the score down and made the model think it was a bad match. It turns out that the game being Free to Play was present and that pushed the score down heavily. This suggests the user tends to avoid free games and might prefer premium titles without microtransactions.
            </p>
            <div class="plot-container">
                <img src="{{ url_for('static', filename='lime_not_recommended.png') }}" alt="LIME Not Recommended Instance">
                <div class="caption">Figure 5: A LIME explanation for a game that the system actively decided not to recommend.</div>
            </div>

            <p>
                One issue with LIME is that it works by generating random samples to test the model and this can sometimes give unstable answers. I ran a stability check to see what happens when we run LIME with a very small sample size. Running it with too few samples gives completely unreliable results where the feature importance jumps all over the place. Using five hundred samples proved to be the safe minimum to get a consistent and trustworthy explanation every time.
            </p>
            <div class="plot-container">
                <img src="{{ url_for('static', filename='lime_stability.png') }}" alt="LIME Stability">
                <div class="caption">Figure 6: A stability test comparing LIME outputs generated using three different sample sizes.</div>
            </div>

            <p>
                Finally I wanted to compare how SHAP and LIME see the world side by side. The chart compares the global feature rankings from SHAP with the local rankings from one single LIME instance. While both tools generally agreed on what the most important features were overall they disagree completely at the instance level. This happens because LIME only looks at one specific game for one specific person while SHAP averages everything out across the entire dataset.
            </p>
            <div class="plot-container">
                <img src="{{ url_for('static', filename='shap_vs_lime.png') }}" alt="SHAP vs LIME">
                <div class="caption">Figure 7: A direct comparison between the global average from SHAP and a single local explanation from LIME.</div>
            </div>
        </div>

        <!-- BIAS SECTION -->
        <div id="bias" class="section card">
            <h2>Bias and Fairness</h2>
            <p>
                The final part of my project looks at something called popularity bias which is a really common problem in machine learning. Popularity bias happens when the model tends to recommend the exact same well known games over and over again to everyone. It does this because those huge games have way more data to learn from compared to smaller obscure titles. This means that niche games never get recommended and users just see the same top sellers they already know about.
            </p>

            <p>
                To see how bad this bias actually is I compared what games exist in the dataset against what users actually interact with. The chart shows the total catalogue of games on the left versus the sheer volume of interactions on the right. You can clearly see that Indie games make up a massive chunk of the available games but they get very little attention from players. Meanwhile a tiny number of massive games are eating up almost all the player interactions.
            </p>
            <div class="plot-container">
                <img src="{{ url_for('static', filename='popularity_bias.png') }}" alt="Popularity Bias">
                <div class="caption">Figure 8: A comparison showing the imbalance between the number of games available and the number of interactions they receive.</div>
            </div>

            <p>
                This next chart shows how the bias affects the actual recommendations the system hands out to players. It compares the true share of genres in the catalogue against the share of genres that actually show up in recommendation lists. Action games appear far more often in recommendations than their catalogue share would suggest is fair. This proves the model is playing it safe by just pushing popular action titles instead of exploring the full catalogue.
            </p>
            <div class="plot-container">
                <img src="{{ url_for('static', filename='genre_share_bias.png') }}" alt="Genre Share Bias">
                <div class="caption">Figure 9: The true percentage of games by genre versus how often those genres are actually recommended by the system.</div>
            </div>

            <h3>Mitigation 1: Inverse Popularity Penalty</h3>
            <p>
                To fix this I tried a mitigation strategy where I penalise games for being too popular. I used a parameter called beta to control exactly how strong this penalty should be. Lowering the beta value gives less popular games a much fairer chance of showing up in the recommendations. However we have to be careful because pushing too many unknown games risks recommending things the user might not actually enjoy playing.
            </p>
            <div class="plot-container">
                <img src="{{ url_for('static', filename='beta_sweep.png') }}" alt="Beta Sweep for Popularity Penalty">
                <div class="caption">Figure 10: How the average popularity of recommended games drops as the beta penalty gets stronger.</div>
            </div>

            <h3>Mitigation 2: Diversity Constraint</h3>
            <p>
                My second idea to fix the bias was to force the system to pick games from a wider variety of genres. I measured how well this worked using something called Shannon entropy. Higher entropy just means there is much more variety and less repetition in the final recommendation list. The chart shows that applying this diversity constraint produced a small but very real improvement in how varied the suggestions were for most users.
            </p>
            <div class="plot-container">
                <img src="{{ url_for('static', filename='entropy_comparison.png') }}" alt="Entropy Comparison">
                <div class="caption">Figure 11: A histogram showing the increase in genre diversity after turning on the diversity constraint.</div>
            </div>
            
            <table class="metrics-table">
                <tr>
                    <th>Configuration</th>
                    <th>Average Genre Entropy</th>
                </tr>
                <tr>
                    <td>Standard Recommender</td>
                    <td>{{ "%.3f"|format(results.bias.mean_std_entropy) }}</td>
                </tr>
                <tr>
                    <td>Diversified Recommender</td>
                    <td>{{ "%.3f"|format(results.bias.mean_div_entropy) }}</td>
                </tr>
            </table>
        </div>
    </div>
</body>
</html>
"""

@app.route('/')
def index():
    if os.path.exists('results.json'):
        with open('results.json', 'r') as f:
            results = json.load(f)
    else:
        results = {"NDCG_10": 0.0, "MAP_10": 0.0, "bias": {"mean_std_entropy": 0.0, "mean_div_entropy": 0.0}}
    
    return render_template_string(HTML_TEMPLATE, results=results)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
