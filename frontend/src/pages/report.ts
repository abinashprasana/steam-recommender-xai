import { api, seriesColor, type Results, type Table } from '../lib/api';
import { observeReveals, countUp, magnetic } from '../lib/motion';
import { mountConstellation } from '../hero/constellation';
import { mountCloud } from '../hero/cloud';
import { accuracyCoverageScatter } from '../charts/scatter';
import { barChart } from '../charts/bars';
import { slopeChart } from '../charts/slope';
import { calibrationChart } from '../charts/calibration';
import { flowDiagram } from '../charts/flow';
import { mountContents } from '../components/contents';

const METRICS = ['NDCG@10', 'MAP@10', 'Recall@10', 'HitRate@10', 'Coverage', 'Gini'];

function metricTable(table: Table): string {
  const rows = Object.entries(table).map(([model, row]) => {
    const cells = METRICS.map(m => {
      const cell = row[m];
      if (!cell) return '<td>n/a</td>';
      const [mean, ci] = cell;
      return `<td>${mean.toFixed(4)}${ci > 0 ? ` <span class="ci">±${ci.toFixed(3)}</span>` : ''}</td>`;
    }).join('');
    return `<tr><td><span class="swatch" style="background:${seriesColor(model)}"></span>${model}</td>${cells}</tr>`;
  }).join('');
  return `<div class="table-scroll"><table class="data">
    <thead><tr><th>Model</th>${METRICS.map(m => `<th>${m}</th>`).join('')}</tr></thead>
    <tbody>${rows}</tbody></table></div>`;
}

function toScatter(table: Table) {
  return Object.entries(table).map(([model, row]) => ({
    model,
    ndcg: row['NDCG@10']?.[0] ?? 0,
    coverage: row['Coverage']?.[0] ?? 0,
    gini: row['Gini']?.[0] ?? 0,
  }));
}

/* Sorted by score rather than left in JSON key order, so the ranking is legible
   without reading every label. */
const toBars = (table: Table) =>
  Object.entries(table)
    .map(([model, row]) => ({
      model, value: row['NDCG@10']?.[0] ?? 0, ci: row['NDCG@10']?.[1] ?? 0,
    }))
    .sort((a, b) => b.value - a.value);

const byScore = (table: Table): Table =>
  Object.fromEntries(
    Object.entries(table).sort(
      (a, b) => (b[1]['NDCG@10']?.[0] ?? 0) - (a[1]['NDCG@10']?.[0] ?? 0),
    ),
  );


/* The explanation, drawn small: three games converging on one recommendation.
   Same idea as the mark and the hero trace, at card scale, so the three read as
   one visual system rather than three separate graphics. */
function traceGlyph(supporters: number): string {
  const n = Math.max(1, Math.min(3, supporters));
  const ys = n === 1 ? [21] : n === 2 ? [11, 31] : [7, 21, 35];
  const lines = ys.map(y =>
    `<path d="M11 ${y} Q32 ${y} 47 21" fill="none" stroke="var(--amber)"
       stroke-width="1.1" opacity=".45"/>`).join('');
  const dots = ys.map(y =>
    `<circle cx="8" cy="${y}" r="2.6" fill="var(--amber)" opacity=".85"/>`).join('');
  return `<svg class="trace-glyph" viewBox="0 0 60 42" aria-hidden="true">
    ${lines}${dots}
    <circle cx="50" cy="21" r="5" fill="var(--accent-hi)"/>
  </svg>`;
}

export async function renderReport(host: HTMLElement): Promise<void> {
  host.innerHTML = `
    <section class="hero">
      <div class="hero__scrim" aria-hidden="true"></div>
      <div class="wrap hero__inner">
        <div class="hero__copy">
          <div class="hero__eyebrow"><span class="nav__dot"></span> 6,684 players · 2,758 games</div>
          <h1 class="hero__title">The best model was the one that <em>recommends almost nothing</em></h1>
          <p class="hero__sub">
            Steam recommendations, benchmarked against real baselines and explained by the
            model that actually does the ranking.
          </p>
          <div class="hero__actions">
            <a class="btn btn--primary" href="/recommend" data-route>See it recommend</a>
            <a class="btn" href="#finding">Read the finding</a>
          </div>
        </div>
        <div class="hero__stage">
          <canvas class="hero__canvas" id="hero-canvas"
                  aria-label="The game catalogue as a rotating point cloud" role="img"></canvas>
          <div class="hero__legend" id="hero-legend">
            <span class="hero__legend-hint">Drag to turn · hover a point</span>
          </div>
        </div>
      </div>
    </section>
    <div class="wrap" id="report-body"><div class="empty">Loading results…</div></div>`;

  const canvas = host.querySelector<HTMLCanvasElement>('#hero-canvas');
  const legend = host.querySelector<HTMLElement>('#hero-legend');
  const hint = legend?.querySelector<HTMLElement>('.hero__legend-hint');
  if (hint && window.matchMedia('(hover: none)').matches) {
    hint.textContent = 'Drag to turn';
  }

  const body = host.querySelector<HTMLElement>('#report-body')!;

  let r: Results;
  try {
    r = await api.results();
  } catch (e) {
    body.innerHTML = `<div class="empty error">${(e as Error).message}</div>`;
    return;
  }

  const rnd = r.random_split;
  const tmp = r.temporal_split;
  const pop = rnd?.['Popularity'];
  const hyb = rnd?.['Hybrid'];
  const bpr = rnd?.['BPR'];

  body.innerHTML = `
    <section class="section reveal" id="finding">
      <div class="section__head">
        <div class="section__kicker">The finding</div>
        <h2>A non-personalised baseline won, and that is the result</h2>
      </div>
      <p class="lede">
        Popularity recommends the same handful of games to everyone. It beat every
        personalised model here on accuracy while touching a fraction of a percent of the
        catalogue. Accuracy alone would have declared the least useful possible system the
        winner, which is why coverage and concentration sit in the same table.
      </p>
      <div class="stats">
        <div class="stat stat--accent">
          <div class="stat__value" data-count="${pop?.['NDCG@10']?.[0] ?? 0}">0.0000</div>
          <div class="stat__label">Popularity NDCG@10, the best in the table</div>
          <div class="stat__note">A model with no notion of "you"</div>
        </div>
        <div class="stat stat--warn">
          <div class="stat__value" data-count="${pop?.['Coverage']?.[0] ?? 0}" data-suffix="">0.0000</div>
          <div class="stat__label">…of the catalogue it ever recommends</div>
          <div class="stat__note">Gini ${(pop?.['Gini']?.[0] ?? 0).toFixed(3)}, near-total concentration</div>
        </div>
        <div class="stat">
          <div class="stat__value" data-count="${hyb?.['NDCG@10']?.[0] ?? 0}">0.0000</div>
          <div class="stat__label">The hybrid this project set out to build</div>
          <div class="stat__note">Below plain BPR (${(bpr?.['NDCG@10']?.[0] ?? 0).toFixed(4)}), intervals overlapping</div>
        </div>
      </div>
      <div id="scatter" class="card" style="margin-top:var(--s6)"></div>
      <p style="margin-top:var(--s4);color:var(--text-muted);font-size:14px">
        Up and to the right is good. Popularity sits at the top-left edge: accurate, and
        serving nothing. ItemKNN reaches most of the catalogue but ranks poorly. Nothing
        here reaches the corner that would matter.
      </p>
    </section>

    <section class="section reveal" id="method">
      <div class="section__head">
        <div class="section__kicker">How it works</div>
        <h2>From 59,305 reviews to an explained recommendation</h2>
      </div>
      <p class="lede">
        Every figure on this page comes out of the same pipeline. Reviews are joined to
        the catalogue, players with fewer than three are dropped, and what remains is
        split, trained, scored and explained.
      </p>
      <div id="pipeline" class="card" style="margin-top:var(--s5);overflow-x:auto"></div>
    </section>

    <section class="section reveal" id="models">
      <div class="section__head">
        <div class="section__kicker">Benchmark</div>
        <h2>Every model through one harness</h2>
      </div>
      <p class="lede">
        An earlier version reported a single NDCG figure described as the hybrid's. The
        evaluation loop was scoring plain BPR, so the content blend and both fairness
        re-rankers were never measured for accuracy at all, and no baseline existed to
        interpret the number against. Each figure below now describes the model beside it.
      </p>
      <div style="display:flex;gap:var(--s4);align-items:center;margin:var(--s5) 0">
        <div class="toggle" role="group" aria-label="Choose evaluation split">
          <button data-split="random" aria-pressed="true">Random split</button>
          <button data-split="temporal" aria-pressed="false">Temporal split</button>
        </div>
        <span id="split-note" style="font-size:13px;color:var(--text-faint)"></span>
      </div>
      <div id="bars" class="card"></div>
      <div id="table" style="margin-top:var(--s5)"></div>
    </section>

    <section class="section reveal" id="protocol">
      <div class="section__head">
        <div class="section__kicker">Where the old number came from</div>
        <h2>Ranking against 99 sampled negatives, not the catalogue</h2>
      </div>
      <p class="lede">
        Same data, same model, two protocols. Scoring each relevant item against 99 random
        negatives instead of all 2,758 games inflated the result almost fourfold. Together
        with a bug in the ideal-DCG calculation, that accounts for essentially all of the
        0.5203 this project originally reported.
      </p>
      <div id="slope" class="card" style="margin-top:var(--s5)"></div>
    </section>

    <section class="section reveal" id="explanations">
      <div class="section__head">
        <div class="section__kicker">Explanation</div>
        <h2>Why this game, for this person</h2>
      </div>
      <p class="lede">
        Explanations attribute a recommendation to the user's own history rather than to
        item attributes. Each counterfactual removes a game and <strong>re-fits that user's
        latent vector</strong> before re-scoring. Without the refit, the original vector
        still encodes the removed game and the measurement means nothing.
      </p>
      <div id="examples" class="grid" style="margin-top:var(--s5)"></div>
      <div class="note note--accent" id="faith"></div>
    </section>

    <section class="section reveal" id="calibration">
      <div class="section__head">
        <div class="section__kicker">Fairness</div>
        <h2>Calibration is a dial, not a setting</h2>
      </div>
      <p class="lede">
        If a player's history is 70% strategy and 30% racing, their recommendations should
        be roughly 70/30 too. Miscalibration is the KL divergence between those two
        distributions; λ trades it against relevance. The result is a curve, not a number.
      </p>
      <div id="calib" class="card" style="margin-top:var(--s5)"></div>
    </section>

    <section class="section reveal" id="figures">
      <div class="section__head">
        <div class="section__kicker">Catalogue analysis</div>
        <h2>What attributes go with a well-reviewed game</h2>
      </div>
      <div class="note">
        <strong>What this is not.</strong> These explain a RandomForest trained on game
        attributes with no notion of a user. It plays no part in ranking, so these describe
        the catalogue, not why anything was recommended to anyone. The
        <a href="#explanations">explanations above</a> answer that question.
      </div>
      <div class="grid grid--2" id="figs"></div>
    </section>

    <footer class="footer">
      <p style="margin:0">
        Waypoint. BPR, EASE, ItemKNN and popularity baselines, counterfactual
        explanation, Steck-2018 calibration. Figures regenerate from
        <span class="mono">python train.py</span>.
      </p>
      <p class="footnote">Abinash Prasana Selvanathan</p>
    </footer>`;

  // Charts
  if (rnd) {
    accuracyCoverageScatter(body.querySelector<HTMLElement>('#scatter')!, toScatter(rnd));
    barChart(body.querySelector<HTMLElement>('#bars')!, toBars(rnd));
    body.querySelector<HTMLElement>('#table')!.innerHTML = metricTable(byScore(rnd));
  }

  const note = body.querySelector<HTMLElement>('#split-note')!;
  note.textContent = 'A random split lets a model train on a user\'s later reviews to predict earlier ones.';
  body.querySelectorAll<HTMLButtonElement>('[data-split]').forEach(btn => {
    btn.addEventListener('click', () => {
      const which = btn.dataset.split as 'random' | 'temporal';
      const table = which === 'random' ? rnd : tmp;
      if (!table) return;
      body.querySelectorAll<HTMLButtonElement>('[data-split]').forEach(b =>
        b.setAttribute('aria-pressed', String(b === btn)));
      barChart(body.querySelector<HTMLElement>('#bars')!, toBars(table));
      body.querySelector<HTMLElement>('#table')!.innerHTML = metricTable(byScore(table));
      accuracyCoverageScatter(body.querySelector<HTMLElement>('#scatter')!, toScatter(table));
      note.textContent = which === 'temporal'
        ? 'Time-ordered: train on the past, test on what came after. Every model falls, and the personalised ones fall furthest.'
        : 'A random split lets a model train on a user\'s later reviews to predict earlier ones.';
    });
  });

  if (r.protocol) {
    slopeChart(
      body.querySelector<HTMLElement>('#slope')!,
      { label: '99 sampled negatives', value: r.protocol.sampled_99['NDCG@10'], note: 'the original protocol' },
      { label: 'Full catalogue ranking', value: r.protocol.full['NDCG@10'], note: 'all 2,758 games, train masked' },
    );
  }

  if (r.explanations) {
    const ex = body.querySelector<HTMLElement>('#examples')!;
    ex.innerHTML = r.explanations.examples.slice(0, 5).map(e =>
      `<div class="explain">
         ${traceGlyph(e.supporter_idx?.length ?? e.supporters?.length ?? 3)}
         <p class="explain__text">${e.text}</p>
       </div>`).join('');
    const ci = r.explanations.faithfulness_ci;
    body.querySelector<HTMLElement>('#faith')!.innerHTML =
      `<strong>Faithfulness ${(r.explanations.mean_faithfulness * 100).toFixed(1)}%` +
      `${ci ? ` ± ${(ci * 100).toFixed(1)}` : ''}</strong> across ${r.explanations.n_checked}
       explanations. Each claim was carried out by removing the named games, re-fitting the
       user, and recomputing the ranking. This is the share where the predicted change
       actually happened. Roughly half hold, which is the honest figure rather than a
       flattering one.`;
  }

  if (r.calibration_sweep?.length) {
    calibrationChart(body.querySelector<HTMLElement>('#calib')!, r.calibration_sweep);
  }

  const figs: Array<[string, string]> = [
    ['shap_bar.png', 'Attributes that most influence the attribute-only classifier.'],
    ['shap_beeswarm.png', 'Direction and strength of each attribute.'],
    ['popularity_bias.png', 'Games available per genre versus the interactions they attract.'],
    ['genre_share_bias.png', 'Catalogue genre share versus recommended genre share.'],
  ];
  body.querySelector<HTMLElement>('#figs')!.innerHTML = figs.map(([f, cap]) =>
    `<figure class="figure"><img src="/static/${f}" alt="${cap}" loading="lazy">
     <figcaption>${cap}</figcaption></figure>`).join('');

  body.querySelectorAll<HTMLImageElement>('.figure img').forEach(img => {
    img.addEventListener('click', () => {
      const box = document.createElement('div');
      box.className = 'lightbox';
      box.innerHTML = `<img src="${img.src}" alt="${img.alt}">`;
      box.addEventListener('click', () => box.remove());
      document.addEventListener('keydown', function esc(e) {
        if (e.key === 'Escape') { box.remove(); document.removeEventListener('keydown', esc); }
      });
      document.body.appendChild(box);
    });
  });

  body.querySelectorAll<HTMLElement>('[data-count]').forEach(el => {
    countUp(el, Number(el.dataset.count), 4, el.dataset.suffix ?? '');
  });
  // The hero traces a real explanation: the games that hold a recommendation up,
  // highlighted in the actual factor space. Falls back to the 2D field where
  // WebGL2 is unavailable.
  if (canvas) {
    const ex = r.explanations?.examples?.find(e => typeof e.target_idx === 'number');
    const trace = ex && typeof ex.target_idx === 'number'
      ? { target: ex.target_idx, supporters: ex.supporter_idx ?? [] }
      : null;
    const handle = await mountCloud(canvas, trace, name => {
      if (!legend) return;
      const label = legend.querySelector<HTMLElement>('.hero__legend-name');
      if (name) {
        if (label) label.textContent = name;
        else legend.insertAdjacentHTML('afterbegin',
          `<span class="hero__legend-name">${name}</span>`);
      } else label?.remove();
    });
    if (!handle) void mountConstellation(canvas);
    if (trace && ex) {
      legend?.insertAdjacentHTML('beforeend',
        `<span class="hero__legend-trace">
           <b>${ex.target}</b> traced back to ${(ex.supporter_idx ?? []).length} games in one player's library
         </span>`);
    }
  }

  // Weights are the real surviving volume, so the spine taper and the share of
  // particles that fall away are both driven by the data rather than chosen.
  flowDiagram(body.querySelector<HTMLElement>('#pipeline')!, [
    { value: '59,305', label: 'Steam reviews', weight: 1.0,
      detail: 'Every review in the Australian user set, before anything is filtered.' },
    { value: '31,799', label: 'interactions kept', weight: 0.536,
      detail: '27,506 rows fall away here: reviews of games missing from the catalogue, and players with fewer than three reviews. Sparse users cannot be evaluated fairly.' },
    { value: '6,684 × 2,758', label: 'players × games', weight: 0.5,
      detail: 'What survives is a 0.17% dense matrix. That sparsity is why a popularity baseline is so hard to beat.' },
    { value: '64', label: 'latent factors', weight: 0.46,
      detail: 'BPR learns a 64-dimension vector per player and per game. The hero above is these vectors projected to three dimensions.' },
    { value: '7', label: 'models, one harness', weight: 0.42,
      detail: 'Popularity, ItemKNN, EASE, BPR, content, hybrid and calibrated, all scored through identical code on two splits and three seeds.' },
    { value: '150', label: 'explanations verified', weight: 0.38,
      detail: 'Each counterfactual claim is carried out: the named games removed, the player re-fit, the ranking recomputed, and the outcome checked.' },
  ]);

  body.querySelectorAll<HTMLElement>('.card').forEach(c => magnetic(c, 2));
  observeReveals(body);
  mountContents(body);
}
