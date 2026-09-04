import { api, type RecommendResponse } from '../lib/api';
import { observeReveals, reducedMotion } from '../lib/motion';

function recCard(r: RecommendResponse['recommendations'][number], i: number): string {
  const max = Math.max(1, ...r.influences.map(f => f.rank_delta));
  const bars = r.influences.map(f => `
    <div class="rec__bar">
      <span>${f.name}</span>
      <span class="rec__track"><span class="rec__fill" data-w="${f.rank_delta / max}"></span></span>
    </div>`).join('');

  return `
    <article class="rec reveal">
      <div class="rec__rank">${String(i + 1).padStart(2, '0')}</div>
      <div>
        <div class="rec__name">${r.name}</div>
        ${r.genres.length ? `<div class="rec__genres">${r.genres.map(g =>
          `<span class="rec__genre">${g}</span>`).join('')}</div>` : ''}
        <div class="rec__why">${r.explanation}</div>
        ${bars ? `<div class="rec__bars">${bars}</div>` : ''}
      </div>
    </article>`;
}

export async function renderRecommend(host: HTMLElement): Promise<void> {
  host.innerHTML = `
    <div class="wrap">
      <section class="section" style="border-top:0;padding-top:var(--s8)">
        <div class="section__head">
          <div class="section__kicker">Live</div>
          <h2>Recommendations, computed now</h2>
          <p class="lede" style="margin-top:var(--s3)">
            Scored at request time from the trained factors. Each result names the games in
            this player's own library that hold it up. Remove those and the model suggests
            something else.
          </p>
        </div>
        <div class="card">
          <form id="form" style="display:flex;gap:var(--s3);flex-wrap:wrap;align-items:center">
            <input class="input" id="uid" placeholder="Steam user id" aria-label="Steam user id">
            <button class="btn btn--primary" type="submit">Recommend</button>
          </form>
          <div style="margin-top:var(--s4)">
            <div class="micro">Pick a player</div>
            <div class="chips" id="samples"></div>
          </div>
        </div>
        <div id="out" style="margin-top:var(--s6)"></div>
      </section>
    </div>`;

  const out = host.querySelector<HTMLElement>('#out')!;
  const input = host.querySelector<HTMLInputElement>('#uid')!;

  try {
    const { users } = await api.users();
    host.querySelector<HTMLElement>('#samples')!.innerHTML = users.map(u =>
      `<button class="player" data-uid="${u.id}" title="Steam id ${u.id}">
         <span class="player__name">${u.display}</span>
         <span class="player__taste">${u.taste || 'mixed library'}</span>
         <span class="player__count">${u.history_size} games</span>
       </button>`).join('');
    host.querySelectorAll<HTMLButtonElement>('[data-uid]').forEach(b =>
      b.addEventListener('click', () => { input.value = b.dataset.uid!; void load(b.dataset.uid!); }));
  } catch {
    host.querySelector<HTMLElement>('#samples')!.innerHTML =
      '<span class="error">Model not loaded. Run python train.py first.</span>';
  }

  async function load(id: string) {
    out.innerHTML = '<div class="empty">Scoring and explaining…</div>';
    let data: RecommendResponse;
    try {
      data = await api.recommend(id);
    } catch (e) {
      out.innerHTML = `<div class="empty error">${(e as Error).message}</div>`;
      return;
    }

    out.innerHTML = `
      <div style="display:flex;justify-content:space-between;align-items:baseline;flex-wrap:wrap;gap:var(--s3);margin-bottom:var(--s4)">
        <h3>Top ${data.recommendations.length} for <span style="color:var(--amber)">${data.display ?? data.user_id}</span></h3>
        <span class="micro">${data.history.length} games in library</span>
      </div>
      ${data.recommendations.map(recCard).join('')}
      <div class="card" style="margin-top:var(--s5)">
        <div class="micro">Library</div>
        <div class="chips">${data.history.map(h => `<span class="chip">${h.name}</span>`).join('')}</div>
      </div>`;

    observeReveals(out);
    // Influence bars grow after paint so the width transition actually runs.
    requestAnimationFrame(() => {
      out.querySelectorAll<HTMLElement>('.rec__fill').forEach(el => {
        const w = Number(el.dataset.w ?? 0);
        el.style.transform = reducedMotion() ? `scaleX(${w})` : `scaleX(${w})`;
      });
    });
  }

  host.querySelector<HTMLFormElement>('#form')!.addEventListener('submit', e => {
    e.preventDefault();
    if (input.value.trim()) void load(input.value.trim());
  });
}
