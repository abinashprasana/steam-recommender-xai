import { api } from '../lib/api';
import { observeReveals } from '../lib/motion';

export async function renderColdStart(host: HTMLElement): Promise<void> {
  host.innerHTML = `
    <div class="wrap">
      <section class="section" style="border-top:0;padding-top:var(--s8)">
        <div class="section__head">
          <div class="section__kicker">Cold start</div>
          <h2>No history yet</h2>
          <p class="lede" style="margin-top:var(--s3)">
            A new player has no interactions, so collaborative filtering has nothing to work
            with. Pick genres and the catalogue is ranked by community sentiment instead.
          </p>
        </div>
        <div class="card">
          <div class="genre-grid" id="genres"></div>
          <div style="margin-top:var(--s5);display:flex;gap:var(--s3);align-items:center;flex-wrap:wrap">
            <button class="btn btn--primary" id="go">Show games</button>
            <button class="btn" id="clear">Clear</button>
            <span id="count" class="micro"></span>
          </div>
        </div>
        <div id="out" style="margin-top:var(--s6)"></div>
      </section>
    </div>`;

  const out = host.querySelector<HTMLElement>('#out')!;
  const countEl = host.querySelector<HTMLElement>('#count')!;
  const selected = new Set<string>();

  let genres: string[] = [];
  try {
    genres = (await api.coldStart([])).available_genres;
  } catch (e) {
    out.innerHTML = `<div class="empty error">${(e as Error).message}</div>`;
    return;
  }

  const grid = host.querySelector<HTMLElement>('#genres')!;
  grid.innerHTML = genres.map(g =>
    `<label class="genre" data-on="false">
       <input type="checkbox" value="${g}"> <span>${g}</span>
     </label>`).join('');

  const sync = () => {
    countEl.textContent = selected.size
      ? `${selected.size} selected` : 'Pick at least one genre';
  };
  sync();

  grid.querySelectorAll<HTMLInputElement>('input').forEach(cb => {
    cb.addEventListener('change', () => {
      const label = cb.closest<HTMLElement>('.genre')!;
      if (cb.checked) selected.add(cb.value); else selected.delete(cb.value);
      label.dataset.on = String(cb.checked);
      sync();
    });
  });

  host.querySelector<HTMLButtonElement>('#clear')!.addEventListener('click', () => {
    selected.clear();
    grid.querySelectorAll<HTMLInputElement>('input').forEach(cb => {
      cb.checked = false;
      cb.closest<HTMLElement>('.genre')!.dataset.on = 'false';
    });
    out.innerHTML = '';
    sync();
  });

  host.querySelector<HTMLButtonElement>('#go')!.addEventListener('click', async () => {
    if (!selected.size) { sync(); return; }
    out.innerHTML = '<div class="empty">Ranking by community sentiment…</div>';
    const data = await api.coldStart([...selected]);
    if (!data.games.length) {
      out.innerHTML = `<div class="empty">${data.message ?? 'No games matched those genres.'}</div>`;
      return;
    }
    out.innerHTML = `
      <h3 style="margin-bottom:var(--s4)">Top picks for ${[...selected].join(', ')}</h3>
      ${data.games.map((g, i) => `
        <article class="rec reveal">
          <div class="rec__rank">${String(i + 1).padStart(2, '0')}</div>
          <div>
            <div class="rec__name">${g.name}</div>
            <div class="rec__genres">${g.genres.map(x => `<span class="rec__genre">${x}</span>`).join('')}</div>
            <div class="rec__why">Community sentiment: <strong style="color:var(--success)">${g.sentiment || 'unknown'}</strong>${
              g.price ? ` · ${g.price}` : ''}</div>
          </div>
        </article>`).join('')}`;
    observeReveals(out);
  });
}
