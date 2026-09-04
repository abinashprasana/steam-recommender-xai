import { scaleLinear } from 'd3-scale';
import { makeSvg, svgEl, yGrid, text, cssVar, fmt4, pct } from './helpers';
import { seriesColor } from '../lib/api';
import { tooltip, reducedMotion } from '../lib/motion';

export interface ScatterPoint { model: string; ndcg: number; coverage: number; gini: number }

/**
 * Accuracy against catalogue coverage.
 *
 * The most important chart in the project and the one the old dashboard had no
 * equivalent of: it puts Popularity's win and its uselessness in the same frame.
 * Top-left means "accurate, and serving almost nothing".
 */
export function accuracyCoverageScatter(host: HTMLElement, data: ScatterPoint[]): void {
  const W = 760, H = 460;
  const m = { top: 24, right: 28, bottom: 54, left: 62 };
  const svg = makeSvg(W, H);
  svg.setAttribute('aria-label',
    'Ranking accuracy against catalogue coverage for every model');

  const x = scaleLinear().domain([0, Math.max(0.8, ...data.map(d => d.coverage)) * 1.08])
    .range([m.left, W - m.right]);
  const y = scaleLinear().domain([0, Math.max(...data.map(d => d.ndcg)) * 1.18])
    .range([H - m.bottom, m.top]);

  yGrid(svg, y.ticks(5), y, m.left, W - m.right, fmt4);

  for (const t of x.ticks(6)) {
    svg.appendChild(text(pct(t), { x: x(t), y: H - m.bottom + 20, 'text-anchor': 'middle' }));
  }

  svg.appendChild(text('Catalogue coverage  →  share of games ever recommended',
    { x: (m.left + W - m.right) / 2, y: H - 14, 'text-anchor': 'middle' }, 'axis-label'));
  svg.appendChild(text('NDCG@10',
    { x: -(H - m.bottom + m.top) / 2, y: 16, 'text-anchor': 'middle',
      transform: 'rotate(-90)' }, 'axis-label'));

  // A quiet guide marking the corner nobody wants to sit in.
  const warn = svgEl('rect', {
    x: m.left, y: m.top, width: x(0.08) - m.left, height: (H - m.bottom) - m.top,
    fill: 'rgba(232,168,56,0.05)',
  });
  svg.appendChild(warn);
  svg.appendChild(text('narrow catalogue', { x: m.left + 8, y: m.top + 16 }));

  const tip = tooltip();

  /* BPR and Hybrid land within a few pixels of each other, so their labels
     printed on top of one another. Place each label above its point, and if
     that box overlaps one already placed, step it further away until it clears.
     Cheap, deterministic, and it keeps every model readable. */
  const placed: Array<{ x1: number; x2: number; y1: number; y2: number }> = [];
  const labelY = (cx: number, cy: number, label: string): number => {
    const halfW = label.length * 3.6 + 6;
    for (const dy of [-22, -34, 16, -46, 28, -58]) {
      const box = { x1: cx - halfW, x2: cx + halfW, y1: cy + dy - 11, y2: cy + dy + 3 };
      const hits = placed.some(p =>
        box.x1 < p.x2 && box.x2 > p.x1 && box.y1 < p.y2 && box.y2 > p.y1);
      if (!hits) { placed.push(box); return cy + dy; }
    }
    return cy - 22;
  };

  data.forEach((d, i) => {
    const g = svgEl('g');
    const colour = cssVar(seriesColor(d.model));

    const halo = svgEl('circle', {
      cx: x(d.coverage), cy: y(d.ndcg), r: 16, fill: colour, opacity: 0.14,
    });
    const dot = svgEl('circle', {
      cx: x(d.coverage), cy: y(d.ndcg), r: 6.5, fill: colour,
      stroke: 'var(--bg-void)', 'stroke-width': 2,
    });
    const label = text(d.model, {
      x: x(d.coverage), y: labelY(x(d.coverage), y(d.ndcg), d.model),
      'text-anchor': 'middle', fill: 'var(--text-body)',
    });
    label.style.fontSize = '12.5px';

    g.append(halo, dot, label);

    if (!reducedMotion()) {
      g.style.opacity = '0';
      g.style.transition = `opacity 500ms var(--ease-out-expo) ${i * 70}ms`;
      requestAnimationFrame(() => { g.style.opacity = '1'; });
    }

    const enter = (e: PointerEvent) => {
      dot.setAttribute('r', '9');
      tip.show(
        `<strong>${d.model}</strong><br>` +
        `<span class="mono">NDCG ${fmt4(d.ndcg)} · coverage ${pct(d.coverage)} · Gini ${d.gini.toFixed(3)}</span>`,
        e.clientX, e.clientY,
      );
    };
    g.addEventListener('pointerenter', enter);
    g.addEventListener('pointermove', enter);
    g.addEventListener('pointerleave', () => { dot.setAttribute('r', '6.5'); tip.hide(); });

    svg.appendChild(g);
  });

  host.replaceChildren(svg);
}
