import { scaleLinear } from 'd3-scale';
import { line as d3line } from 'd3-shape';
import { makeSvg, svgEl, text, fmt4 } from './helpers';
import { reducedMotion } from '../lib/motion';

export interface SweepPoint {
  lambda: number; 'NDCG@10': number; miscalibration: number; coverage: number;
}

/**
 * The accuracy/calibration trade-off with a draggable lambda.
 *
 * Steck's point is that calibration is a dial, not a setting, so the control is
 * the chart: dragging lambda moves a marker along both curves at once and
 * reports the pair of values it costs and buys.
 */
export function calibrationChart(
  host: HTMLElement, sweep: SweepPoint[], onChange?: (p: SweepPoint) => void,
): void {
  const W = 760, H = 420;
  const m = { top: 26, right: 66, bottom: 62, left: 66 };
  const svg = makeSvg(W, H);
  svg.setAttribute('aria-label', 'NDCG and KL miscalibration against lambda');

  const x = scaleLinear().domain([0, 1]).range([m.left, W - m.right]);
  const yA = scaleLinear().domain([0, Math.max(...sweep.map(s => s['NDCG@10'])) * 1.25])
    .range([H - m.bottom, m.top]);
  const yB = scaleLinear().domain([0, Math.max(...sweep.map(s => s.miscalibration)) * 1.15])
    .range([H - m.bottom, m.top]);

  for (const t of yA.ticks(5)) {
    svg.appendChild(svgEl('line', {
      x1: m.left, x2: W - m.right, y1: yA(t), y2: yA(t), class: 'grid-line', 'stroke-width': 1,
    }));
    svg.appendChild(text(fmt4(t), { x: m.left - 8, y: yA(t) + 4, 'text-anchor': 'end' }));
  }
  for (const t of yB.ticks(5)) {
    svg.appendChild(text(t.toFixed(2), {
      x: W - m.right + 8, y: yB(t) + 4, 'text-anchor': 'start', fill: 'var(--warn)',
    }));
  }
  for (const t of x.ticks(5)) {
    svg.appendChild(text(t.toFixed(2), { x: x(t), y: H - m.bottom + 20, 'text-anchor': 'middle' }));
  }

  svg.appendChild(text('λ   0 = pure relevance  →  1 = pure calibration',
    { x: (m.left + W - m.right) / 2, y: H - 16, 'text-anchor': 'middle' }, 'axis-label'));

  const mk = (yScale: typeof yA, key: 'NDCG@10' | 'miscalibration', colour: string) => {
    const path = d3line<SweepPoint>()
      .x(d => x(d.lambda)).y(d => yScale(d[key]))(sweep)!;
    const el = svgEl('path', {
      d: path, fill: 'none', stroke: colour, 'stroke-width': 2.5,
      'stroke-linecap': 'round', 'stroke-linejoin': 'round',
    });
    if (!reducedMotion()) {
      const len = (el as SVGPathElement).getTotalLength?.() ?? 800;
      el.style.strokeDasharray = String(len);
      el.style.strokeDashoffset = String(len);
      el.style.transition = 'stroke-dashoffset 1000ms var(--ease-out-expo)';
      requestAnimationFrame(() => { el.style.strokeDashoffset = '0'; });
    }
    svg.appendChild(el);
    sweep.forEach(s => svg.appendChild(svgEl('circle', {
      cx: x(s.lambda), cy: yScale(s[key]), r: 3.5, fill: colour,
    })));
  };

  mk(yA, 'NDCG@10', 'var(--accent-hi)');
  mk(yB, 'miscalibration', 'var(--warn)');

  const marker = svgEl('line', {
    x1: x(0), x2: x(0), y1: m.top, y2: H - m.bottom,
    stroke: 'var(--text-primary)', 'stroke-width': 1, opacity: 0.35,
    'stroke-dasharray': '3 3',
  });
  svg.appendChild(marker);

  host.replaceChildren(svg);

  // Native range input: keyboard-operable for free, which a custom drag handle
  // would have to reimplement badly.
  const controls = document.createElement('div');
  controls.style.cssText = 'display:flex;align-items:center;gap:var(--s4);margin-top:var(--s4);flex-wrap:wrap';
  const slider = document.createElement('input');
  slider.type = 'range';
  slider.min = '0';
  slider.max = String(sweep.length - 1);
  slider.step = '1';
  slider.value = '0';
  slider.setAttribute('aria-label', 'Calibration weight lambda');
  slider.style.cssText = 'flex:1;min-width:200px;accent-color:var(--accent)';

  const readout = document.createElement('div');
  readout.className = 'mono';
  readout.style.cssText = 'font-variant-numeric:tabular-nums;color:var(--text-muted);font-size:13px';

  const apply = () => {
    const p = sweep[Number(slider.value)];
    marker.setAttribute('x1', String(x(p.lambda)));
    marker.setAttribute('x2', String(x(p.lambda)));
    readout.innerHTML =
      `λ ${p.lambda.toFixed(2)} · NDCG <span style="color:var(--accent-hi)">${fmt4(p['NDCG@10'])}</span>` +
      ` · KL <span style="color:var(--warn)">${p.miscalibration.toFixed(4)}</span>`;
    onChange?.(p);
  };
  slider.addEventListener('input', apply);
  apply();

  controls.append(slider, readout);
  host.appendChild(controls);

  const legend = document.createElement('div');
  legend.className = 'legend';
  legend.innerHTML =
    `<span class="legend__item"><span class="swatch" style="background:var(--accent-hi)"></span>NDCG@10 (left axis)</span>` +
    `<span class="legend__item"><span class="swatch" style="background:var(--warn)"></span>KL miscalibration (right axis)</span>`;
  host.appendChild(legend);
}
