import { scaleLinear, scaleBand } from 'd3-scale';
import { makeSvg, svgEl, yGrid, text, cssVar, fmt4 } from './helpers';
import { seriesColor } from '../lib/api';
import { tooltip, reducedMotion } from '../lib/motion';

export interface BarDatum { model: string; value: number; ci: number }

/**
 * Grouped model comparison with real confidence whiskers.
 *
 * Re-rendering with the other split animates every bar to its new height, which
 * shows the cost of random splitting far better than two tables side by side.
 */
export function barChart(host: HTMLElement, data: BarDatum[], yLabel = 'NDCG@10'): void {
  const W = 760, H = 400;
  const m = { top: 20, right: 20, bottom: 62, left: 66 };
  const svg = makeSvg(W, H);
  svg.setAttribute('aria-label', `${yLabel} by model, with 95% confidence intervals`);

  const x = scaleBand<string>().domain(data.map(d => d.model))
    .range([m.left, W - m.right]).padding(0.28);
  const yMax = Math.max(...data.map(d => d.value + d.ci)) * 1.16;
  const y = scaleLinear().domain([0, yMax]).range([H - m.bottom, m.top]);

  yGrid(svg, y.ticks(5), y, m.left, W - m.right, fmt4);
  svg.appendChild(text(yLabel,
    { x: -(H - m.bottom + m.top) / 2, y: 16, 'text-anchor': 'middle', transform: 'rotate(-90)' },
    'axis-label'));

  const tip = tooltip();

  data.forEach((d, i) => {
    const bx = x(d.model)!;
    const bw = x.bandwidth();
    const colour = cssVar(seriesColor(d.model));
    const g = svgEl('g');

    const bar = svgEl('rect', {
      x: bx, y: y(d.value), width: bw, height: Math.max(0, H - m.bottom - y(d.value)),
      fill: colour, rx: 5, class: 'bar',
    });

    if (!reducedMotion()) {
      bar.style.transformOrigin = `0 ${H - m.bottom}px`;
      bar.style.transform = 'scaleY(0)';
      bar.style.transition = `transform 800ms var(--ease-out-expo) ${i * 40}ms`;
      requestAnimationFrame(() => { bar.style.transform = 'scaleY(1)'; });
    }
    g.appendChild(bar);

    // Whiskers only where an interval actually exists -- deterministic models
    // get none rather than a fake zero-width tick.
    if (d.ci > 0) {
      const cx = bx + bw / 2;
      const w = svgEl('path', {
        d: `M${cx},${y(d.value - d.ci)} L${cx},${y(d.value + d.ci)}
            M${cx - 5},${y(d.value + d.ci)} L${cx + 5},${y(d.value + d.ci)}
            M${cx - 5},${y(d.value - d.ci)} L${cx + 5},${y(d.value - d.ci)}`,
        stroke: 'var(--text-primary)', 'stroke-width': 1.4, opacity: 0.65, fill: 'none',
      });
      g.appendChild(w);
    }

    const label = text(d.model, {
      x: bx + bw / 2, y: H - m.bottom + 20, 'text-anchor': 'middle',
    });
    label.style.fontSize = '12px';
    g.appendChild(label);

    const val = text(fmt4(d.value), {
      x: bx + bw / 2, y: y(d.value + d.ci) - 8, 'text-anchor': 'middle',
      fill: 'var(--text-muted)',
    });
    val.style.fontSize = '11.5px';
    g.appendChild(val);

    const enter = (e: PointerEvent) => tip.show(
      `<strong>${d.model}</strong><br><span class="mono">${fmt4(d.value)}` +
      (d.ci > 0 ? ` ± ${d.ci.toFixed(4)}` : ' (deterministic)') + '</span>',
      e.clientX, e.clientY,
    );
    g.addEventListener('pointerenter', enter);
    g.addEventListener('pointermove', enter);
    g.addEventListener('pointerleave', () => tip.hide());

    svg.appendChild(g);
  });

  host.replaceChildren(svg);
}
