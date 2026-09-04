import { scaleLinear } from 'd3-scale';
import { makeSvg, svgEl, text, fmt4 } from './helpers';
import { reducedMotion } from '../lib/motion';

/**
 * The protocol comparison, drawn as a slope rather than two bars.
 *
 * The finding is "this number fell by 3.9x when measured honestly". A slope
 * reads as a fall; paired bars read as two unrelated facts.
 */
export function slopeChart(
  host: HTMLElement,
  left: { label: string; value: number; note: string },
  right: { label: string; value: number; note: string },
): void {
  const W = 700, H = 380;
  const m = { top: 48, right: 180, bottom: 52, left: 180 };
  const svg = makeSvg(W, H);
  svg.setAttribute('aria-label',
    `${left.label} ${fmt4(left.value)} versus ${right.label} ${fmt4(right.value)}`);

  const y = scaleLinear()
    .domain([0, Math.max(left.value, right.value) * 1.2])
    .range([H - m.bottom, m.top]);

  const xL = m.left, xR = W - m.right;

  for (const [x, d, anchor] of [[xL, left, 'end'], [xR, right, 'start']] as const) {
    svg.appendChild(svgEl('line', {
      x1: x, x2: x, y1: m.top - 16, y2: H - m.bottom, stroke: 'var(--border)', 'stroke-width': 1,
    }));
    const cap = text(d.label, {
      x, y: m.top - 26, 'text-anchor': 'middle', fill: 'var(--text-muted)',
    });
    cap.style.fontSize = '13px';
    svg.appendChild(cap);

    const note = text(d.note, {
      x, y: H - m.bottom + 22, 'text-anchor': 'middle', fill: 'var(--text-faint)',
    });
    note.style.fontSize = '11.5px';
    svg.appendChild(note);

    const colour = d === left ? 'var(--warn)' : 'var(--accent-hi)';
    svg.appendChild(svgEl('circle', { cx: x, cy: y(d.value), r: 7, fill: colour }));

    const val = text(fmt4(d.value), {
      x: anchor === 'end' ? x - 16 : x + 16, y: y(d.value) + 5,
      'text-anchor': anchor, fill: 'var(--text-primary)',
    });
    val.style.fontSize = '19px';
    val.style.fontFamily = 'var(--font-mono)';
    svg.appendChild(val);
  }

  const line = svgEl('line', {
    x1: xL, y1: y(left.value), x2: xR, y2: y(right.value),
    stroke: 'var(--warn)', 'stroke-width': 2.5, 'stroke-linecap': 'round', opacity: 0.85,
  });
  if (!reducedMotion()) {
    const len = Math.hypot(xR - xL, y(right.value) - y(left.value));
    line.style.strokeDasharray = String(len);
    line.style.strokeDashoffset = String(len);
    line.style.transition = 'stroke-dashoffset 1000ms var(--ease-out-expo)';
    requestAnimationFrame(() => { line.style.strokeDashoffset = '0'; });
  }
  svg.appendChild(line);

  const ratio = left.value / Math.max(right.value, 1e-9);
  const badge = text(`${ratio.toFixed(1)}× inflation`, {
    x: (xL + xR) / 2, y: (y(left.value) + y(right.value)) / 2 - 16,
    'text-anchor': 'middle', fill: 'var(--warn)',
  });
  badge.style.fontSize = '14px';
  badge.style.fontWeight = '600';
  svg.appendChild(badge);

  host.replaceChildren(svg);
}
