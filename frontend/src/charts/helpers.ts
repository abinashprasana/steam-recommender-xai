export const NS = 'http://www.w3.org/2000/svg';

export function svgEl<K extends keyof SVGElementTagNameMap>(
  name: K, attrs: Record<string, string | number> = {},
): SVGElementTagNameMap[K] {
  const el = document.createElementNS(NS, name);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, String(v));
  return el;
}

export function makeSvg(width: number, height: number): SVGSVGElement {
  const svg = svgEl('svg', {
    viewBox: `0 0 ${width} ${height}`,
    class: 'chart',
    preserveAspectRatio: 'xMidYMid meet',
    role: 'img',
  });
  svg.style.width = '100%';
  svg.style.height = 'auto';
  return svg;
}

/** Horizontal gridlines plus the y axis ticks, drawn first so marks sit above. */
export function yGrid(
  svg: SVGSVGElement, ticks: number[], y: (v: number) => number,
  x0: number, x1: number, fmt: (v: number) => string = String,
): void {
  for (const t of ticks) {
    svg.appendChild(svgEl('line', {
      x1: x0, x2: x1, y1: y(t), y2: y(t), class: 'grid-line', 'stroke-width': 1,
    }));
    const label = svgEl('text', { x: x0 - 8, y: y(t) + 4, 'text-anchor': 'end' });
    label.textContent = fmt(t);
    svg.appendChild(label);
  }
}

export function text(
  content: string, attrs: Record<string, string | number>, cls = '',
): SVGTextElement {
  const t = svgEl('text', attrs);
  if (cls) t.setAttribute('class', cls);
  t.textContent = content;
  return t;
}

/** Resolves a CSS custom property to a real colour so canvas/SVG gradients and
 *  JS-set fills match the stylesheet instead of drifting from it. */
export function cssVar(name: string): string {
  const raw = name.startsWith('var(') ? name.slice(4, -1) : name;
  if (!raw.startsWith('--')) return name;
  return getComputedStyle(document.documentElement).getPropertyValue(raw).trim() || '#8fa2b4';
}

export const fmt4 = (v: number) => v.toFixed(4);
export const fmt2 = (v: number) => v.toFixed(2);
export const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
