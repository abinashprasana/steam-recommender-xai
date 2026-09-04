import { makeSvg, svgEl, text } from './helpers';
import { reducedMotion } from '../lib/motion';

/**
 * The pipeline, as moving data.
 *
 * The previous version was six identical rounded boxes with chevrons between
 * them. Equal boxes threw away the only interesting property the numbers have:
 * 59,305 reviews become 31,799 kept. Here the spine tapers in proportion to what
 * survives, particles flow along it, and at the filter the discarded share
 * visibly peels off and falls away. The motion carries the meaning rather than
 * decorating it.
 */

export interface FlowStage {
  value: string;
  label: string;
  note?: string;
  /** Relative volume surviving at this stage, 0..1. Drives spine thickness. */
  weight: number;
  /** Longer explanation, revealed on hover or keyboard focus. */
  detail: string;
}

const W = 1040;
const H = 208;
const PAD = 28;
const SPINE_Y = 104;

/* Values above the spine, labels below, one lane each.
 *
 * The first version alternated stages above and below to save width. That put
 * stage 02's label and the discarded-branch label on the same baseline, so
 * "31,799", "interactions kept" and "27,506 discarded" printed on top of one
 * another. Uniform lanes cost a little height and make collisions impossible. */
const LANE_INDEX = SPINE_Y - 60;
const LANE_VALUE = SPINE_Y - 32;
/* Labels sit clear of the particle band. Particles reach SPINE_Y + 18 (spread)
   + 12 (fall cap) = 34, so a label baseline at +52 never has one crossing it. */
const LANE_LABEL = SPINE_Y + 52;
const LANE_DROP = SPINE_Y + 82;

export function flowDiagram(host: HTMLElement, stages: FlowStage[]): void {
  const still = reducedMotion();
  const n = stages.length;
  const x0 = PAD + 46;
  const x1 = W - PAD - 46;
  const xs = stages.map((_, i) => x0 + (i * (x1 - x0)) / (n - 1));

  const wrap = document.createElement('div');
  wrap.className = 'flow';

  const canvas = document.createElement('canvas');
  canvas.className = 'flow__canvas';
  canvas.setAttribute('aria-hidden', 'true');

  const svg = makeSvg(W, H);
  svg.classList.add('flow__svg');
  svg.setAttribute('aria-label',
    'Pipeline: ' + stages.map(s => `${s.label} ${s.value}`).join(', then '));

  // --- spine: a tapering ribbon, thickness proportional to surviving volume ---
  const halfAt = (i: number) => 2 + stages[i].weight * 16;
  const top: string[] = [];
  const bot: string[] = [];
  for (let i = 0; i < n; i++) {
    top.push(`${xs[i]},${SPINE_Y - halfAt(i)}`);
    bot.push(`${xs[i]},${SPINE_Y + halfAt(i)}`);
  }
  const ribbon = svgEl('path', {
    d: `M${top.join(' L')} L${bot.reverse().join(' L')} Z`,
    fill: 'var(--ill-stroke)', opacity: 0.55,
  });
  svg.appendChild(ribbon);

  // --- the discarded branch: what falls away at the filter ---
  const dropIdx = stages.findIndex((s, i) => i > 0 && s.weight < stages[i - 1].weight);
  if (dropIdx > 0) {
    const bx = (xs[dropIdx - 1] + xs[dropIdx]) / 2;
    svg.appendChild(svgEl('path', {
      d: `M${bx},${SPINE_Y + 6} C${bx + 4},${SPINE_Y + 34} ${bx + 2},${SPINE_Y + 52} ${bx},${LANE_DROP - 12}`,
      fill: 'none', stroke: 'var(--ill-stroke)', 'stroke-width': 1.2,
      'stroke-dasharray': '2 4', opacity: 0.9,
    }));
    // Sits in its own lane, 32px clear of the stage labels above it.
    const lost = text('27,506 discarded', { x: bx, y: LANE_DROP, 'text-anchor': 'middle',
      fill: 'var(--text-faint)' });
    lost.style.fontSize = '12px';
    svg.appendChild(lost);
  }

  // --- stages ---
  stages.forEach((s, i) => {
    const g = svgEl('g', { class: 'flow__stage', tabindex: '0', role: 'button' });
    g.setAttribute('aria-label', `${s.label}: ${s.value}. ${s.detail}`);

    // A generous invisible hit area: the visible node is 5px and would be a
    // miserable target otherwise.
    g.appendChild(svgEl('rect', {
      x: xs[i] - 76, y: LANE_INDEX - 14, width: 152, height: LANE_LABEL - LANE_INDEX + 24,
      fill: 'transparent',
    }));

    g.appendChild(svgEl('line', {
      x1: xs[i], x2: xs[i], y1: SPINE_Y - halfAt(i), y2: LANE_VALUE + 8,
      stroke: 'var(--border-strong)', 'stroke-width': 1,
    }));
    g.appendChild(svgEl('line', {
      x1: xs[i], x2: xs[i], y1: SPINE_Y + halfAt(i), y2: LANE_LABEL - 13,
      stroke: 'var(--border-strong)', 'stroke-width': 1,
    }));

    const node = svgEl('circle', {
      class: 'flow__node', cx: xs[i], cy: SPINE_Y, r: 4.5,
      fill: 'var(--bg-void)', stroke: 'var(--amber)', 'stroke-width': 1.6,
    });
    g.appendChild(node);

    const idx = text(String(i + 1).padStart(2, '0'),
      { x: xs[i], y: LANE_INDEX, 'text-anchor': 'middle', fill: 'var(--text-faint)' });
    idx.style.fontFamily = 'var(--font-mono)';
    idx.style.fontSize = '10.5px';
    g.appendChild(idx);

    const val = text(s.value, { x: xs[i], y: LANE_VALUE, 'text-anchor': 'middle', fill: 'var(--text-primary)' });
    val.style.fontFamily = 'var(--font-mono)';
    val.style.fontSize = '19px';
    val.style.fontVariantNumeric = 'tabular-nums';
    val.style.letterSpacing = '-0.02em';
    g.appendChild(val);

    const lab = text(s.label, { x: xs[i], y: LANE_LABEL, 'text-anchor': 'middle', fill: 'var(--text-muted)' });
    lab.style.fontSize = '13.5px';
    g.appendChild(lab);

    svg.appendChild(g);
  });

  const detail = document.createElement('p');
  detail.className = 'flow__detail';
  detail.textContent = stages[0]?.detail ?? '';

  wrap.append(canvas, svg, detail);
  host.replaceChildren(wrap);

  // --- stage focus/hover reveals its detail and pauses the flow ---
  let paused = false;
  const setActive = (i: number | null) => {
    paused = i !== null;
    svg.querySelectorAll<SVGGElement>('.flow__stage').forEach((el, j) => {
      el.classList.toggle('is-active', j === i);
    });
    detail.textContent = i === null ? (stages[0]?.detail ?? '') : stages[i].detail;
  };
  svg.querySelectorAll<SVGGElement>('.flow__stage').forEach((el, i) => {
    el.addEventListener('pointerenter', () => setActive(i));
    el.addEventListener('focus', () => setActive(i));
    el.addEventListener('pointerleave', () => setActive(null));
    el.addEventListener('blur', () => setActive(null));
  });

  if (still) return;   // the tapered spine and every number already tell the story

  // --- particle flow -------------------------------------------------------
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  interface P { t: number; off: number; speed: number; dropped: boolean; life: number }
  const particles: P[] = [];
  // The share that survives the filter, taken from the real weights rather than
  // picked to look right.
  const survive = dropIdx > 0 ? stages[dropIdx].weight / stages[dropIdx - 1].weight : 1;

  const spawn = (): P => ({
    t: -Math.random() * 0.15,
    off: (Math.random() * 2 - 1),
    speed: 0.055 + Math.random() * 0.045,
    dropped: Math.random() > survive,
    life: 1,
  });
  for (let i = 0; i < 150; i++) { const p = spawn(); p.t = Math.random(); particles.push(p); }

  const spineAt = (t: number) => {
    const f = Math.max(0, Math.min(1, t)) * (n - 1);
    const i = Math.min(n - 2, Math.floor(f));
    const k = f - i;
    return {
      x: xs[i] + (xs[i + 1] - xs[i]) * k,
      half: halfAt(i) + (halfAt(i + 1) - halfAt(i)) * k,
    };
  };
  const dropT = dropIdx > 0 ? (dropIdx - 0.5) / (n - 1) : 2;

  let dpr = 1, cw = 0, ch = 0, raf = 0, last = performance.now();

  function resize() {
    const r = canvas.getBoundingClientRect();
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    cw = r.width; ch = r.height;
    canvas.width = Math.max(1, Math.round(cw * dpr));
    canvas.height = Math.max(1, Math.round(ch * dpr));
    ctx!.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function frame(now: number) {
    const dt = Math.min(0.05, (now - last) / 1000);
    last = now;
    const sx = cw / W, sy = ch / H;   // the SVG scales to the box; match it

    ctx!.clearRect(0, 0, cw, ch);

    for (const p of particles) {
      if (!paused) {
        p.t += p.speed * dt;
        if (p.dropped && p.t > dropT) p.life -= dt * 3.2;
      }
      if (p.t > 1.05 || p.life <= 0) { Object.assign(p, spawn()); continue; }
      if (p.t < 0) continue;

      const { x, half } = spineAt(p.t);
      let y = SPINE_Y + p.off * half;

      // Past the filter, discarded rows fall away instead of continuing. The
      // fall is capped well above the label lane: particles drifting across the
      // numbers was half of why this diagram read as noisy.
      if (p.dropped && p.t > dropT) {
        const fall = (p.t - dropT) * 8;
        y += Math.min(12, fall * fall * 26);
      }

      const a = (p.dropped && p.t > dropT) ? Math.max(0, p.life) * 0.5 : 0.75;
      ctx!.fillStyle = p.dropped && p.t > dropT
        ? `rgba(132,150,165,${a.toFixed(3)})`
        : `rgba(240,168,104,${a.toFixed(3)})`;
      ctx!.beginPath();
      ctx!.arc(x * sx, y * sy, 1.5, 0, Math.PI * 2);
      ctx!.fill();
    }

    raf = requestAnimationFrame(frame);
  }

  const ro = new ResizeObserver(resize);
  ro.observe(canvas);
  resize();

  // Same pause-when-offscreen pattern as the hero cloud.
  const io = new IntersectionObserver(entries => {
    const visible = entries.some(e => e.isIntersecting);
    if (visible && !raf) { last = performance.now(); raf = requestAnimationFrame(frame); }
    if (!visible && raf) { cancelAnimationFrame(raf); raf = 0; }
  }, { threshold: 0.05 });
  io.observe(canvas);
}
