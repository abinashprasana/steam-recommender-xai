import { api } from '../lib/api';
import { reducedMotion } from '../lib/motion';

/**
 * The hero background is the model, not decoration.
 *
 * Each point is one of the 2,758 games, positioned by a PCA projection of its
 * learned BPR factor vector -- so games that sit near each other on screen are
 * games the recommender considers similar. Hovering names one.
 *
 * Canvas 2D on purpose. A three.js scene would cost 150KB+ to render a shape
 * that means nothing; this costs a few KB and is honest about what it shows.
 */
export async function mountConstellation(canvas: HTMLCanvasElement): Promise<void> {
  let points: Array<[number, number, string]>;
  try {
    // The projection file is 3D now; this fallback only needs the first two
    // components, with the name always last.
    const raw = (await api.embedding()).points as unknown[][];
    points = raw.map(p => [Number(p[0]), Number(p[1]), String(p[p.length - 1] ?? '')]);
  } catch {
    canvas.style.background =
      'radial-gradient(ellipse at 50% 35%, rgba(26,159,255,0.10), transparent 62%)';
    return;
  }

  const ctx = canvas.getContext('2d', { alpha: true });
  if (!ctx) return;

  const still = reducedMotion();
  let w = 0, h = 0, dpr = 1;
  let pointer = { x: -1e4, y: -1e4, active: false };

  /* Three graded stops rather than one flat cyan. A single bright colour at
     uniform alpha reads as noise over the headline; grading by depth gives the
     field air, and keeps the brightest values rare so the type still wins. */
  const RAMP = [
    { at: 0.0, rgb: [22, 41, 58] },    // #16293A  far, deep slate
    { at: 0.55, rgb: [46, 92, 125] },  // #2E5C7D  mid, steel
    { at: 1.0, rgb: [127, 179, 213] }, // #7FB3D5  near, soft light blue
  ];

  const shade = (d: number): [number, number, number] => {
    let lo = RAMP[0], hi = RAMP[RAMP.length - 1];
    for (let i = 0; i < RAMP.length - 1; i++) {
      if (d >= RAMP[i].at && d <= RAMP[i + 1].at) { lo = RAMP[i]; hi = RAMP[i + 1]; break; }
    }
    const t = (d - lo.at) / Math.max(1e-6, hi.at - lo.at);
    return [0, 1, 2].map(k => Math.round(lo.rgb[k] + (hi.rgb[k] - lo.rgb[k]) * t)) as
      [number, number, number];
  };

  // Depth is deterministic per index, so the field is stable between reloads.
  // Drift amplitude and size both scale with it, which reads as parallax.
  const nodes = points.map(([x, y, name], i) => {
    const depth = ((i * 2654435761) % 1000) / 1000;
    const [r0, g0, b0] = shade(depth);
    return {
      x, y, name, depth,
      phase: (i * 0.618) % (Math.PI * 2),
      amp: (0.003 + depth * 0.006),
      r: 0.5 + depth * 1.5,
      fill: `rgba(${r0},${g0},${b0},${(0.22 + depth * 0.4).toFixed(3)})`,
    };
  });

  const label = document.createElement('div');
  label.className = 'tooltip';
  canvas.parentElement?.appendChild(label);

  function resize() {
    const rect = canvas.getBoundingClientRect();
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    w = rect.width; h = rect.height;
    canvas.width = Math.max(1, Math.floor(w * dpr));
    canvas.height = Math.max(1, Math.floor(h * dpr));
    ctx!.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  const project = (n: { x: number; y: number }, t: number, ph: number, amp: number) => {
    const s = Math.min(w, h) * 0.46;
    return {
      px: w / 2 + n.x * s + Math.sin(t * 0.00022 + ph) * amp * s,
      py: h / 2 + n.y * s * 0.72 + Math.cos(t * 0.00019 + ph) * amp * s,
    };
  };

  function frame(t: number) {
    ctx!.clearRect(0, 0, w, h);

    const drawn: Array<{ px: number; py: number; r: number; name: string; depth: number; fill: string }> = [];
    for (const n of nodes) {
      const { px, py } = project(n, still ? 0 : t, n.phase, n.amp);
      if (px < -40 || px > w + 40 || py < -40 || py > h + 40) continue;
      drawn.push({ px, py, r: n.r, name: n.name, depth: n.depth, fill: n.fill });
    }

    // Link near neighbours. Bucketed by a coarse grid so this stays O(n) rather
    // than comparing every pair each frame.
    const cell = 46;
    const buckets = new Map<string, typeof drawn>();
    for (const d of drawn) {
      const k = `${(d.px / cell) | 0},${(d.py / cell) | 0}`;
      (buckets.get(k) ?? buckets.set(k, []).get(k)!).push(d);
    }
    ctx!.lineWidth = 1;
    for (const [key, list] of buckets) {
      const [cx, cy] = key.split(',').map(Number);
      const near = [...list, ...(buckets.get(`${cx + 1},${cy}`) ?? []), ...(buckets.get(`${cx},${cy + 1}`) ?? [])];
      for (let i = 0; i < list.length; i++) {
        for (let j = i + 1; j < near.length; j++) {
          const a = list[i], b = near[j];
          // Only the nearer half of the field links up. Linking everything at
          // this density produced a solid haze rather than a constellation.
          if (a.depth < 0.55 || b.depth < 0.55) continue;
          const dx = a.px - b.px, dy = a.py - b.py;
          const dist2 = dx * dx + dy * dy;
          if (dist2 > cell * cell) continue;
          ctx!.strokeStyle = `rgba(62,107,140,${((1 - Math.sqrt(dist2) / cell) * 0.09).toFixed(3)})`;
          ctx!.beginPath();
          ctx!.moveTo(a.px, a.py);
          ctx!.lineTo(b.px, b.py);
          ctx!.stroke();
        }
      }
    }

    let hovered: typeof drawn[number] | null = null;
    for (const d of drawn) {
      const dx = d.px - pointer.x, dy = d.py - pointer.y;
      const isNear = pointer.active && dx * dx + dy * dy < 120;
      if (isNear && (!hovered || dx * dx + dy * dy < 40)) hovered = d;
      ctx!.beginPath();
      ctx!.arc(d.px, d.py, isNear ? d.r + 2.2 : d.r, 0, Math.PI * 2);
      // Steam's bright blue is reserved for the point under the cursor, so it
      // stays a highlight instead of becoming the base colour of the field.
      ctx!.fillStyle = isNear ? '#66c0f4' : d.fill;
      ctx!.fill();
      if (isNear) {
        ctx!.beginPath();
        ctx!.arc(d.px, d.py, d.r + 7, 0, Math.PI * 2);
        ctx!.strokeStyle = 'rgba(102,192,244,0.45)';
        ctx!.stroke();
      }
    }

    if (hovered && hovered.name) {
      const rect = canvas.getBoundingClientRect();
      label.textContent = hovered.name;
      label.dataset.show = 'true';
      label.style.position = 'absolute';
      label.style.transform = `translate(${hovered.px + 14}px, ${hovered.py + 12}px)`;
      void rect;
    } else {
      label.dataset.show = 'false';
    }

    if (!still) raf = requestAnimationFrame(frame);
  }

  let raf = 0;
  resize();
  window.addEventListener('resize', () => { resize(); if (still) frame(0); });

  canvas.addEventListener('pointermove', e => {
    const r = canvas.getBoundingClientRect();
    pointer = { x: e.clientX - r.left, y: e.clientY - r.top, active: true };
  });
  canvas.addEventListener('pointerleave', () => { pointer.active = false; });

  // Stop drawing when the hero is off-screen -- no reason to burn a frame budget
  // animating something nobody is looking at.
  const io = new IntersectionObserver(entries => {
    const visible = entries.some(e => e.isIntersecting);
    if (visible && !raf && !still) raf = requestAnimationFrame(frame);
    if (!visible && raf) { cancelAnimationFrame(raf); raf = 0; }
  }, { threshold: 0.02 });
  io.observe(canvas);

  frame(0);
}
