import { reducedMotion } from '../lib/motion';

/**
 * Page starfield.
 *
 * The aurora in base.css gives the void colour; this gives it depth. Three
 * parallax bands of points drifting slowly upward, fixed to the viewport behind
 * everything, so scrolling the page reads as travelling through something rather
 * than as sliding panels over a flat fill.
 *
 * It deliberately does NOT run over the hero. The hero already draws a WebGL
 * network of 6,276 real neighbour edges, and a second field of points on top of
 * it is noise competing with signal. Opacity is driven from scroll position and
 * held at zero until the hero has largely left the viewport.
 *
 * Cost is the reason it is written this way rather than as another WebGL pass:
 * 300 points, 30fps, integer-snapped fillRect, and fillStyle set three times per
 * frame instead of 300. globalAlpha carries the per-star twinkle, which avoids
 * building a colour string per point per frame.
 */

const COUNT = 300;
const FPS = 30;
const FRAME_MS = 1000 / FPS;

/** Three depth bands. Nearer points are larger, brighter and drift faster,
 *  which is what produces parallax without a second camera. */
const BANDS = [
  { share: 0.52, size: 1.0, alpha: 0.30, speed: 0.0022, fill: 'rgb(150, 178, 202)' },
  { share: 0.34, size: 1.6, alpha: 0.46, speed: 0.0041, fill: 'rgb(196, 216, 234)' },
  { share: 0.14, size: 2.2, alpha: 0.60, speed: 0.0068, fill: 'rgb(240, 168, 104)' },
];

interface Star { x: number; y: number; phase: number; rate: number }

export function installStarfield(): () => void {
  const canvas = document.createElement('canvas');
  canvas.className = 'starfield';
  canvas.setAttribute('aria-hidden', 'true');
  const ctx = canvas.getContext('2d', { alpha: true });
  if (!ctx) return () => {};
  document.body.appendChild(canvas);

  let w = 0, h = 0, dpr = 1;
  const stars: Star[] = [];

  // Positions are normalised 0..1 so a resize never has to rebuild the field.
  // Stars are appended band by band, so each band owns one contiguous slice and
  // the draw loop can walk it directly instead of filtering the whole array
  // three times per frame.
  const ranges: Array<[number, number]> = [];
  let acc = 0;
  BANDS.forEach((b, i) => {
    const n = i === BANDS.length - 1 ? COUNT - acc : Math.round(COUNT * b.share);
    const from = stars.length;
    acc += n;
    for (let k = 0; k < n; k++) {
      stars.push({
        x: Math.random(),
        y: Math.random(),
        phase: Math.random() * Math.PI * 2,
        rate: 0.6 + Math.random() * 0.9,
      });
    }
    ranges.push([from, stars.length]);
  });

  function resize(): void {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    w = window.innerWidth;
    h = window.innerHeight;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    canvas.style.width = w + 'px';
    canvas.style.height = h + 'px';
    ctx!.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  /**
   * Hold at zero over the hero, ramp in once it has gone.
   *
   * Measured from the hero's actual height rather than a guessed pixel value,
   * because the hero is a viewport-relative block and a hard-coded threshold
   * would be wrong at every size except the one it was tuned on.
   */
  function heroFade(): number {
    const hero = document.querySelector('.hero');
    const hh = hero ? (hero as HTMLElement).offsetHeight : window.innerHeight;
    const start = hh * 0.55;
    const end = hh * 1.0;
    const t = (window.scrollY - start) / Math.max(end - start, 1);
    return Math.max(0, Math.min(1, t));
  }

  let shownOpacity = -1;
  function applyFade(): void {
    const v = Math.round(heroFade() * 100) / 100;
    // Only touch the style when it actually changed. Assigning an identical
    // value every frame still dirties the element.
    if (v !== shownOpacity) {
      shownOpacity = v;
      canvas.style.opacity = String(v);
    }
  }

  function draw(t: number): void {
    ctx!.clearRect(0, 0, w, h);
    for (let i = 0; i < BANDS.length; i++) {
      const b = BANDS[i];
      ctx!.fillStyle = b.fill;
      const [from, to] = ranges[i];
      for (let s = from; s < to; s++) {
        const st = stars[s];
        // Drift upward, wrapping at the top. Modulo on the normalised
        // coordinate keeps this exact however long the page stays open.
        const y = (st.y - t * b.speed * 0.001) % 1;
        const yy = (y < 0 ? y + 1 : y) * h;
        const tw = 0.72 + 0.28 * Math.sin(t * 0.0012 * st.rate + st.phase);
        ctx!.globalAlpha = b.alpha * tw;
        // Snapping to whole pixels keeps a 1px point a crisp 1px point instead
        // of a blurred two-pixel smear.
        ctx!.fillRect(Math.round(st.x * w), Math.round(yy), b.size, b.size);
      }
    }
    ctx!.globalAlpha = 1;
  }

  let raf = 0;
  let last = 0;
  let running = false;

  function loop(now: number): void {
    raf = requestAnimationFrame(loop);
    if (now - last < FRAME_MS) return;
    last = now;
    applyFade();
    draw(now);
  }

  function start(): void {
    if (running) return;
    running = true;
    last = 0;
    raf = requestAnimationFrame(loop);
  }

  function stop(): void {
    if (!running) return;
    running = false;
    cancelAnimationFrame(raf);
  }

  const still = reducedMotion();

  // The canvas is position:fixed, so it is always intersecting and an
  // IntersectionObserver would never pause it. visibilitychange is the signal
  // that actually corresponds to nobody looking.
  function onVisibility(): void {
    if (document.hidden) stop();
    else start();
  }

  let resizeTimer = 0;
  function onResize(): void {
    window.clearTimeout(resizeTimer);
    resizeTimer = window.setTimeout(() => {
      resize();
      if (still) { applyFade(); draw(0); }
    }, 150);
  }

  resize();

  if (still) {
    // Static field: drawn once, and the fade still tracks scroll so it never
    // sits on top of the hero. Reduced motion removes movement, not the page.
    applyFade();
    draw(0);
    window.addEventListener('scroll', applyFade, { passive: true });
    window.addEventListener('resize', onResize);
    return () => {
      window.removeEventListener('scroll', applyFade);
      window.removeEventListener('resize', onResize);
      canvas.remove();
    };
  }

  start();
  document.addEventListener('visibilitychange', onVisibility);
  window.addEventListener('resize', onResize);

  return () => {
    stop();
    document.removeEventListener('visibilitychange', onVisibility);
    window.removeEventListener('resize', onResize);
    canvas.remove();
  };
}
