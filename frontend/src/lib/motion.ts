export const reducedMotion = () =>
  window.matchMedia('(prefers-reduced-motion: reduce)').matches;

/** Reveal on scroll. Content is present in the DOM and readable regardless --
 *  this only animates its arrival, so nothing is gated behind an observer. */
export function observeReveals(root: ParentNode = document): void {
  const items = Array.from(root.querySelectorAll<HTMLElement>('.reveal:not([data-in])'));
  if (reducedMotion()) {
    items.forEach(el => el.setAttribute('data-in', 'true'));
    return;
  }
  const io = new IntersectionObserver(
    entries => {
      entries.forEach((entry, i) => {
        const el = entry.target as HTMLElement;
        // Reveal on intersection, and also for anything already above the
        // viewport. Sections scrolled past from below report as not
        // intersecting, so keying purely off isIntersecting left them
        // permanently invisible when the reader scrolled back up.
        const passed = entry.boundingClientRect.top < 0;
        if (!entry.isIntersecting && !passed) return;
        window.setTimeout(() => el.setAttribute('data-in', 'true'), passed ? 0 : i * 60);
        io.unobserve(el);
      });
    },
    // A fixed margin rather than a percentage: these sections are taller than
    // the viewport, so a percentage threshold demanded hundreds of pixels of
    // visibility and the reveal trailed the scroll badly.
    { rootMargin: '0px 0px -40px 0px', threshold: 0 },
  );
  items.forEach(el => io.observe(el));
}

/** Count a number up when it scrolls into view. Width never shifts because the
 *  element uses tabular figures. */
export function countUp(el: HTMLElement, to: number, decimals = 4, suffix = ''): void {
  const render = (v: number) => (el.textContent = v.toFixed(decimals) + suffix);
  if (reducedMotion()) { render(to); return; }

  let started = false;
  const io = new IntersectionObserver(entries => {
    if (!entries.some(e => e.isIntersecting) || started) return;
    started = true;
    io.disconnect();
    const dur = 900;
    const t0 = performance.now();
    const tick = (now: number) => {
      const p = Math.min(1, (now - t0) / dur);
      render(to * (1 - Math.pow(1 - p, 3)));
      if (p < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }, { threshold: 0.4 });
  io.observe(el);
  render(0);
}

/** Subtle pull toward the cursor. Skipped entirely on touch and reduced motion. */
export function magnetic(el: HTMLElement, strength = 4): void {
  if (reducedMotion() || window.matchMedia('(hover: none)').matches) return;
  el.addEventListener('pointermove', e => {
    const r = el.getBoundingClientRect();
    const dx = ((e.clientX - r.left) / r.width - 0.5) * strength * 2;
    const dy = ((e.clientY - r.top) / r.height - 0.5) * strength * 2;
    el.style.transform = `translate(${dx.toFixed(2)}px, ${dy.toFixed(2)}px)`;
  });
  el.addEventListener('pointerleave', () => { el.style.transform = ''; });
}

/** Native View Transitions where available, plain swap everywhere else.
 *
 *  Defensive on purpose: startViewTransition throws InvalidStateError when the
 *  document is hidden (a backgrounded tab, or a preview pane that has not been
 *  shown yet), and its promises reject when a transition is superseded. None of
 *  that should ever prevent the DOM update -- the animation is optional, the
 *  content is not.
 */
export function transition(update: () => void): void {
  const doc = document as Document & {
    startViewTransition?: (cb: () => void) => { finished?: Promise<void>; ready?: Promise<void> };
  };

  if (reducedMotion() ||
      typeof doc.startViewTransition !== 'function' ||
      document.visibilityState !== 'visible') {
    update();
    return;
  }

  try {
    const vt = doc.startViewTransition(update);
    vt.finished?.catch(() => {});
    vt.ready?.catch(() => {});
  } catch {
    update();
  }
}

let tipEl: HTMLElement | null = null;
export function tooltip() {
  if (!tipEl) {
    tipEl = document.createElement('div');
    tipEl.className = 'tooltip';
    document.body.appendChild(tipEl);
  }
  const el = tipEl;
  return {
    show(html: string, x: number, y: number) {
      el.innerHTML = html;
      el.dataset.show = 'true';
      const pad = 14;
      const r = el.getBoundingClientRect();
      // Flip before the tooltip would leave the viewport, so it never covers
      // the point the reader is inspecting.
      const left = x + pad + r.width > window.innerWidth ? x - r.width - pad : x + pad;
      const top = y + pad + r.height > window.innerHeight ? y - r.height - pad : y + pad;
      el.style.transform = `translate(${left}px, ${top}px)`;
    },
    hide() { el.dataset.show = 'false'; },
  };
}
