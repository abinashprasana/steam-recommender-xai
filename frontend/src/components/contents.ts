import { reducedMotion } from '../lib/motion';

/**
 * Contents rail with scroll-spy, plus a scroll-progress bar.
 *
 * The report runs to roughly 8,000px across seven sections with no way to see
 * where you are or jump anywhere. This is the distill.pub pattern: a sticky list
 * that highlights whatever is in view. Entries are built from the sections that
 * actually exist in the DOM, so the rail cannot drift out of sync with the page.
 */

export function mountContents(scope: HTMLElement): () => void {
  const sections = [...scope.querySelectorAll<HTMLElement>('.section[id]')];
  if (sections.length < 3) return () => {};

  const items = sections.map(s => ({
    id: s.id,
    label: s.querySelector('h2')?.textContent?.trim() ?? s.id,
    kicker: s.querySelector('.section__kicker')?.textContent?.trim() ?? '',
    el: s,
  }));

  const nav = document.createElement('nav');
  nav.className = 'toc';
  nav.setAttribute('aria-label', 'Contents');
  nav.innerHTML =
    `<div class="toc__title">Contents</div><ol class="toc__list">` +
    items.map(i =>
      `<li><a class="toc__link" href="#${i.id}" data-toc="${i.id}">
        <span class="toc__kicker">${i.kicker}</span>
        <span class="toc__label">${i.label}</span>
      </a></li>`).join('') +
    `</ol>`;
  scope.prepend(nav);

  const links = new Map(
    [...nav.querySelectorAll<HTMLAnchorElement>('[data-toc]')].map(a => [a.dataset.toc!, a]),
  );

  let current = '';
  const setActive = (id: string) => {
    if (id === current) return;
    links.get(current)?.removeAttribute('aria-current');
    links.get(id)?.setAttribute('aria-current', 'true');
    current = id;
  };

  // The active section is the last one whose top has passed the upper third of
  // the viewport. Keying off "is intersecting" alone flickers between two
  // sections whenever both are on screen, which they usually are here.
  const hero = document.querySelector<HTMLElement>('.hero');

  const pick = () => {
    const line = window.innerHeight * 0.33;
    let best = items[0];
    for (const it of items) {
      if (it.el.getBoundingClientRect().top <= line) best = it;
    }
    setActive(best.id);

    // Keep the rail out of the way until the hero has scrolled off. It has
    // nothing to point at while the title is still filling the screen.
    if (hero) {
      const past = hero.getBoundingClientRect().bottom < 120;
      nav.dataset.hidden = past ? 'false' : 'true';
    }
  };

  const bar = document.createElement('div');
  bar.className = 'scroll-progress';
  bar.innerHTML = '<span></span>';
  document.body.appendChild(bar);
  const fill = bar.firstElementChild as HTMLElement;

  // 100vw includes the scrollbar but the centred wrap does not, so a CSS-only
  // calc() put the rail 8px under the content column. clientWidth excludes it.
  const placeRail = () => {
    const maxw = parseFloat(
      getComputedStyle(document.documentElement).getPropertyValue('--maxw')) || 1180;
    const avail = document.documentElement.clientWidth;
    nav.style.left = `${Math.max(0, (avail - maxw) / 2) + 24}px`;
  };

  let ticking = false;
  const onScroll = () => {
    if (ticking) return;
    ticking = true;
    requestAnimationFrame(() => {
      ticking = false;
      pick();
      const max = document.documentElement.scrollHeight - window.innerHeight;
      const ratio = max > 0 ? Math.min(1, Math.max(0, window.scrollY / max)) : 0;
      fill.style.transform = `scaleX(${ratio.toFixed(4)})`;
    });
  };

  const onResize = () => { placeRail(); onScroll(); };
  window.addEventListener('scroll', onScroll, { passive: true });
  window.addEventListener('resize', onResize, { passive: true });
  placeRail();
  // Synchronously, not through the rAF-throttled scroll path: in a backgrounded
  // tab that frame never arrives and the rail would be left in its default state.
  pick();
  onScroll();

  if (reducedMotion()) fill.style.transition = 'none';

  return () => {
    window.removeEventListener('scroll', onScroll);
    window.removeEventListener('resize', onResize);
    bar.remove();
    nav.remove();
  };
}
