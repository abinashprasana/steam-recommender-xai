import './styles/base.css';

import { brandLockup, installFavicon, animateMark } from './brand/logo';
import { installStarfield } from './background/field';
import { renderReport } from './pages/report';
import { renderRecommend } from './pages/recommend';
import { renderColdStart } from './pages/coldstart';

type Route = { path: string; label: string; render: (host: HTMLElement) => Promise<void> };

const ROUTES: Route[] = [
  { path: '/', label: 'Results', render: renderReport },
  { path: '/recommend', label: 'Live Recommendations', render: renderRecommend },
  { path: '/new-player', label: 'New Player', render: renderColdStart },
];

const app = document.getElementById('app')!;
const nav = document.getElementById('nav-links')!;
const brandSlot = document.getElementById('nav-brand')!;

installFavicon();
installStarfield();
brandSlot.innerHTML = brandLockup();
animateMark(brandSlot);

nav.innerHTML = ROUTES.map(r =>
  `<a class="nav__link" href="${r.path}" data-route>${r.label}</a>`).join('');

function match(pathname: string): Route {
  return ROUTES.find(r => r.path === pathname) ?? ROUTES[0];
}

let rendered = '';

async function navigate(pathname: string, push = true): Promise<void> {
  const route = match(pathname);
  rendered = route.path;
  if (push && location.pathname !== route.path) {
    history.pushState({}, '', route.path);
  }
  nav.querySelectorAll<HTMLAnchorElement>('[data-route]').forEach(a => {
    if (a.getAttribute('href') === route.path) a.setAttribute('aria-current', 'page');
    else a.removeAttribute('aria-current');
  });

  // View Transitions where the browser has them, plain swap otherwise. The
  // render itself is awaited outside the transition so the API round trip does
  // not sit inside a frozen frame.
  // The placeholder is written synchronously and never inside a view
  // transition. startViewTransition defers its callback to the next frame, so
  // scheduling the placeholder through it raced the render and clobbered the
  // finished page with "Loading..." a frame later.
  app.innerHTML = '<div class="wrap"><div class="empty">Loading…</div></div>';
  window.scrollTo({ top: 0, behavior: 'instant' as ScrollBehavior });

  // A failure inside a page must show itself rather than leaving the loading
  // placeholder on screen forever.
  try {
    await route.render(app);
    app.classList.remove('page-in');
    void app.offsetWidth;   // restart the animation on every navigation
    app.classList.add('page-in');
  } catch (err) {
    app.innerHTML =
      `<div class="wrap"><div class="empty error">Could not load this page.<br>
       <span class="mono">${(err as Error).message}</span></div></div>`;
  }
}

document.addEventListener('click', e => {
  const link = (e.target as HTMLElement).closest<HTMLAnchorElement>('a[data-route]');
  if (!link) return;
  e.preventDefault();
  void navigate(link.getAttribute('href')!);
});

// Chrome fires popstate for fragment changes as well as history traversal, so
// re-rendering on every popstate meant a contents-rail link tore the page down
// and refetched it. Only act when the route itself actually changed.
window.addEventListener('popstate', () => {
  if (match(location.pathname).path === rendered) return;
  void navigate(location.pathname, false);
});

void navigate(location.pathname, false);
