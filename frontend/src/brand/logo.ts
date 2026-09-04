import { reducedMotion } from '../lib/motion';

/**
 * Waypoint identity.
 *
 * A waypoint is the marker for where to go next, which is exactly what a
 * recommender outputs. It is also native gaming vocabulary: every open-world map
 * floats one over the objective. That keeps the navigation metaphor the rest of
 * the product is built on -- the hero constellation, and the navigational-star
 * pseudonyms in app.py -- so the name and the visual system agree.
 *
 * The mark is ONE path. That is not a stylistic preference -- measuring how
 * premium marks are actually built settles it: Vercel's whole logo is
 * `M57.5 0 115 100H0z`, a single triangle, and Linear's is likewise one path.
 * An earlier mark here was seven elements (four circles, three strokes) and
 * dissolved into scattered dots at nav size, because it was drawn as a diagram
 * rather than as a silhouette.
 *
 * Construction: the HUD objective marker. An inverted triangle, a second
 * triangle cut from it to leave a bracket, and a point at the centre of the
 * cavity. All three subpaths live in one element under fill-rule="evenodd" --
 * the dot sits inside two boundaries, so it lands on an odd crossing count and
 * fills. One element, one colour, three shapes.
 */

// Geometry is derived, not eyeballed. The cavity is the outer triangle scaled
// about its incentre (12, 10.172; inradius 5.572) by (r - 3.2) / r, which is
// what makes the bracket a UNIFORM 3.2 units thick on all three sides rather
// than thick at the base and thin at the point.
//
// 3.2 units is the floor, not a preference: the 24-unit grid scales by 0.667 at
// 16px, so 3.2 lands at 2.13px. The mark this replaced had a 2-unit slot that
// closed to mud below 24px, and that is the failure being designed out.
const FRAME  = 'M12 22.2 L2.8 4.6 L21.2 4.6 Z';
const CAVITY = 'M12 15.29 L8.08 7.8 L15.92 7.8 Z';

// The point being marked. Cubic quarters rather than an elliptical arc: no
// large-arc or sweep flags to get wrong, and the curve is identical in every
// renderer. Radius 1.1 leaves 1.27 units of clearance to the cavity walls, so
// the dot never merges into the bracket at nav size.
const POINT =
  'M12 9.07 C12.608 9.07 13.1 9.562 13.1 10.17 C13.1 10.778 12.608 11.27 12 11.27 ' +
  'C11.392 11.27 10.9 10.778 10.9 10.17 C10.9 9.562 11.392 9.07 12 9.07 Z';

export function markSVG(size = 22, decorative = true): string {
  return `<svg class="wp-mark" width="${size}" height="${size}" viewBox="0 0 24 24"
    ${decorative ? 'aria-hidden="true"' : 'role="img" aria-label="Waypoint"'}>
    <path class="wp-frame" d="${FRAME} ${CAVITY} ${POINT}" fill="currentColor" fill-rule="evenodd"/>
  </svg>`;
}

/** Nav lockup: mark plus wordmark as real text, wrapped in one labelled link.
 *  The wordmark is set in the display serif so the identity matches the page. */
export function brandLockup(href = '/'): string {
  return `<a class="brand" href="${href}" data-route aria-label="Waypoint">
    ${markSVG(24)}<span class="brand__word">Waypoint</span>
  </a>`;
}

/**
 * SVG favicon as a data URI: one asset, every size, no PNG set.
 *
 * The full mark is used, centre point included. That was not the assumption --
 * the plan was to drop the dot on the theory it would close the cavity at 16px
 * -- but rasterising the path and counting inked pixels along a scanline says
 * otherwise: at 16px the walls and the dot come out as three separate 2px runs
 * with clear gaps between them. The 3.2-unit walls were sized for exactly this
 * case and they hold, so the favicon can be the same mark rather than a reduced
 * variant of it.
 */
export function installFavicon(): void {
  const svg =
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">` +
    `<rect width="24" height="24" rx="5" fill="#0b141c"/>` +
    `<path d="${FRAME} ${CAVITY} ${POINT}" fill="#f0a868" fill-rule="evenodd"/>` +
    `</svg>`;

  const href = 'data:image/svg+xml,' + encodeURIComponent(svg);
  let link = document.querySelector<HTMLLinkElement>('link[rel="icon"]');
  if (!link) {
    link = document.createElement('link');
    link.rel = 'icon';
    document.head.appendChild(link);
  }
  link.type = 'image/svg+xml';
  link.href = href;
}

/** Draws itself in on first load only. A logo that re-animates on every route
 *  change reads as a glitch rather than as polish.
 *
 *  Transform only, never opacity. The first version faded in from `opacity: 0`
 *  and restored it inside requestAnimationFrame -- which never fires in a
 *  backgrounded tab, so the logo stayed permanently invisible for anyone who
 *  opened the site in a background tab. Visibility must never depend on an
 *  animation frame; motion here is decoration that degrades to nothing.
 */
let animated = false;
export function animateMark(root: ParentNode = document): void {
  if (animated || reducedMotion()) return;
  animated = true;

  const path = root.querySelector<SVGPathElement>('.wp-frame');
  if (!path) return;

  path.style.transformOrigin = '12px 14px';
  path.style.transform = 'scale(.78) rotate(-10deg)';
  requestAnimationFrame(() => {
    path.style.transition = 'transform 620ms var(--ease-out-expo)';
    path.style.transform = 'none';
  });
}
