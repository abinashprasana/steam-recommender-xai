import { api } from '../lib/api';
import { reducedMotion } from '../lib/motion';

/**
 * The hero is the model, orbiting.
 *
 * Every point is one of the 2,758 games, positioned by a PCA of its learned
 * 64-dimension BPR factors down to three. Games that sit near each other on
 * screen are games the recommender considers similar, so this is not a particle
 * effect that happens to look technical -- it is the thing the project built.
 *
 * Hand-written WebGL2 rather than three.js. A point cloud with orbit is one
 * shader pair and three matrices; three.js would cost roughly 150KB gzipped
 * against a 24KB bundle to do the same job.
 */

export interface Trace {
  target: number;
  supporters: number[];
}

const VERT = `#version 300 es
precision highp float;
in vec3 aPos;
in float aSeed;
in float aRole;          // 0 catalogue, 1 supporter, 2 target
in float aPhase;         // same wave that travels the edges
uniform mat4 uProj, uView;
uniform float uTime, uDpr, uTraceT, uStill;
out float vRole;
out float vDepth;
out float vLit;
void main() {
  vec3 p = aPos;
  // Per-point drift so the cloud breathes instead of rotating as a solid lump.
  p += 0.012 * vec3(sin(uTime * 0.3 + aSeed * 6.28),
                    cos(uTime * 0.26 + aSeed * 5.11),
                    sin(uTime * 0.22 + aSeed * 4.13));
  vec4 mv = uView * vec4(p, 1.0);
  gl_Position = uProj * mv;
  vDepth = clamp((-mv.z - 1.0) / 2.8, 0.0, 1.0);
  vRole = aRole;
  // A node lights as the wave passes through it.
  float head = fract(uTime * 0.16 - aPhase);
  vLit = exp(-head * head * 40.0) * (1.0 - uStill);
  float base = mix(9.0, 2.6, vDepth) * (1.0 + vLit * 0.55);
  // Highlighted points swell as the trace plays, then settle.
  float pulse = aRole > 0.5 ? (1.0 + 2.2 * uTraceT) : 1.0;
  gl_PointSize = base * pulse * uDpr;
}`;

const FRAG = `#version 300 es
precision highp float;
in float vRole;
in float vDepth;
in float vLit;
uniform float uTraceT;
out vec4 outColor;
void main() {
  // Round, soft-edged points. Square points are the giveaway of an unfinished
  // WebGL sketch.
  vec2 d = gl_PointCoord - vec2(0.5);
  float r = length(d);
  if (r > 0.5) discard;
  float alpha = smoothstep(0.5, 0.12, r);

  // Depth ramp through the illustration palette: far points recede into the
  // ground, near points come forward.
  vec3 far  = vec3(0.165, 0.259, 0.341);   // #2A4257
  vec3 mid  = vec3(0.306, 0.498, 0.651);   // #4E7FA6
  vec3 near = vec3(0.478, 0.796, 0.980);   // brighter near tone
  float t = 1.0 - vDepth;
  vec3 col = t < 0.5 ? mix(far, mid, t * 2.0) : mix(mid, near, (t - 0.5) * 2.0);
  float a = alpha * mix(0.30, 0.95, t);
  col = mix(col, vec3(0.478, 0.796, 0.980), vLit * 0.7);
  a = min(1.0, a + vLit * 0.35);

  if (vRole > 1.5) {                       // target: Steam blue, bright
    col = vec3(0.400, 0.753, 0.957);
    a = alpha * mix(0.6, 1.0, uTraceT);
  } else if (vRole > 0.5) {                // supporters: amber
    col = vec3(0.941, 0.659, 0.408);       // #F0A868
    a = alpha * mix(0.5, 1.0, uTraceT);
  }
  outColor = vec4(col, a);
}`;


/* The network: the model's own nearest-neighbour graph, with light travelling
   through it. Each edge carries a phase derived from its breadth-first depth, so
   pulses sweep outward across the graph like a signal propagating rather than
   every edge blinking independently. */
const NET_VERT = `#version 300 es
precision highp float;
in vec3 aPos;
in float aT;             // 0 at one end of the edge, 1 at the other
in float aPhase;         // BFS depth, normalised
uniform mat4 uProj, uView;
out float vT;
out float vPhase;
out float vFade;
void main() {
  vec4 mv = uView * vec4(aPos, 1.0);
  gl_Position = uProj * mv;
  vT = aT;
  vPhase = aPhase;
  vFade = 1.0 - clamp((-mv.z - 1.0) / 2.8, 0.0, 1.0);
}`;

const NET_FRAG = `#version 300 es
precision highp float;
in float vT;
in float vPhase;
in float vFade;
uniform float uTime;
uniform float uStill;
out vec4 outColor;
void main() {
  float head = fract(uTime * 0.16 - vPhase);
  float d = abs(vT - head);
  d = min(d, 1.0 - d);                        // wrap, so the pulse has no seam
  float pulse = exp(-d * d * 55.0) * (1.0 - uStill);
  // The base term keeps the mesh readable between pulses. At 0.05 on a 1px
  // hairline it was effectively invisible, so the network only existed while a
  // pulse crossed it -- the structure has to be legible at rest.
  float a = (0.16 + pulse * 0.60) * mix(0.30, 1.0, vFade);
  vec3 dim  = vec3(0.165, 0.259, 0.341);
  vec3 lit  = vec3(0.478, 0.796, 0.980);
  outColor = vec4(mix(dim, lit, pulse), a);
}`;

const LINE_VERT = `#version 300 es
precision highp float;
in vec3 aPos;
in float aT;             // 0 at supporter, 1 at target
uniform mat4 uProj, uView;
uniform float uTraceT;
out float vT;
void main() {
  gl_Position = uProj * uView * vec4(aPos, 1.0);
  vT = aT;
}`;

const LINE_FRAG = `#version 300 es
precision highp float;
in float vT;
uniform float uTraceT;
out vec4 outColor;
void main() {
  // The line draws itself from supporter toward target as the trace advances.
  float reveal = smoothstep(vT - 0.15, vT + 0.02, uTraceT);
  outColor = vec4(0.941, 0.659, 0.408, 0.5 * reveal);
}`;

function compile(gl: WebGL2RenderingContext, vs: string, fs: string): WebGLProgram | null {
  const mk = (type: number, src: string) => {
    const sh = gl.createShader(type)!;
    gl.shaderSource(sh, src);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
      console.warn('shader:', gl.getShaderInfoLog(sh));
      return null;
    }
    return sh;
  };
  const v = mk(gl.VERTEX_SHADER, vs);
  const f = mk(gl.FRAGMENT_SHADER, fs);
  if (!v || !f) return null;
  const p = gl.createProgram()!;
  gl.attachShader(p, v);
  gl.attachShader(p, f);
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
    console.warn('link:', gl.getProgramInfoLog(p));
    return null;
  }
  return p;
}

/* Three matrices, written out rather than pulled from a library. */
function perspective(fovy: number, aspect: number, near: number, far: number): Float32Array {
  const f = 1 / Math.tan(fovy / 2);
  const nf = 1 / (near - far);
  return new Float32Array([
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (far + near) * nf, -1,
    0, 0, 2 * far * near * nf, 0,
  ]);
}

function lookAt(eye: number[], center: number[], up: number[]): Float32Array {
  const [ex, ey, ez] = eye;
  let zx = ex - center[0], zy = ey - center[1], zz = ez - center[2];
  let l = Math.hypot(zx, zy, zz) || 1; zx /= l; zy /= l; zz /= l;
  let xx = up[1] * zz - up[2] * zy, xy = up[2] * zx - up[0] * zz, xz = up[0] * zy - up[1] * zx;
  l = Math.hypot(xx, xy, xz) || 1; xx /= l; xy /= l; xz /= l;
  const yx = zy * xz - zz * xy, yy = zz * xx - zx * xz, yz = zx * xy - zy * xx;
  return new Float32Array([
    xx, yx, zx, 0,
    xy, yy, zy, 0,
    xz, yz, zz, 0,
    -(xx * ex + xy * ey + xz * ez),
    -(yx * ex + yy * ey + yz * ez),
    -(zx * ex + zy * ey + zz * ez), 1,
  ]);
}

export interface CloudHandle { destroy(): void }

export async function mountCloud(
  canvas: HTMLCanvasElement,
  trace: Trace | null,
  onHover?: (name: string | null) => void,
): Promise<CloudHandle | null> {
  const gl = canvas.getContext('webgl2', {
    alpha: true, antialias: true, premultipliedAlpha: false,
  });
  if (!gl) return null;

  let points: Array<[number, number, number, string]>;
  let edgeIdx: number[] = [];
  let edgePhase: number[] = [];
  try {
    const data = await api.embedding();
    points = data.points as unknown as Array<[number, number, number, string]>;
    if (!points.length || points[0].length < 4) return null;   // 2D file, wrong shape
    edgeIdx = (data.edges as number[]) ?? [];
    edgePhase = (data.phases as number[]) ?? [];
  } catch {
    return null;
  }

  const n = points.length;
  const pos = new Float32Array(n * 3);
  const seed = new Float32Array(n);
  const role = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    pos[i * 3] = points[i][0];
    pos[i * 3 + 1] = points[i][1];
    pos[i * 3 + 2] = points[i][2];
    seed[i] = ((i * 2654435761) % 1000) / 1000;
  }
  if (trace) {
    if (trace.target >= 0 && trace.target < n) role[trace.target] = 2;
    for (const s of trace.supporters) if (s >= 0 && s < n) role[s] = 1;
  }

  // Node phase averaged from the edges it belongs to, so a point lights at the
  // same moment as the connections running into it.
  const nodePhase = new Float32Array(n);
  const nodeDeg = new Float32Array(n);
  for (let e = 0; e < edgePhase.length; e++) {
    const a = edgeIdx[e * 2], b = edgeIdx[e * 2 + 1];
    if (a < n) { nodePhase[a] += edgePhase[e]; nodeDeg[a]++; }
    if (b < n) { nodePhase[b] += edgePhase[e]; nodeDeg[b]++; }
  }
  for (let i = 0; i < n; i++) nodePhase[i] = nodeDeg[i] ? nodePhase[i] / nodeDeg[i] : 0;

  const prog = compile(gl, VERT, FRAG);
  const lineProg = compile(gl, LINE_VERT, LINE_FRAG);
  const netProg = compile(gl, NET_VERT, NET_FRAG);
  if (!prog || !lineProg || !netProg) return null;

  const buf = (data: Float32Array) => {
    const b = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, b);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
    return b;
  };
  const bindAttr = (p: WebGLProgram, name: string, b: WebGLBuffer, size: number) => {
    const loc = gl.getAttribLocation(p, name);
    if (loc < 0) return;
    gl.bindBuffer(gl.ARRAY_BUFFER, b);
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, 0);
  };

  const vao = gl.createVertexArray()!;
  gl.bindVertexArray(vao);
  bindAttr(prog, 'aPos', buf(pos), 3);
  bindAttr(prog, 'aSeed', buf(seed), 1);
  bindAttr(prog, 'aRole', buf(role), 1);
  bindAttr(prog, 'aPhase', buf(nodePhase), 1);
  gl.bindVertexArray(null);

  // Network: two vertices per edge, one draw call over ~12.5k vertices.
  let netVao: WebGLVertexArrayObject | null = null;
  let netCount = 0;
  if (edgePhase.length) {
    const ep: number[] = [], et: number[] = [], eph: number[] = [];
    for (let e = 0; e < edgePhase.length; e++) {
      const a = edgeIdx[e * 2], b = edgeIdx[e * 2 + 1];
      if (a >= n || b >= n) continue;
      ep.push(points[a][0], points[a][1], points[a][2]);
      et.push(0); eph.push(edgePhase[e]);
      ep.push(points[b][0], points[b][1], points[b][2]);
      et.push(1); eph.push(edgePhase[e]);
    }
    netCount = et.length;
    netVao = gl.createVertexArray()!;
    gl.bindVertexArray(netVao);
    bindAttr(netProg, 'aPos', buf(new Float32Array(ep)), 3);
    bindAttr(netProg, 'aT', buf(new Float32Array(et)), 1);
    bindAttr(netProg, 'aPhase', buf(new Float32Array(eph)), 1);
    gl.bindVertexArray(null);
  }

  // Trace lines: supporter -> target, two vertices each.
  let lineVao: WebGLVertexArrayObject | null = null;
  let lineCount = 0;
  if (trace && trace.supporters.length) {
    const lp: number[] = [], lt: number[] = [];
    for (const s of trace.supporters) {
      if (s < 0 || s >= n) continue;
      lp.push(points[s][0], points[s][1], points[s][2]);
      lt.push(0);
      lp.push(points[trace.target][0], points[trace.target][1], points[trace.target][2]);
      lt.push(1);
    }
    lineCount = lt.length;
    lineVao = gl.createVertexArray()!;
    gl.bindVertexArray(lineVao);
    bindAttr(lineProg, 'aPos', buf(new Float32Array(lp)), 3);
    bindAttr(lineProg, 'aT', buf(new Float32Array(lt)), 1);
    gl.bindVertexArray(null);
  }

  const u = (p: WebGLProgram, k: string) => gl.getUniformLocation(p, k);
  const uProj = u(prog, 'uProj'), uView = u(prog, 'uView');
  const uTime = u(prog, 'uTime'), uDpr = u(prog, 'uDpr'), uTraceT = u(prog, 'uTraceT');
  const uStill = u(prog, 'uStill');
  const nProj = u(netProg, 'uProj'), nView = u(netProg, 'uView');
  const nTime = u(netProg, 'uTime'), nStill = u(netProg, 'uStill');
  const lProj = u(lineProg, 'uProj'), lView = u(lineProg, 'uView'), lTraceT = u(lineProg, 'uTraceT');

  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  gl.disable(gl.DEPTH_TEST);   // additive-ish cloud reads better without z-reject

  const still = reducedMotion();
  let dpr = 1, W = 0, H = 0;
  let yaw = 0.6, pitch = 0.22, dist = 2.45;
  let dragging = false, lastX = 0, lastY = 0;
  let pointer = { x: -1e4, y: -1e4, inside: false };
  let raf = 0, t0 = performance.now(), traceT = 0;

  function resize() {
    const r = canvas.getBoundingClientRect();
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    W = r.width; H = r.height;
    canvas.width = Math.max(1, Math.floor(W * dpr));
    canvas.height = Math.max(1, Math.floor(H * dpr));
    gl!.viewport(0, 0, canvas.width, canvas.height);
  }

  function viewMatrix() {
    const eye = [
      dist * Math.cos(pitch) * Math.sin(yaw),
      dist * Math.sin(pitch),
      dist * Math.cos(pitch) * Math.cos(yaw),
    ];
    return { view: lookAt(eye, [0, 0, 0], [0, 1, 0]), eye };
  }

  /* Hover picking: project every point to screen and take the nearest within a
     small radius. At 2,758 points this is cheaper than a pick buffer and avoids
     a second render pass. */
  function pick(view: Float32Array, proj: Float32Array): number {
    if (!pointer.inside) return -1;
    let best = -1, bestD = 18 * 18;
    for (let i = 0; i < n; i++) {
      const x = pos[i * 3], y = pos[i * 3 + 1], z = pos[i * 3 + 2];
      const cx = view[0] * x + view[4] * y + view[8] * z + view[12];
      const cy = view[1] * x + view[5] * y + view[9] * z + view[13];
      const cz = view[2] * x + view[6] * y + view[10] * z + view[14];
      if (cz > -0.1) continue;
      const cw = -cz;
      const sx = (proj[0] * cx / cw * 0.5 + 0.5) * W;
      const sy = (1 - (proj[5] * cy / cw * 0.5 + 0.5)) * H;
      const dx = sx - pointer.x, dy = sy - pointer.y;
      const d = dx * dx + dy * dy;
      if (d < bestD) { bestD = d; best = i; }
    }
    return best;
  }

  let hovered = -1;

  function frame(now: number) {
    const t = (now - t0) / 1000;
    if (!still && !dragging) yaw += 0.0009;
    // The trace plays once shortly after load, then holds.
    traceT = still ? 1 : Math.min(1, Math.max(0, (t - 1.1) / 1.6));

    resizeIfNeeded();
    const proj = perspective(0.9, W / Math.max(H, 1), 0.1, 50);
    const { view } = viewMatrix();

    gl!.clear(gl!.COLOR_BUFFER_BIT);

    if (netVao && netCount) {
      gl!.useProgram(netProg!);
      gl!.uniformMatrix4fv(nProj, false, proj);
      gl!.uniformMatrix4fv(nView, false, view);
      gl!.uniform1f(nTime, still ? 0 : t);
      gl!.uniform1f(nStill, still ? 1 : 0);
      gl!.bindVertexArray(netVao);
      gl!.drawArrays(gl!.LINES, 0, netCount);
    }

    if (lineVao && lineCount) {
      gl!.useProgram(lineProg!);
      gl!.uniformMatrix4fv(lProj, false, proj);
      gl!.uniformMatrix4fv(lView, false, view);
      gl!.uniform1f(lTraceT, traceT);
      gl!.bindVertexArray(lineVao);
      gl!.drawArrays(gl!.LINES, 0, lineCount);
    }

    gl!.useProgram(prog!);
    gl!.uniformMatrix4fv(uProj, false, proj);
    gl!.uniformMatrix4fv(uView, false, view);
    gl!.uniform1f(uTime, still ? 0 : t);
    gl!.uniform1f(uDpr, dpr);
    gl!.uniform1f(uTraceT, traceT);
    gl!.uniform1f(uStill, still ? 1 : 0);
    gl!.bindVertexArray(vao);
    gl!.drawArrays(gl!.POINTS, 0, n);
    gl!.bindVertexArray(null);

    if (onHover) {
      const h = pick(view, proj);
      if (h !== hovered) {
        hovered = h;
        onHover(h >= 0 ? points[h][3] || null : null);
        canvas.style.cursor = h >= 0 ? 'pointer' : dragging ? 'grabbing' : 'grab';
      }
    }

    if (!still) raf = requestAnimationFrame(frame);
  }

  let lastW = 0, lastH = 0;
  function resizeIfNeeded() {
    const r = canvas.getBoundingClientRect();
    if (Math.abs(r.width - lastW) > 1 || Math.abs(r.height - lastH) > 1) {
      lastW = r.width; lastH = r.height;
      resize();
    }
  }

  resize();
  canvas.style.cursor = 'grab';

  const onDown = (e: PointerEvent) => {
    dragging = true; lastX = e.clientX; lastY = e.clientY;
    canvas.setPointerCapture(e.pointerId);
    canvas.style.cursor = 'grabbing';
  };
  const onMove = (e: PointerEvent) => {
    const r = canvas.getBoundingClientRect();
    pointer = { x: e.clientX - r.left, y: e.clientY - r.top, inside: true };
    if (!dragging) return;
    yaw -= (e.clientX - lastX) * 0.005;
    pitch = Math.max(-1.2, Math.min(1.2, pitch + (e.clientY - lastY) * 0.004));
    lastX = e.clientX; lastY = e.clientY;
    if (still) frame(performance.now());
  };
  const onUp = (e: PointerEvent) => {
    dragging = false;
    try { canvas.releasePointerCapture(e.pointerId); } catch { /* already released */ }
    canvas.style.cursor = 'grab';
  };
  const onLeave = () => { pointer.inside = false; if (hovered !== -1) { hovered = -1; onHover?.(null); } };

  canvas.addEventListener('pointerdown', onDown);
  canvas.addEventListener('pointermove', onMove);
  canvas.addEventListener('pointerup', onUp);
  canvas.addEventListener('pointerleave', onLeave);

  // Stop drawing when the hero is off screen. No reason to hold a frame budget
  // for something nobody is looking at.
  const io = new IntersectionObserver(entries => {
    const visible = entries.some(e => e.isIntersecting);
    if (visible && !raf && !still) { t0 = performance.now() - 3000; raf = requestAnimationFrame(frame); }
    if (!visible && raf) { cancelAnimationFrame(raf); raf = 0; }
  }, { threshold: 0.02 });
  io.observe(canvas);

  if (still) frame(performance.now());
  else raf = requestAnimationFrame(frame);

  return {
    destroy() {
      if (raf) cancelAnimationFrame(raf);
      io.disconnect();
      canvas.removeEventListener('pointerdown', onDown);
      canvas.removeEventListener('pointermove', onMove);
      canvas.removeEventListener('pointerup', onUp);
      canvas.removeEventListener('pointerleave', onLeave);
    },
  };
}
