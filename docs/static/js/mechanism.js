// Animated main figure: (a) the latent reasoning recurrence, (b) SVP-V.
// Both panels play together, once, when scrolled into view; afterwards only the noise draw is
// resampled, to show that each trajectory gets its own perturbed W_V.
(function () {
  var host = document.getElementById('mech-fig');
  if (!host) return;

  var SVG = 'http://www.w3.org/2000/svg';
  var reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var YELLOW = { fill: '#fdeebf', stroke: '#f2b632' }, GREEN = { fill: '#cfe8d6', stroke: '#2f9e4f' };
  var RED = '#e8453c', RED_FILL = '#f9d0cd', PURPLE = '#a995c9', BLUE = '#3f7fe0';
  var C = 14;          // matrix cell size
  var RANK = 4;

  function el(name, attrs, text, parent) {
    var e = document.createElementNS(SVG, name);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    (parent || svg).appendChild(e);
    return e;
  }
  function stage(n, parent) { return el('g', { 'class': 'mf-s', 'data-s': n }, null, parent); }

  // text with optional subscript, set in TeX fonts: math('h', 'T-1').
  // Letters are italic; digits, operators and anything with rm: true are upright.
  function math(x, y, base, sub, parent, attrs) {
    attrs = attrs || {};
    var a = { x: x, y: y, 'text-anchor': 'middle', 'class': 'mf-math' + (attrs.rm ? ' mf-rm' : '') };
    for (var k in attrs) if (k !== 'rm') a[k] = attrs[k];
    var t = el('text', a, null, parent);
    // a trailing transpose mark is set as a raised, upright superscript
    var sup = base.slice(-1) === '\u22a4';
    el('tspan', {}, sup ? base.slice(0, -1) : base, t);
    if (sup) el('tspan', { dy: -8, dx: 1, 'font-size': '0.6em', 'class': 'mf-rm' }, '\u22a4', t);
    if (sub) {
      (sub.match(/[A-Za-z]+|[^A-Za-z]+/g) || []).forEach(function (run, i) {
        var s = { 'font-size': '0.68em' };
        if (i === 0) s.dy = 4;
        if (!/[A-Za-z]/.test(run)) s['class'] = 'mf-rm';
        el('tspan', s, run, t);
      });
    }
    return t;
  }
  function token(x, y, style, parent, dashed) {
    return el('rect', { x: x, y: y, width: 40, height: 32, rx: 6, fill: style.fill, stroke: style.stroke,
      'stroke-width': 1.3, 'stroke-dasharray': dashed ? '4 3' : 'none', 'fill-opacity': dashed ? 0.45 : 1 }, null, parent);
  }
  function dots(x, y, parent) { return el('text', { x: x, y: y, 'text-anchor': 'middle', 'class': 'mf-dots' }, '…', parent); }

  // cols x rows grid; returns its cell rects (row-major)
  function grid(x, y, cols, rows, opt, parent) {
    var g = el('g', {}, null, parent), cells = [];
    for (var r = 0; r < rows; r++) for (var c = 0; c < cols; c++) {
      var diagOnly = opt.diag && r !== c;
      cells.push(el('rect', { x: x + c * C, y: y + r * C, width: C, height: C,
        fill: diagOnly ? '#fff' : opt.fill, 'fill-opacity': opt.diag && !diagOnly ? 1 - r * 0.22 : 1,
        stroke: opt.line, 'stroke-width': 0.6 }, null, g));
    }
    if (opt.frame) el('rect', { x: x, y: y, width: cols * C, height: rows * C, fill: 'none', stroke: opt.frame, 'stroke-width': 3 }, null, g);
    return cells;
  }

  var svg = el('svg', { viewBox: '0 0 1040 586', role: 'img',
    'aria-label': 'Overview: a latent reasoning model feeds each hidden state back as the next input embedding; SVP-V decomposes the Value projection by SVD and multiplies its singular values by Gaussian noise to obtain a perturbed Value projection.' }, null, host);

  var defs = el('defs', {});
  var head = el('marker', { id: 'mf-head', viewBox: '0 0 8 8', refX: 7, refY: 4, markerWidth: 7, markerHeight: 7, orient: 'auto' }, null, defs);
  el('path', { d: 'M0 0.5 L8 4 L0 7.5 L2 4 Z', fill: '#111' }, null, head);

  // ------------------------------------------------ (a) latent reasoning model
  el('text', { x: 16, y: 24, 'class': 'mf-title' }, '(a) Latent Reasoning Model');
  [['output token', 50], ['final activation', 99], ['input embedding', 253], ['input token', 292]].forEach(function (l) {
    el('text', { x: 905, y: l[1], 'class': 'mf-side' }, l[0]);
  });

  var TOP = 78, BOT = 232;
  var inj = el('g', { 'class': 'mf-inject' });
  el('rect', { x: 328, y: 60, width: 400, height: 218, rx: 18, fill: 'none', stroke: '#2f9e4f', 'stroke-width': 1.8, 'stroke-dasharray': '7 5' }, null, inj);
  el('line', { x1: 262, y1: 94, x2: 328, y2: 94, stroke: '#2f9e4f', 'stroke-width': 1.8, 'stroke-dasharray': '7 5' }, null, inj);
  el('text', { x: 170, y: 86, 'text-anchor': 'middle', 'class': 'mf-note' }, 'Inject stochasticity', inj);
  el('text', { x: 170, y: 106, 'text-anchor': 'middle', 'class': 'mf-note' }, 'during latent reasoning', inj);

  // recurrence arrows are drawn under the LRM block, like the paper figure
  var arrows = [[345, 425, 2], [425, 510, 4], [590, 670, 6]].map(function (a) {
    var x1 = a[0] + 40, x2 = a[1], g = stage(a[2]);
    // leaves the state on its right, runs behind the LRM block, enters the next input from the left
    el('path', { d: 'M' + x1 + ' ' + (TOP + 16) +
        ' C' + (x1 + 14) + ' ' + (TOP + 16) + ' ' + (x1 + 19) + ' ' + (TOP + 32) + ' ' + (x1 + 21) + ' 130' +
        ' L' + (x2 - 17) + ' 214' +
        ' C' + (x2 - 15) + ' ' + (BOT + 6) + ' ' + (x2 - 11) + ' ' + (BOT + 16) + ' ' + (x2 - 2) + ' ' + (BOT + 16),
      pathLength: 1, 'class': 'mf-arrow', 'marker-end': 'url(#mf-head)' }, null, g);
    return g;
  });
  el('rect', { x: 40, y: 130, width: 845, height: 84, rx: 16, fill: '#9aa0a8', stroke: '#868d96' });
  el('text', { x: 462, y: 184, 'text-anchor': 'middle', 'class': 'mf-lrm' }, 'LRM');

  [55, 130, 270, 345].forEach(function (x) { token(x, BOT, YELLOW); });
  dots(222, BOT + 24);
  el('text', { x: 170, y: 294, 'text-anchor': 'middle', 'class': 'mf-tok-b' }, '[question]');
  el('text', { x: 365, y: 294, 'text-anchor': 'middle', 'class': 'mf-tok' }, '<start-latent>');

  var g1 = stage(1); token(345, TOP, GREEN, g1); math(365, TOP + 21, 'h', '0', g1);
  token(425, BOT, GREEN, arrows[0]);
  var g3 = stage(3); token(425, TOP, GREEN, g3); math(445, TOP + 21, 'h', '1', g3);
  token(510, BOT, GREEN, arrows[1]);
  var g5 = stage(5); dots(530, TOP + 24, g5); token(590, TOP, GREEN, g5); math(610, TOP + 21, 'h', 'T−1', g5); dots(610, BOT + 24, g5);
  token(670, BOT, GREEN, arrows[2]);
  var g7 = stage(7); token(670, TOP, GREEN, g7, true); math(690, TOP + 21, 'h', 'T', g7);
  var g8 = stage(8);
  token(770, BOT, YELLOW, g8); dots(845, BOT + 24, g8);
  el('text', { x: 790, y: 294, 'text-anchor': 'middle', 'class': 'mf-tok' }, '<end-latent>', g8);
  token(770, TOP, YELLOW, g8); dots(845, TOP + 24, g8);
  el('text', { x: 790, y: 52, 'text-anchor': 'middle', 'class': 'mf-tok-b' }, '[answer]', g8);

  el('line', { x1: 8, y1: 318, x2: 1032, y2: 318, stroke: '#9aa0a8', 'stroke-width': 2.5 });

  // ------------------------------------------------ (b) SVP-V
  el('text', { x: 16, y: 348, 'class': 'mf-title' }, '(b) SVP-V');
  var red = { fill: RED_FILL, line: RED }, redF = { fill: RED_FILL, line: RED, frame: RED };
  var My = 412;        // top of the tall matrices

  el('rect', { x: 10, y: 362, width: 418, height: 192, rx: 8, fill: '#fcf5f4', stroke: '#d9534f', 'stroke-width': 1.2 });
  var g9 = stage(9); math(72, My - 10, 'W', 'V', g9); grid(30, My, 6, 8, red, g9);
  var g10 = stage(10);
  el('text', { x: 146, y: My + 48, 'text-anchor': 'middle', 'class': 'mf-math mf-rm', 'font-size': 14 }, 'SVD', g10);
  el('path', { d: 'M126 ' + (My + 58) + ' h32 m-7 -5 l8 5 l-8 5', 'class': 'mf-garrow' }, null, g10);
  math(210, My - 10, 'U', null, g10); grid(182, My, RANK, 8, redF, g10);
  math(280, My + 18, 'Σ', null, g10, { rm: true }); grid(252, My + 28, RANK, RANK, { fill: RED_FILL, line: RED, frame: RED, diag: true }, g10);
  math(364, My + 18, 'R⊤', null, g10); grid(322, My + 28, 6, RANK, redF, g10);

  var g11 = stage(11);
  el('text', { x: 462, y: My + 72, 'text-anchor': 'middle', 'class': 'mf-op' }, '+', g11);
  math(526, My - 10, 'U', null, g11); grid(498, My, RANK, 8, redF, g11);
  math(762, My + 18, 'R⊤', null, g11); grid(720, My + 28, 6, RANK, redF, g11);
  el('rect', { x: 566, y: 330, width: 140, height: 224, rx: 6, fill: '#f7f4fb', stroke: PURPLE, 'stroke-width': 1.2 }, null, g11);
  math(606, 352, 'Σ', null, g11, { rm: true }); grid(578, 360, RANK, RANK, { fill: RED_FILL, line: RED, frame: RED, diag: true }, g11);
  el('text', { x: 636, y: 576, 'text-anchor': 'middle', 'class': 'mf-note' }, 'Gaussian noise on singular values', g11);

  var g12 = stage(12);
  el('circle', { cx: 654, cy: 388, r: 8, fill: 'none', stroke: '#222', 'stroke-width': 1.2 }, null, g12);
  el('circle', { cx: 654, cy: 388, r: 1.8, fill: '#222' }, null, g12);
  math(682, 352, 'ε', null, g12);
  var epsCells = grid(675, 360, 1, RANK, { fill: BLUE, line: '#fff' }, g12);

  var g13 = stage(13);
  el('path', { d: 'M636 ' + 428 + ' v22 m-5 -7 l5 8 l5 -8', 'class': 'mf-garrow' }, null, g13);
  var pertCells = grid(608, 470, RANK, RANK, { fill: '#fff', line: PURPLE, frame: PURPLE }, g13);

  var g14 = stage(14);
  el('text', { x: 836, y: My + 72, 'text-anchor': 'middle', 'class': 'mf-op' }, '=', g14);
  // W with a tilde accent: the TeX italic face has no precomposed W-tilde
  math(912, My - 10, 'W', 'V', g14);
  el('text', { x: 909, y: My - 14, 'text-anchor': 'middle', 'class': 'mf-math mf-rm' }, '\u02dc', g14);
  var outCells = grid(870, My, 6, 8, { fill: PURPLE, line: '#8f7bb3' }, g14);

  // ------------------------------------------------ noise draw
  function gauss() {
    var u = 1 - Math.random(), v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  function sample() {
    var eps = [];
    for (var i = 0; i < RANK; i++) eps.push(gauss());
    epsCells.forEach(function (c, i) { c.setAttribute('fill-opacity', Math.min(0.95, 0.15 + Math.abs(eps[i]) * 0.4).toFixed(2)); });
    pertCells.forEach(function (c, i) {
      var r = Math.floor(i / RANK), col = i % RANK;
      if (r !== col) return;
      // s_i (1 + alpha eps_i), shown as intensity
      var s = (1 - r * 0.22) * (1 + 0.6 * eps[r]);
      c.setAttribute('fill', PURPLE);
      c.setAttribute('fill-opacity', Math.max(0.08, Math.min(1, s)).toFixed(2));
    });
    outCells.forEach(function (c) { c.setAttribute('fill-opacity', (0.3 + Math.random() * 0.5).toFixed(2)); });
  }
  sample();

  // ------------------------------------------------ timeline
  var staged = Array.prototype.slice.call(svg.querySelectorAll('.mf-s'));
  var LAST = 14, timers = [], resampler = null;
  // Panel (a) is stages 1-8 and panel (b) stages 9-14; both start together and
  // are paced to finish together.
  var A_LAST = 8, T0 = 300, A_STEP = 520, B_STEP = A_STEP * A_LAST / (LAST - A_LAST);
  function showAll(on) {
    staged.forEach(function (g) { g.classList.toggle('on', on); });
    inj.classList.toggle('on', on);
  }
  function showStage(n) {
    staged.forEach(function (g) { if (+g.dataset.s === n) g.classList.add('on'); });
    if (n === A_LAST) inj.classList.add('on');
  }
  function at(n) { return T0 + (n <= A_LAST ? n * A_STEP : (n - A_LAST) * B_STEP); }
  function play() {
    timers.forEach(clearTimeout); timers = [];
    clearInterval(resampler);
    if (reduceMotion) { showAll(true); return; }
    showAll(false);
    for (var n = 1; n <= LAST; n++) {
      (function (n) { timers.push(setTimeout(function () { showStage(n); }, at(n))); })(n);
    }
    // flicker while epsilon is drawn, then keep resampling once everything is shown
    [0, 120, 240, 360].forEach(function (d) { timers.push(setTimeout(sample, at(12) + d)); });
    timers.push(setTimeout(function () { resampler = setInterval(sample, 2600); }, at(LAST) + 2500));
  }

  var replay = document.getElementById('mech-replay');
  if (replay) replay.addEventListener('click', play);

  if (reduceMotion || !('IntersectionObserver' in window)) { play(); return; }
  showAll(false);
  var io = new IntersectionObserver(function (entries) {
    if (entries[0].isIntersecting) { io.disconnect(); play(); }
  }, { threshold: 0.25 });
  io.observe(host);
})();
