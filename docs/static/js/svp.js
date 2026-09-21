// Interactive result widgets: model toggle for the coverage table, and the
// animated SVP-V-GRPO bar chart (data is read from #rl-table, so the table
// stays the single source of truth).
(function () {
  var reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  function activate(buttons, active) {
    buttons.forEach(function (b) {
      var on = b === active;
      b.classList.toggle('is-active', on);
      b.setAttribute('aria-selected', on);
    });
  }

  // ---- coverage table: one <tbody> per model
  var covToggle = document.getElementById('cov-toggle');
  if (covToggle) {
    var covButtons = Array.prototype.slice.call(covToggle.querySelectorAll('button'));
    var bodies = document.querySelectorAll('#cov-table tbody');
    covButtons.forEach(function (btn) {
      btn.addEventListener('click', function () {
        activate(covButtons, btn);
        bodies.forEach(function (tb) {
          var show = tb.dataset.model === btn.dataset.model;
          tb.hidden = !show;
          tb.classList.toggle('is-entering', show);
        });
      });
    });
  }

  // ---- GRPO bar chart
  var table = document.getElementById('rl-table');
  var barsEl = document.getElementById('rl-bars');
  var rlToggle = document.getElementById('rl-toggle');
  if (!table || !barsEl || !rlToggle) return;

  var benchmarks = Array.prototype.slice.call(table.querySelectorAll('thead th'), 1)
    .map(function (th) { return th.textContent.trim(); });
  var ablationBtn = document.getElementById('rl-ablation');
  var card = barsEl.closest('.svp-bars-card');
  var axisNote = document.getElementById('rl-axis-note');
  var rows = [];
  var inGrpo = false, started = false, current = 0, showAblation = false, toggleTimer;

  // Columns: released checkpoints + SVP-V-GRPO by default; the matched GRPO
  // controls (ablation) are revealed by the button.
  table.querySelectorAll('tbody tr').forEach(function (tr) {
    if (tr.classList.contains('group-head')) { inGrpo = rows.length > 0; return; }
    var cells = tr.querySelectorAll('td');
    var ours = tr.classList.contains('ours');
    var row = {
      values: Array.prototype.slice.call(cells, 1).map(function (td) { return parseFloat(td.textContent); }),
      grpo: inGrpo, ablation: inGrpo && !ours, shown: 0
    };

    var el = document.createElement('div');
    el.className = 'vbar-col' + (ours ? ' is-ours' : inGrpo ? ' is-grpo' : '');
    // ablation columns stay in the layout, folded to zero width, so they can animate
    el.classList.toggle('is-collapsed', row.ablation);
    if (row.ablation) el.setAttribute('aria-hidden', 'true');
    el.innerHTML = '<div class="vbar-plot">' +
      '<span class="bar-value">0.0</span><div class="bar"></div></div><div class="bar-label"></div>';
    el.querySelector('.bar-label').textContent = cells[0].textContent.replace(/\s*\(.*\)/, '');
    row.el = el;
    row.bar = el.querySelector('.bar');
    row.valueEl = el.querySelector('.bar-value');
    barsEl.appendChild(el);
    rows.push(row);
  });

  function countTo(row, target) {
    if (reduceMotion) { row.valueEl.textContent = target.toFixed(1); row.shown = target; return; }
    var from = row.shown, t0 = null;
    cancelAnimationFrame(row.raf);
    function step(t) {
      if (t0 === null) t0 = t;
      var k = Math.min((t - t0) / 800, 1);
      k = 1 - Math.pow(1 - k, 3);
      row.shown = from + (target - from) * k;
      row.valueEl.textContent = row.shown.toFixed(1);
      if (k < 1) row.raf = requestAnimationFrame(step);
    }
    row.raf = requestAnimationFrame(step);
  }

  function render(i) {
    current = i;
    if (!started) return;
    var visible = rows.filter(function (r) { return showAblation || !r.ablation; });
    // Common scale over all eight rows, so revealing the ablation never rescales the
    // bars. The axis is truncated (it starts a sixth of the data range below the
    // lowest value) to make the differences legible; the break is drawn and stated.
    var all = rows.map(function (r) { return r.values[i]; });
    var max = Math.max.apply(null, all), min = Math.min.apply(null, all);
    var lo = Math.max(0, Math.round((min - (max - min) / 6) * 10) / 10);
    if (axisNote) axisNote.textContent = 'Bars start at ' + lo.toFixed(1) + '%, not zero: the axis is truncated to make differences visible.';
    var best = Math.max.apply(null, visible.map(function (r) { return r.values[i]; }));
    visible.forEach(function (r) {
      var v = r.values[i];
      // leave ~20% of the plot height for the value labels
      r.bar.style.height = ((v - lo) / (max - lo) * 80).toFixed(2) + '%';
      r.el.classList.toggle('is-best', v === best);
      countTo(r, v);
    });
  }

  if (ablationBtn) {
    ablationBtn.addEventListener('click', function () {
      showAblation = !showAblation;
      ablationBtn.setAttribute('aria-pressed', showAblation);
      ablationBtn.classList.toggle('is-active', showAblation);
      ablationBtn.querySelector('.ablation-verb').textContent = showAblation ? 'Hide' : 'Show';
      card.classList.toggle('show-ablation', showAblation);
      // hold the labels on one line while the columns are mid-fold
      card.classList.add('is-toggling');
      clearTimeout(toggleTimer);
      toggleTimer = setTimeout(function () { card.classList.remove('is-toggling'); }, 550);
      rows.forEach(function (r) {
        if (!r.ablation) return;
        r.el.classList.toggle('is-collapsed', !showAblation);
        r.el.setAttribute('aria-hidden', !showAblation);
        if (!showAblation) {        // sink back to the axis while folding away
          cancelAnimationFrame(r.raf);
          r.bar.style.height = '0';
          r.shown = 0;
        }
      });
      render(current);
    });
  }

  var rlButtons = benchmarks.map(function (name, i) {
    var b = document.createElement('button');
    b.type = 'button';
    b.setAttribute('role', 'tab');
    b.textContent = name;
    b.addEventListener('click', function () { activate(rlButtons, b); render(i); });
    rlToggle.appendChild(b);
    return b;
  });
  activate(rlButtons, rlButtons[0]);

  // first animation plays when the chart scrolls into view
  function start() { started = true; render(current); }
  if ('IntersectionObserver' in window) {
    var io = new IntersectionObserver(function (entries) {
      if (entries[0].isIntersecting) { io.disconnect(); start(); }
    }, { threshold: 0.3 });
    io.observe(barsEl);
  } else {
    start();
  }
})();


// ---- Semantic branching: animated LM-head readout tree (GSM8K-test #334).
// Each entry is one group of trajectories: [h0, h1, h3, h6, count], the top-1
// numeric readout at each probed latent depth.
(function () {
  var host = document.getElementById('branch-tree');
  var toggle = document.getElementById('branch-toggle');
  if (!host || !toggle) return;

  var METHODS = {
    det: { name: 'Deterministic', color: '#555555', paths: [[8, 22, 4, 7, 1]] },
    agn: { name: 'Activation-Gaussian', color: '#1f77b4', paths: [
      [8, 22, 4, 7, 7], [8, 26, 4, 7, 3], [8, 24, 4, 8, 2], [8, 24, 4, 10, 1],
      [8, 28, 4, 8, 1], [8, 22, 4, 8, 1], [8, 26, 3, 8, 1]] },
    svp: { name: 'SVP-V', color: '#c62828', paths: [
      [8, 24, 4, 7, 3], [8, 24, 4, 6, 2], [8, 23, 2, 6, 1], [8, 22, 4, 6, 1],
      [8, 16, 4, 5, 1], [8, 23, 4, 7, 1], [4, 23, 4, 7, 1], [8, 26, 4, 8, 1],
      [8, 28, 4, 8, 1], [8, 20, 3, 7, 1], [8, 24, 4, 8, 1], [8, 23, 4, 8, 1],
      [8, 21, 4, 5, 1]] }
  };
  var COLS = [['h', '0', 'first operand'], ['h', '1', 'total'], ['h', '3', 'divisor'], ['h', '6', 'answer']];
  var GOLD = [null, 24, 4, 6];     // values on the correct solution path
  var ANSWER = 6, CRITICAL = 1, ROW = 34, NODE_H = 24, TOP = 30, MAX_ROWS = 8;
  var SVG = 'http://www.w3.org/2000/svg';
  var reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var current = 'svp', started = false, narrow = null;

  function el(name, attrs, text) {
    var e = document.createElementNS(SVG, name);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    return e;
  }

  function build(key) {
    var m = METHODS[key];
    var W = narrow ? 400 : 740, NW = narrow ? 52 : 62;
    var PAD = 18;   // room for the axis captions under the outer columns
    var xs = COLS.map(function (_, c) { return PAD + c * (W - 2 * PAD - NW) / (COLS.length - 1); });
    var plotH = MAX_ROWS * ROW, H = TOP + plotH + 44;
    var total = m.paths.reduce(function (s, p) { return s + p[4]; }, 0);

    // nodes per column, most common first (ties: smaller value first)
    var cols = COLS.map(function (_, c) {
      var counts = {};
      m.paths.forEach(function (p) { counts[p[c]] = (counts[p[c]] || 0) + p[4]; });
      var vals = Object.keys(counts).map(Number).sort(function (a, b) {
        return counts[b] - counts[a] || a - b;
      });
      var y0 = TOP + (plotH - vals.length * ROW) / 2 + ROW / 2;
      var nodes = {};
      vals.forEach(function (v, i) { nodes[v] = { v: v, n: counts[v], y: y0 + i * ROW }; });
      return nodes;
    });

    var svg = el('svg', { viewBox: '0 0 ' + W + ' ' + H, role: 'img',
      'aria-label': m.name + ': LM-head readouts of ' + total + ' latent trajectories at h0, h1, h3 and h6' });

    // the decision-critical column
    svg.appendChild(el('rect', { x: xs[CRITICAL] - 10, y: 4, width: NW + 20, height: H - 8, rx: 8, 'class': 'bt-critical' }));
    svg.appendChild(el('text', { x: xs[CRITICAL] + NW / 2, y: 19, 'text-anchor': 'middle', 'class': 'bt-critical-label' },
      narrow ? 'decision-critical' : 'decision-critical step'));

    // edges, thin ones first so the main routes sit on top
    for (var c = 0; c < COLS.length - 1; c++) {
      var edges = {};
      m.paths.forEach(function (p) {
        var id = p[c] + '>' + p[c + 1];
        var e = edges[id] || (edges[id] = { a: p[c], b: p[c + 1], n: 0, ok: false });
        e.n += p[4];
        if (p[3] === ANSWER) e.ok = true;
      });
      Object.keys(edges).map(function (id) { return edges[id]; })
        .sort(function (a, b) { return a.n - b.n; })
        .forEach(function (e) {
          var x1 = xs[c] + NW, x2 = xs[c + 1], mx = (x1 + x2) / 2;
          var y1 = cols[c][e.a].y, y2 = cols[c + 1][e.b].y;
          var path = el('path', {
            d: 'M' + x1 + ' ' + y1 + ' C' + mx + ' ' + y1 + ' ' + mx + ' ' + y2 + ' ' + x2 + ' ' + y2,
            pathLength: 1, 'class': 'bt-edge' + (e.ok ? ' is-ok' : ''),
            stroke: m.color, 'stroke-width': Math.max(1.4, e.n * 1.45).toFixed(1)
          });
          path.style.transitionDelay = (c * 0.7 + 0.2) + 's';
          svg.appendChild(path);
        });
    }

    // nodes and axis labels
    COLS.forEach(function (col, c) {
      Object.keys(cols[c]).forEach(function (v) {
        var node = cols[c][v], gold = GOLD[c] === node.v;
        var g = el('g', { 'class': 'bt-node' + (gold ? ' is-gold' : '') });
        g.style.transitionDelay = (c * 0.7) + 's';
        g.appendChild(el('rect', { x: xs[c], y: node.y - NODE_H / 2, width: NW, height: NODE_H, rx: 4 }));
        g.appendChild(el('text', { x: xs[c] + 8, y: node.y + 4.5, 'class': 'bt-val' }, node.v));
        if (total > 1) {
          g.appendChild(el('text', { x: xs[c] + NW - 6, y: node.y + 4, 'text-anchor': 'end', 'class': 'bt-count' }, '×' + node.n));
        }
        svg.appendChild(g);
      });
      var lab = el('text', { x: xs[c] + NW / 2, y: H - 24, 'text-anchor': 'middle', 'class': 'bt-axis' });
      lab.appendChild(el('tspan', { 'font-style': 'italic' }, col[0]));
      lab.appendChild(el('tspan', { dy: 3, 'font-size': '0.75em' }, col[1]));
      svg.appendChild(lab);
      svg.appendChild(el('text', { x: xs[c] + NW / 2, y: H - 8, 'text-anchor': 'middle', 'class': 'bt-axis-sub' }, col[2]));
    });

    host.innerHTML = '';
    host.appendChild(svg);

    // summary numbers for the selected method
    var h1 = cols[CRITICAL], correct = 0;
    m.paths.forEach(function (p) { if (p[3] === ANSWER) correct += p[4]; });
    document.getElementById('bt-stat-distinct').textContent = Object.keys(h1).length;
    document.getElementById('bt-stat-total').textContent = (h1[GOLD[CRITICAL]] ? h1[GOLD[CRITICAL]].n : 0) + ' / ' + total;
    document.getElementById('bt-stat-correct').textContent = correct + ' / ' + total;
    return svg;
  }

  function play() {
    var svg = build(current);
    if (reduceMotion || !started) { if (started) svg.classList.add('is-playing'); return; }
    svg.getBoundingClientRect();     // flush, so the staged transitions run
    svg.classList.add('is-playing');
  }

  var buttons = Array.prototype.slice.call(toggle.querySelectorAll('button'));
  buttons.forEach(function (b) {
    b.addEventListener('click', function () {
      buttons.forEach(function (o) {
        o.classList.toggle('is-active', o === b);
        o.setAttribute('aria-selected', o === b);
      });
      current = b.dataset.method;
      play();
    });
  });
  var replay = document.getElementById('branch-replay');
  if (replay) replay.addEventListener('click', play);

  function layout() {
    var n = host.clientWidth < 560;
    if (n === narrow) return;
    narrow = n;
    play();
  }
  window.addEventListener('resize', layout);
  layout();

  if ('IntersectionObserver' in window) {
    var io = new IntersectionObserver(function (entries) {
      if (entries[0].isIntersecting) { io.disconnect(); started = true; play(); }
    }, { threshold: 0.3 });
    io.observe(host);
  } else {
    started = true; play();
  }
})();


// Smooth open/close for the <details> disclosures (abstract, full table, method cards): the
// native toggle is instant, so animate the element's height instead.
(function () {
  var reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (reduceMotion || !Element.prototype.animate) return;

  document.querySelectorAll('details.svp-abstract, details.svp-fulltable, details.svp-collapse').forEach(function (d) {
    var summary = d.querySelector('summary');
    var expanded = d.open, anim = null;
    if (!summary) return;

    summary.addEventListener('click', function (e) {
      e.preventDefault();
      expanded = !expanded;
      var opening = expanded;
      var from = d.offsetHeight;          // mid-animation height if interrupted
      if (anim) anim.cancel();
      // measure the target before clipping: overflow:hidden would pull child margins in
      d.style.overflow = '';
      d.open = opening;
      var to = d.offsetHeight;
      d.open = true;                      // stay open while the height animates
      d.classList.toggle('is-closing', !opening);
      d.style.overflow = 'hidden';
      var a = anim = d.animate({ height: [from + 'px', to + 'px'] },
                               { duration: 400, easing: 'cubic-bezier(0.4, 0, 0.2, 1)' });
      a.onfinish = function () {
        if (anim !== a) return;
        if (!opening) d.open = false;
        d.classList.remove('is-closing');
        d.style.overflow = '';
        anim = null;
      };
    });
  });
})();
