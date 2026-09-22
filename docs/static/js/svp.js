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


// ---- Semantic branching: animated LM-head readout trees (Evidence 2).
// PROBLEMS[*].methods[*].paths — one row per group of trajectories,
// [d0, d1, d2, d3, count]: the top-1 numeric readout at the four probed
// latent depths (cols) and how many trajectories share that route.
(function () {
  var host = document.getElementById('branch-tree');
  var toggle = document.getElementById('branch-toggle');
  var ptoggle = document.getElementById('branch-problem');
  if (!host || !toggle) return;

  var PROBLEMS = [
    {
      key: 'q334', tag: 'GSM8K-test #334',
      question: 'Nick, Richard, Jason and DJ each have paintball guns. DJ has 8 guns, Nick has 10 guns, RJ has 1 gun and Richard has 5 guns. If they were to share their guns equally, how many guns would each of them have?',
      solution: '8 + 10 + 1 + 5 = <b>24</b> &nbsp;&middot;&nbsp; 24 &divide; <b>4</b> = <b>6</b>',
      cols: [['h', '0', 'first operand'], ['h', '1', 'total'], ['h', '3', 'divisor'], ['h', '6', 'answer']],
      gold: [null, 24, 4, 6], answer: 6, critical: 1, critical_text: 'the correct total 24',
      story: 'Every method reads the first operand 8 at h<sub>0</sub> and re-converges on the divisor 4 at h<sub>3</sub>. The clean model reads 22 at h<sub>1</sub> instead of the total 24 and answers 7. Three Act-Gaussian samples do read 24, but none of them answers 6. Under SVP-V, 24 becomes the most common readout at h<sub>1</sub> (6 of 16), and four trajectories go on to the answer 6.',
      methods: {
        det: { name: 'Deterministic', color: '#555555', paths: [[8, 22, 4, 7, 1]] },
        agn: { name: 'Activation-Gaussian', color: '#1f77b4', paths: [
          [8, 22, 4, 7, 7], [8, 26, 4, 7, 3], [8, 24, 4, 8, 2], [8, 24, 4, 10, 1],
          [8, 28, 4, 8, 1], [8, 22, 4, 8, 1], [8, 26, 3, 8, 1]] },
        svp: { name: 'SVP-V', color: '#c62828', paths: [
          [8, 24, 4, 7, 3], [8, 24, 4, 6, 2], [8, 23, 2, 6, 1], [8, 22, 4, 6, 1],
          [8, 16, 4, 5, 1], [8, 23, 4, 7, 1], [4, 23, 4, 7, 1], [8, 26, 4, 8, 1],
          [8, 28, 4, 8, 1], [8, 20, 3, 7, 1], [8, 24, 4, 8, 1], [8, 23, 4, 8, 1],
          [8, 21, 4, 5, 1]] }
      }
    },
    {
      key: 'q397', tag: 'GSM8K-test #397',
      question: 'Mark has $50 in his bank account. He earns $10 per day at his work. If he wants to buy a bike that costs $300, how many days does Mark have to save his money?',
      solution: '300 &minus; 50 = <b>250</b> &nbsp;&middot;&nbsp; 250 &divide; <b>10</b> = <b>25</b>',
      cols: [['h', '0', 'target cost'], ['h', '1', 'remaining'], ['h', '3', 'per-day rate'], ['h', '6', 'answer']],
      gold: [null, 250, 10, 25], answer: 25, critical: 2, critical_text: 'the per-day rate 10',
      story: 'Every method reads the target cost 300 at h<sub>0</sub>. The clean model already drifts at h<sub>1</sub> (it reads 25, not the remaining 250) and then divides by the wrong number at h<sub>3</sub> — 50 instead of the $10/day rate — so it answers 30. The 16 Act-Gaussian samples never place 10 at h<sub>3</sub>. Under SVP-V the three trajectories that reach 250 → 10 are exactly the three that answer 25.',
      methods: {
        det: { name: 'Deterministic', color: '#555555', paths: [[300, 25, 50, 30, 1]] },
        agn: { name: 'Activation-Gaussian', color: '#1f77b4', paths: [
          [300, 30, 30, 30, 4], [300, 30, 0, 30, 3], [300, 30, 50, 30, 2], [300, 0, 0, 30, 1],
          [300, 0, 30, 30, 1], [300, 1, 75, 30, 1], [300, 3, 0, 30, 1], [300, 30, 1, 30, 1],
          [300, 30, 75, 30, 1], [300, 50, 75, 30, 1]] },
        svp: { name: 'SVP-V', color: '#c62828', paths: [
          [300, 30, 50, 30, 4], [300, 250, 10, 25, 3], [300, 30, 0, 30, 2], [50, 25, 50, 30, 1],
          [50, 30, 0, 30, 1], [50, 350, 100, 35, 1], [300, 1, 333, 50, 1], [300, 25, 50, 30, 1],
          [300, 30, 1, 30, 1], [300, 255, 30, 30, 1]] }
      }
    },
    {
      key: 'q853', tag: 'GSM8K-test #853',
      question: 'There are 90 rooms at the KozyInn Motel. It takes housekeeping 20 minutes to clean each room. How many hours would it take to clean one-half of the rooms?',
      solution: '90 &times; 20 = <b>1800</b> &nbsp;&middot;&nbsp; 1800 &divide; 2 = <b>900</b> &nbsp;&middot;&nbsp; 900 &divide; 60 = <b>15</b>',
      cols: [['h', '0', 'rooms'], ['h', '1', 'total minutes'], ['h', '4', 'half'], ['h', '6', 'answer (hours)']],
      gold: [null, 1800, 900, 15], answer: 15, critical: 2, critical_text: 'the half-total 900',
      story: 'The total 1800 minutes is read at h<sub>1</sub> by the clean model and by most samples of both methods — that step is not the problem. The clean model then reads 30 at h<sub>4</sub> (halving the 60-minute hour instead of the total) and answers 30. No Act-Gaussian sample reads 900 at h<sub>4</sub>; one SVP-V sample does and answers 15, and a second reaches 15 from 30.',
      methods: {
        det: { name: 'Deterministic', color: '#555555', paths: [[90, 1800, 30, 3, 1]] },
        agn: { name: 'Activation-Gaussian', color: '#1f77b4', paths: [
          [90, 45, 0, 0, 3], [90, 45, 7, 7, 3], [90, 1800, 30, 3, 3], [90, 1800, 30, 30, 3],
          [90, 1800, 3, 3, 2], [1, 45, 0, 0, 1], [90, 1800, 7, 7, 1]] },
        svp: { name: 'SVP-V', color: '#c62828', paths: [
          [90, 1800, 30, 3, 6], [90, 1800, 3, 3, 5], [90, 45, 22, 1, 1], [90, 45, 90, 1, 1],
          [90, 45, 900, 15, 1], [90, 1800, 30, 5, 1], [90, 1800, 30, 15, 1]] }
      }
    }
  ];
  var ROW = 34, NODE_H = 24, TOP = 30, MAX_ROWS = 8;
  var SVG = 'http://www.w3.org/2000/svg';
  var reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var current = 'svp', problem = PROBLEMS[0], started = false, narrow = null;

  function el(name, attrs, text) {
    var e = document.createElementNS(SVG, name);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    return e;
  }
  function byId(id) { return document.getElementById(id); }
  function setHTML(id, h) { var e = byId(id); if (e) e.innerHTML = h; }

  function build(key) {
    var P = problem, COLS = P.cols, GOLD = P.gold, ANSWER = P.answer, CRITICAL = P.critical;
    var m = P.methods[key];
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
      'aria-label': m.name + ': LM-head readouts of ' + total + ' latent trajectories, ' + P.tag });

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
    var crit = cols[CRITICAL], correct = 0;
    m.paths.forEach(function (p) { if (p[3] === ANSWER) correct += p[4]; });
    var sub = 'h<sub>' + COLS[CRITICAL][1] + '</sub>';
    setHTML('bt-stat-distinct', '<b>' + Object.keys(crit).length + '</b>distinct readouts at ' + sub);
    setHTML('bt-stat-total', '<b>' + (crit[GOLD[CRITICAL]] ? crit[GOLD[CRITICAL]].n : 0) + ' / ' + total +
      '</b>read ' + P.critical_text + ' at ' + sub);
    setHTML('bt-stat-correct', '<b>' + correct + ' / ' + total + '</b>reach the correct answer ' + ANSWER);
    return svg;
  }

  function play() {
    var svg = build(current);
    if (reduceMotion || !started) { if (started) svg.classList.add('is-playing'); return; }
    svg.getBoundingClientRect();     // flush, so the staged transitions run
    svg.classList.add('is-playing');
  }

  function showProblem(P) {
    problem = P;
    setHTML('bt-tag', P.tag);
    var q = byId('bt-question'); if (q) q.textContent = P.question;
    setHTML('bt-solution', P.solution);
    setHTML('bt-story', P.story);
    play();
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
  if (ptoggle) {
    var pbuttons = Array.prototype.slice.call(ptoggle.querySelectorAll('button'));
    pbuttons.forEach(function (b) {
      b.addEventListener('click', function () {
        pbuttons.forEach(function (o) {
          o.classList.toggle('is-active', o === b);
          o.setAttribute('aria-selected', o === b);
        });
        for (var i = 0; i < PROBLEMS.length; i++) {
          if (PROBLEMS[i].key === b.dataset.problem) showProblem(PROBLEMS[i]);
        }
      });
    });
  }
  var replay = byId('branch-replay');
  if (replay) replay.addEventListener('click', play);

  function layout() {
    var n = host.clientWidth < 560;
    if (n === narrow) return;
    narrow = n;
    showProblem(problem);
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
