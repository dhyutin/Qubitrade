// Qubitrade dashboard — Chart View + PPO Trader (Q-Atlas integrated inline).

const state = { ticker: "NVDA", timeframe: "ALL", forecast: 5, model: "lightgbm" };

const el = {
  tickers:    document.getElementById("tickers"),
  timeframes: document.getElementById("timeframes"),
  chart:      document.getElementById("chart"),
  skeleton:   document.getElementById("skeleton"),
  forecast:   document.getElementById("forecast"),
  forecastVal:document.getElementById("forecast-val"),
  model:      document.getElementById("model"),
  stats:      document.getElementById("stats"),
};

// Populated by renderModels — keeps full per-model info (desc + metrics) so
// the model-card can refresh on dropdown change without another fetch.
const modelInfo = new Map();

const fmtPct = (v) => (v == null ? "–" : `${v >= 0 ? "+" : ""}${v.toFixed(2)}%`);
const arrow = (v) => (v >= 0 ? "▴" : "▾");

async function boot() {
  const [meta, modelsResp] = await Promise.all([
    fetch("/api/tickers").then((r) => r.json()),
    fetch("/api/models").then((r) => r.json()),
  ]);
  renderTickers(meta.sectors);
  renderTimeframes(meta.timeframes);
  renderModels(modelsResp.models);

  el.forecast.addEventListener("input", () => {
    state.forecast = +el.forecast.value;
    el.forecastVal.textContent = state.forecast;
  });
  el.forecast.addEventListener("change", refresh);

  el.model.addEventListener("change", () => {
    state.model = el.model.value;
    refresh();
  });

  qatlas._bindDateControls();
  refresh();
  qatlas.queryForTicker(state.ticker);  // initial panel load for default ticker
}

function renderTickers(sectors) {
  el.tickers.innerHTML = "";
  for (const s of sectors) {
    const group = document.createElement("div");
    group.className = "sector-group";
    group.innerHTML = `<span class="sector-label">${s.label}</span>`;
    const chips = document.createElement("div");
    chips.className = "sector-chips";
    for (const t of s.tickers) {
      const chip = document.createElement("button");
      chip.className = "chip" + (t === state.ticker ? " active" : "");
      chip.textContent = t;
      chip.dataset.ticker = t;
      chip.onclick = () => selectTicker(t);
      chips.appendChild(chip);
    }
    group.appendChild(chips);
    el.tickers.appendChild(group);
  }
}

function renderTimeframes(tfs) {
  el.timeframes.innerHTML = "";
  for (const tf of tfs) {
    const b = document.createElement("button");
    b.className = "tf" + (tf === state.timeframe ? " active" : "");
    b.textContent = tf;
    b.dataset.tf = tf;
    b.onclick = () => selectTimeframe(tf);
    el.timeframes.appendChild(b);
  }
}

function renderModels(models) {
  el.model.innerHTML = "";
  modelInfo.clear();
  for (const m of models) {
    modelInfo.set(m.key, m);
    const opt = document.createElement("option");
    opt.value = m.key;
    opt.textContent = m.label;
    if (m.desc) opt.title = m.desc;
    if (m.key === state.model) opt.selected = true;
    el.model.appendChild(opt);
  }
  // Default to the first available model if the saved choice is gone.
  if (!modelInfo.has(state.model) && models.length) {
    state.model = models[0].key;
    el.model.value = state.model;
  }
  // Surface model metrics as dropdown option titles (tooltip on hover)
  el.model.querySelectorAll("option").forEach((opt) => {
    const m = modelInfo.get(opt.value);
    if (m?.metrics) {
      const mt = m.metrics;
      const f = (v, pct) => v == null ? "–" : pct ? `${(v*100).toFixed(1)}%` : v.toFixed(3);
      opt.title = `IC ${f(mt.ic)} · dir ${f(mt.dir_acc,true)} · Sharpe ${f(mt.sharpe)} — ${m.desc || ""}`;
    }
  });
}

function selectTicker(t) {
  const changed = t !== state.ticker;
  state.ticker = t;
  document.querySelectorAll(".chip").forEach((c) =>
    c.classList.toggle("active", c.dataset.ticker === t)
  );
  refresh(changed);
  qatlas.queryForTicker(t);
}

function selectTimeframe(tf) {
  state.timeframe = tf;
  document.querySelectorAll(".tf").forEach((b) =>
    b.classList.toggle("active", b.dataset.tf === tf)
  );
  refresh(false);
}

let inflight = 0;
async function refresh(reloadAnalogs = true) {
  const reqId = ++inflight;
  el.skeleton.hidden = false;

  try {
    const url = `/api/series?ticker=${state.ticker}&timeframe=${state.timeframe}&forecast_days=${state.forecast}&model=${state.model}`;
    const data = await fetch(url).then((r) => {
      if (!r.ok) throw new Error("request failed");
      return r.json();
    });
    if (reqId !== inflight) return;
    draw(data);
    renderStats(data);
  } catch (e) {
    if (reqId === inflight) {
      Plotly.purge(el.chart);
      el.chart.innerHTML = `<div style="padding:40px;color:#8b93a7">Couldn't load ${state.ticker}. ${e.message}</div>`;
    }
  } finally {
    if (reqId === inflight) el.skeleton.hidden = true;
  }
}

function renderStats(data) {
  const s = data.stats;
  const cards = [
    { label: `${data.ticker} last`, value: `$${s.last_price.toFixed(2)}`, cls: "" },
    {
      label: "Today",
      value: `${arrow(s.day_change_pct)} ${fmtPct(s.day_change_pct)}`,
      cls: s.day_change_pct >= 0 ? "up" : "down",
    },
    {
      label: `Predicted ${state.forecast}d`,
      value: `${arrow(s.pred_return_pct)} ${fmtPct(s.pred_return_pct)}`,
      cls: s.direction === "UP" ? "up" : "down",
      sub: `call: ${s.direction}`,
    },
    {
      label: "Overlap hit-rate",
      value: s.hit_rate_pct == null ? "–" : `${s.hit_rate_pct}%`,
      cls: "",
      sub: "predicted vs real direction",
    },
  ];
  el.stats.innerHTML = cards
    .map(
      (c) => `<div class="stat-card">
        <div class="stat-label">${c.label}</div>
        <div class="stat-value ${c.cls}">${c.value}</div>
        ${c.sub ? `<div class="stat-sub">${c.sub}</div>` : ""}
      </div>`
    )
    .join("");
}


function modelLabel(key) {
  const opt = el.model.querySelector(`option[value="${key}"]`);
  return opt ? opt.textContent : key;
}

function draw(data) {
  const p = data.prediction;

  // Overlap region (where a real bar exists under the prediction).
  const realSet = new Set(data.real.dates);
  const overlapLen = p.pred_dates.filter((d) => realSet.has(d)).length;
  const overlapDates = p.pred_dates.slice(0, overlapLen);
  const predOverlap = p.pred_close.slice(0, overlapLen);
  const realOverlap = data.real.close.slice(-overlapLen);

  const up = data.stats.direction === "UP";
  const divColor = up ? "rgba(31,157,77,0.18)" : "rgba(224,82,79,0.16)";

  const bandLower = {
    x: p.pred_dates, y: p.pred_lower, mode: "lines",
    line: { width: 0 }, showlegend: false, hoverinfo: "skip",
  };
  const bandUpper = {
    x: p.pred_dates, y: p.pred_upper, mode: "lines",
    line: { width: 0 }, fill: "tonexty", fillcolor: "rgba(31,157,77,0.12)",
    showlegend: false, hoverinfo: "skip", name: "confidence",
  };
  const realOverlapTrace = {
    x: overlapDates, y: realOverlap, mode: "lines",
    line: { width: 0 }, showlegend: false, hoverinfo: "skip",
  };
  const divergence = {
    x: overlapDates, y: predOverlap, mode: "lines",
    line: { width: 0 }, fill: "tonexty", fillcolor: divColor,
    showlegend: false, hoverinfo: "skip",
  };
  const real = {
    x: data.real.dates, y: data.real.close, mode: "lines", name: "Real",
    line: { color: "#e7ebf3", width: 2.5 },
    hovertemplate: "$%{y:.2f}<extra>real</extra>",
  };
  const pred = {
    x: p.pred_dates, y: p.pred_close, mode: "lines", name: "Prediction",
    line: { color: "#1f9d4d", width: 3 },
    hovertemplate: "$%{y:.2f}<extra>prediction</extra>",
  };

  const shapes = [];
  const annotations = [];

  // Train region (blue tint)
  if (data.data_start && data.train_end) {
    shapes.push({
      type: "rect", xref: "x", yref: "paper",
      x0: data.data_start, x1: data.train_end,
      y0: 0, y1: 1,
      fillcolor: "rgba(79,124,255,0.07)", line: { width: 0 }, layer: "below",
    });
    annotations.push({
      x: data.data_start, y: 0.97, xref: "x", yref: "paper",
      text: "Train  2013–2018", showarrow: false,
      font: { color: "#4f7cff", size: 10 }, xanchor: "left",
    });
  }

  // Test region (amber tint)
  if (data.test_start && data.data_end) {
    shapes.push({
      type: "rect", xref: "x", yref: "paper",
      x0: data.test_start, x1: data.data_end,
      y0: 0, y1: 1,
      fillcolor: "rgba(255,183,77,0.09)", line: { width: 0 }, layer: "below",
    });
    annotations.push({
      x: data.data_end, y: 0.97, xref: "x", yref: "paper",
      text: "Test  2019–2020", showarrow: false,
      font: { color: "#ffb74d", size: 10 }, xanchor: "right",
    });
  }

  // Train/test boundary line
  if (data.test_start) {
    shapes.push({
      type: "line", x0: data.test_start, x1: data.test_start,
      yref: "paper", y0: 0, y1: 1,
      line: { color: "#4f7cff", width: 1, dash: "dash" },
    });
  }

  // Model split-date line (where overlap region starts)
  if (p.split_date) {
    shapes.push({
      type: "line", x0: p.split_date, x1: p.split_date,
      yref: "paper", y0: 0, y1: 1,
      line: { color: "#3a4152", width: 1, dash: "dot" },
    });
  }

  const layout = {
    margin: { l: 52, r: 18, t: 30, b: 36 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: "#8b93a7", size: 12 },
    title: {
      text: `${data.ticker} · ${data.sector}${data.model ? "  ·  " + modelLabel(data.model) : ""}`,
      font: { color: "#e7ebf3", size: 15 }, x: 0.01,
    },
    hovermode: "x unified",
    xaxis: {
      gridcolor: "#1e222e", showline: false, zeroline: false,
      showspikes: true, spikemode: "across", spikethickness: 1,
      spikecolor: "#4f7cff", spikedash: "solid",
    },
    yaxis: {
      gridcolor: "#1e222e", tickprefix: "$", showline: false, zeroline: false,
    },
    legend: {
      orientation: "h", x: 1, xanchor: "right", y: 1.12,
      font: { color: "#e7ebf3" },
    },
    shapes,
    annotations,
    showlegend: true,
    transition: { duration: 350, easing: "cubic-in-out" },
  };

  Plotly.react(
    el.chart,
    [bandLower, bandUpper, realOverlapTrace, divergence, real, pred],
    layout,
    { responsive: true, displayModeBar: false }
  );
}

boot();

// ═══════════════════════════════════════════════════════
//  PPO TRADER
// ═══════════════════════════════════════════════════════

// ── Tab switcher ──────────────────────────────────────
document.querySelectorAll(".vtab").forEach((btn) => {
  btn.addEventListener("click", () => {
    const view = btn.dataset.view;
    document.querySelectorAll(".vtab").forEach((b) => b.classList.toggle("active", b === btn));
    document.getElementById("view-chart").hidden = view !== "chart";
    document.getElementById("view-ppo").hidden   = view !== "ppo";
    if (view === "ppo" && !ppo.loaded) ppo.init();
  });
});

// ── PPO module (self-contained) ────────────────────────
const ppo = {
  loaded:    false,
  traj:      null,   // full trajectory from server { tickers, steps, summary }
  view:      [],     // filtered slice of traj.steps (by date range)
  step:      0,      // index within view[]
  mode:      "historical",
  userPos:   [],     // per-ticker: +1 / 0 / -1
  userCum:   [],     // accumulated user cumulative return per view step
  playing:   false,
  playTimer: null,

  async init() {
    if (this.loaded) return;
    await this._loadMode("historical");
    this._bindControls();
    this._bindModeButtons();
    this._bindDateRange();
  },

  async _loadMode(mode) {
    const bar = document.getElementById("ppo-summary-bar");
    const badge = document.getElementById("ppo-mode-badge");
    const url = mode === "live" ? "/api/ppo/live" : "/api/ppo/trajectory";
    bar.textContent = mode === "live"
      ? "Fetching live yfinance data + running LightGBM + PPO (first load ~15 s)…"
      : "Loading PPO historical trajectory…";
    badge.textContent = mode;
    badge.className = "ppo-mode-badge " + mode;
    try {
      const r = await fetch(url);
      if (!r.ok) throw new Error(await r.text());
      this.traj = await r.json();
    } catch (e) {
      bar.textContent = "PPO backend unavailable: " + e.message;
      return;
    }
    this.mode   = mode;
    this.loaded = true;

    // Default date range = full window
    this._setDateRangeInputs(this.traj.steps[0].date, this.traj.steps.at(-1).date);
    this._applyDateRange();   // sets this.view, resets step, renders everything
  },

  // ── Date range helpers ───────────────────────────────
  _setDateRangeInputs(start, end) {
    document.getElementById("ppo-range-start").value = start;
    document.getElementById("ppo-range-end").value   = end;
  },

  _applyDateRange() {
    const start = document.getElementById("ppo-range-start").value;
    const end   = document.getElementById("ppo-range-end").value;
    this.view = this.traj.steps.filter(
      (s) => (!start || s.date >= start) && (!end || s.date <= end)
    );
    if (!this.view.length) this.view = [...this.traj.steps];
    this.userPos = new Array(this.traj.tickers.length).fill(0);
    this.userCum = new Array(this.view.length).fill(0);
    this._renderSummary();
    if (!document.getElementById("ppo-ticker-rows").children.length) {
      this._renderTickerRows();
    } else {
      this.traj.tickers.forEach((_, i) => this._updatePosBtn(i));
    }
    this.goTo(0);
  },

  // ── Render overview bar ─────────────────────────────
  _renderSummary() {
    const s    = this.traj.summary;
    const fmtP = (v) => `${v >= 0 ? "+" : ""}${(v * 100).toFixed(1)}%`;
    // Compute stats for the current view (date-filtered slice)
    const vPPO = this.view.reduce((c, s) => (1 + c) * (1 + s.ppo_return) - 1, 0);
    const vEW  = this.view.reduce((c, s) => (1 + c) * (1 + s.ew_return)  - 1, 0);
    const vR   = this.view.map((s) => s.ppo_return);
    const vSh  = vR.length > 1
      ? (vR.reduce((a,v)=>a+v,0)/vR.length / (Math.sqrt(vR.reduce((a,v)=>a+(v-vR.reduce((a2,v2)=>a2+v2,0)/vR.length)**2,0)/vR.length)+1e-9) * Math.sqrt(252)).toFixed(2)
      : "–";
    document.getElementById("ppo-summary-bar").innerHTML = `
      <span>${this.traj.version}</span>
      <span class="ppo-summary-sep">|</span>
      <span>Window: <b>${this.view[0]?.date ?? s.test_start}</b> → <b>${this.view.at(-1)?.date ?? s.test_end}</b></span>
      <span class="ppo-summary-sep">|</span>
      <span>PPO Sharpe <b style="color:var(--green)">${vSh}</b></span>
      <span>EW-LGB Sharpe <b>${s.ew_sharpe}</b></span>
      <span class="ppo-summary-sep">|</span>
      <span>PPO return <b style="color:var(--green)">${fmtP(vPPO)}</b></span>
      <span>EW-LGB <b>${fmtP(vEW)}</b></span>
      <span class="ppo-summary-sep">|</span>
      <span style="color:var(--muted);font-size:11px">${this.view.length} days shown · ${s.n_steps} total</span>
    `;
  },

  // ── Render ticker rows (once, weights updated later) ─
  _renderTickerRows() {
    const wrap = document.getElementById("ppo-ticker-rows");
    wrap.innerHTML = this.traj.tickers.map((tk, i) => `
      <div class="ppo-ticker-row" id="ppo-row-${i}">
        <span class="ppo-tk-name">${tk}</span>
        <div class="ppo-bar-wrap">
          <div class="ppo-bar-center"></div>
          <div class="ppo-bar-fill" id="ppo-bar-${i}"></div>
        </div>
        <button class="ppo-pos-btn" id="ppo-pos-${i}" data-idx="${i}">0</button>
      </div>`
    ).join("");
    // Position toggle: 0 → long → short → 0
    document.getElementById("ppo-ticker-rows").querySelectorAll(".ppo-pos-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        const i = +btn.dataset.idx;
        this.userPos[i] = this.userPos[i] === 0 ? 1 : this.userPos[i] === 1 ? -1 : 0;
        this._updatePosBtn(i);
        this._recomputeUserCum();
        this._drawChart();
        this._renderAnalytics();
      });
    });
  },

  // ── Bind nav + action buttons ────────────────────────
  _bindControls() {
    document.getElementById("ppo-prev").addEventListener("click", () => this.step > 0 && this.goTo(this.step - 1));
    document.getElementById("ppo-next").addEventListener("click", () => this.step < this.view.length - 1 && this.goTo(this.step + 1));
    document.getElementById("ppo-play").addEventListener("click", () => this.playing ? this._stopPlay() : this._startPlay());
    document.getElementById("ppo-copy").addEventListener("click", () => this._copyPPO());
    document.getElementById("ppo-blind").addEventListener("click", () => this._blindRunFromHere());
    document.getElementById("ppo-reset").addEventListener("click", () => this._resetPositions());
  },

  // ── Bind mode toggle buttons ─────────────────────────
  _bindModeButtons() {
    document.getElementById("ppo-mode-hist").addEventListener("click", async () => {
      if (this.mode === "historical") return;
      document.getElementById("ppo-mode-hist").classList.add("active");
      document.getElementById("ppo-mode-live").classList.remove("active");
      this.loaded = false;
      this._stopPlay();
      await this._loadMode("historical");
    });
    document.getElementById("ppo-mode-live").addEventListener("click", async () => {
      if (this.mode === "live") return;
      document.getElementById("ppo-mode-live").classList.add("active");
      document.getElementById("ppo-mode-hist").classList.remove("active");
      this.loaded = false;
      this._stopPlay();
      await this._loadMode("live");
    });
  },

  // ── Bind date range controls ─────────────────────────
  _bindDateRange() {
    document.getElementById("ppo-range-apply").addEventListener("click", () => {
      this._stopPlay();
      this.userPos = new Array(this.traj.tickers.length).fill(0);
      this._applyDateRange();
    });
    document.getElementById("ppo-range-reset").addEventListener("click", () => {
      this._stopPlay();
      this._setDateRangeInputs(this.traj.steps[0].date, this.traj.steps.at(-1).date);
      this.userPos = new Array(this.traj.tickers.length).fill(0);
      this._applyDateRange();
    });
  },

  // ── Navigate to a step ──────────────────────────────
  goTo(n) {
    this.step = Math.max(0, Math.min(n, this.view.length - 1));
    const s = this.view[this.step];
    document.getElementById("ppo-step-cur").textContent   = this.step + 1;
    document.getElementById("ppo-step-total").textContent = this.view.length;
    document.getElementById("ppo-date").textContent        = s.date;
    this._updateWeightBars(s.ppo_weights);
    this._drawChart();
    this._renderAnalytics();
  },

  // ── Update PPO weight bars ───────────────────────────
  _updateWeightBars(weights) {
    const maxW = Math.max(...weights.map(Math.abs), 0.01);
    weights.forEach((w, i) => {
      const bar = document.getElementById("ppo-bar-" + i);
      if (!bar) return;
      const pct = Math.abs(w) / maxW * 48;  // max 48% of half-bar
      bar.className = "ppo-bar-fill " + (w >= 0 ? "long" : "short");
      bar.style.width = pct + "%";
    });
  },

  // ── Update a single user-position button ────────────
  _updatePosBtn(i) {
    const btn = document.getElementById("ppo-pos-" + i);
    if (!btn) return;
    const v = this.userPos[i];
    btn.textContent = v === 1 ? "LONG" : v === -1 ? "SHORT" : "0";
    btn.className   = "ppo-pos-btn" + (v === 1 ? " long" : v === -1 ? " short" : "");
  },

  // ── (Re)compute user cumulative returns over all view steps ──
  _recomputeUserCum() {
    let cum = 0;
    const l1 = this.userPos.reduce((s, v) => s + Math.abs(v), 0) || 1;
    const norm = this.userPos.map((v) => v / l1);
    this.userCum = this.view.map((s) => {
      const dayRet = s.fwd_returns.reduce((acc, r, i) => acc + norm[i] * r, 0);
      cum = (1 + cum) * (1 + dayRet) - 1;
      return cum;
    });
  },

  // ── Cumulative series helpers (over view) ────────────
  _ppoCum(from = 0, to = null) {
    const end = to ?? this.view.length;
    let cum = 0;
    return this.view.slice(from, end).map((s) => {
      cum = (1 + cum) * (1 + s.ppo_return) - 1;
      return cum;
    });
  },
  _ewCum(from = 0, to = null) {
    const end = to ?? this.view.length;
    let cum = 0;
    return this.view.slice(from, end).map((s) => {
      cum = (1 + cum) * (1 + s.ew_return) - 1;
      return cum;
    });
  },

  // ── Draw the cumulative return chart ─────────────────
  _drawChart() {
    const steps = this.view;
    const dates = steps.map((s) => s.date);
    const ppoCum = this._ppoCum();
    const ewCum  = this._ewCum();

    const traces = [
      {
        x: dates, y: ppoCum.map((v) => +(v * 100).toFixed(3)),
        mode: "lines", name: "PPO",
        line: { color: "#1f9d4d", width: 2.5 },
        hovertemplate: "%{y:.2f}%<extra>PPO</extra>",
      },
      {
        x: dates, y: ewCum.map((v) => +(v * 100).toFixed(3)),
        mode: "lines", name: "EW-LGB",
        line: { color: "#4f7cff", width: 2, dash: "dot" },
        hovertemplate: "%{y:.2f}%<extra>EW-LGB</extra>",
      },
    ];

    // User line only if they have any positions set
    if (this.userPos.some((v) => v !== 0)) {
      traces.push({
        x: dates, y: this.userCum.map((v) => +(v * 100).toFixed(3)),
        mode: "lines", name: "You",
        line: { color: "#ffb74d", width: 2.5 },
        hovertemplate: "%{y:.2f}%<extra>You</extra>",
      });
    }

    const shapes = [{
      type: "line", x0: steps[this.step].date, x1: steps[this.step].date,
      yref: "paper", y0: 0, y1: 1,
      line: { color: "#8b93a7", width: 1.5, dash: "dash" },
    }];

    Plotly.react("ppo-chart", traces, {
      margin: { l: 52, r: 18, t: 30, b: 36 },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { color: "#8b93a7", size: 12 },
      title: { text: "Cumulative Return — PPO vs EW-LGB vs You", font: { color: "#e7ebf3", size: 14 }, x: 0.01 },
      hovermode: "x unified",
      xaxis: { gridcolor: "#1e222e", showline: false, zeroline: false },
      yaxis: { gridcolor: "#1e222e", ticksuffix: "%", showline: false, zeroline: true, zerolinecolor: "#2a2f3d" },
      legend: { orientation: "h", x: 1, xanchor: "right", y: 1.12, font: { color: "#e7ebf3" } },
      shapes,
      showlegend: true,
      transition: { duration: 200, easing: "cubic-in-out" },
    }, { responsive: true, displayModeBar: false });
  },

  // ── Analytics strip ──────────────────────────────────
  _renderAnalytics() {
    const n   = this.step;
    const tot = this.view.length;

    // Cumulative so far (steps 0..n)
    const ppoCumNow  = this._ppoCum(0, n + 1).at(-1) ?? 0;
    const ewCumNow   = this._ewCum(0, n + 1).at(-1) ?? 0;
    const userCumNow = this.userCum[n] ?? 0;

    // "If you follow PPO from HERE to end" (steps n..end)
    let ppoFromHere = 1;
    for (let i = n; i < tot; i++) ppoFromHere *= (1 + this.view[i].ppo_return);
    ppoFromHere -= 1;

    // Sharpe of user (if positions set)
    let userSharpe = null;
    if (this.userPos.some((v) => v !== 0)) {
      const l1   = this.userPos.reduce((s, v) => s + Math.abs(v), 0) || 1;
      const norm = this.userPos.map((v) => v / l1);
      const rets = this.view.map((s) => s.fwd_returns.reduce((a, r, i) => a + norm[i] * r, 0));
      const mean = rets.reduce((a, v) => a + v, 0) / rets.length;
      const std  = Math.sqrt(rets.reduce((a, v) => a + (v - mean) ** 2, 0) / rets.length) + 1e-9;
      userSharpe = +(mean / std * Math.sqrt(252)).toFixed(2);
    }

    const fmtPct = (v) => `<span class="${v >= 0 ? "up" : "down"}">${v >= 0 ? "+" : ""}${(v * 100).toFixed(2)}%</span>`;
    const card = (label, val, sub = "") => `
      <div class="ppo-an-card">
        <div class="ppo-an-label">${label}</div>
        <div class="ppo-an-val">${val}</div>
        ${sub ? `<div class="ppo-an-sub">${sub}</div>` : ""}
      </div>`;

    document.getElementById("ppo-analytics-strip").innerHTML = [
      card("Your return (so far)",
        this.userPos.some((v) => v !== 0) ? fmtPct(userCumNow) : "–",
        userSharpe !== null ? `Sharpe ${userSharpe}` : "set positions above"),
      card("PPO return (so far)",     fmtPct(ppoCumNow),  `Day ${n + 1}/${tot}`),
      card("EW-LGB return (so far)",  fmtPct(ewCumNow),   "benchmark"),
      card("Follow PPO → end",        fmtPct(ppoFromHere), `from today (${tot - n} days left)`),
      card("Blind PPO (full run)",    fmtPct(this.traj.summary.ppo_total_return),
           `Sharpe ${this.traj.summary.ppo_sharpe}`),
    ].join("");
  },

  // ── Action: copy PPO weights as user positions ───────
  _copyPPO() {
    const weights = this.view[this.step].ppo_weights;
    // Binarize: top-N long, bottom-N short; rest neutral
    const indexed = weights.map((w, i) => [i, w]).sort((a, b) => a[1] - b[1]);
    const n = Math.max(1, Math.floor(this.traj.tickers.length / 5));
    this.userPos = new Array(this.traj.tickers.length).fill(0);
    for (let k = 0; k < n; k++) {
      this.userPos[indexed[k][0]]                         = -1;  // short bottom-N
      this.userPos[indexed[indexed.length - 1 - k][0]]   = +1;  // long  top-N
    }
    this.traj.tickers.forEach((_, i) => this._updatePosBtn(i));
    this._recomputeUserCum();
    this._drawChart();
    this._renderAnalytics();
  },

  // ── Action: follow PPO from current step to end ──────
  _blindRunFromHere() {
    // From step N onward, use PPO weights (binarized)
    const curStep = this.step;
    // Rebuild userCum: keep user returns for 0..curStep, then follow PPO
    let cum = this.userCum[curStep - 1] ?? 0;
    const l1 = this.userPos.reduce((s, v) => s + Math.abs(v), 0) || 1;
    const norm = this.userPos.map((v) => v / l1);

    for (let i = curStep; i < this.view.length; i++) {
      const s = this.view[i];
      // Use raw PPO return (already L1-normalized in backend)
      cum = (1 + cum) * (1 + s.ppo_return) - 1;
      this.userCum[i] = cum;
      // Mark userPos as "ppo" for display
      const weights = s.ppo_weights;
      const indexed = weights.map((w, j) => [j, w]).sort((a, b) => a[1] - b[1]);
      const n = Math.max(1, Math.floor(this.traj.tickers.length / 5));
      const nextPos = new Array(this.traj.tickers.length).fill(0);
      for (let k = 0; k < n; k++) {
        nextPos[indexed[k][0]]                       = -1;
        nextPos[indexed[indexed.length - 1 - k][0]] = +1;
      }
      if (i === curStep) {
        this.userPos = nextPos;
        this.traj.tickers.forEach((_, idx) => this._updatePosBtn(idx));
      }
    }
    this._drawChart();
    this._renderAnalytics();
  },

  // ── Action: reset user positions to neutral ──────────
  _resetPositions() {
    this.userPos = new Array(this.traj.tickers.length).fill(0);
    this.userCum = new Array(this.view.length).fill(0);
    this.traj.tickers.forEach((_, i) => this._updatePosBtn(i));
    this._drawChart();
    this._renderAnalytics();
  },

  // ── Auto-play ────────────────────────────────────────
  _startPlay() {
    const btn = document.getElementById("ppo-play");
    btn.textContent = "⏸ Pause";
    btn.classList.add("accent");
    this.playing = true;
    const tick = () => {
      if (!this.playing) return;
      if (this.step >= this.view.length - 1) { this._stopPlay(); return; }
      this.goTo(this.step + 1);
      this.playTimer = setTimeout(tick, 120);
    };
    tick();
  },
  _stopPlay() {
    this.playing = false;
    clearTimeout(this.playTimer);
    const btn = document.getElementById("ppo-play");
    btn.textContent = "▶ Play";
    btn.classList.remove("accent");
  },
};

// ═══════════════════════════════════════════════════════
//  Q-ATLAS — Inline Quantum Analog Panel
//  Fires automatically when a ticker is selected.
// ═══════════════════════════════════════════════════════

const qatlas = {
  _inflight: null,
  _liveMode: false,
  _newsData: [],     // [{date, sim, news_items}] for the drawer

  _bindDateControls() {
    const dateEl  = document.getElementById("qa-panel-date");
    const clearEl = document.getElementById("qa-panel-clear-date");
    const runBtn  = document.getElementById("qa-panel-run");
    const liveBtn = document.getElementById("qa-live-btn");
    const newsToggle = document.getElementById("qa-news-toggle");
    const newsClose  = document.getElementById("qa-news-close");

    dateEl?.addEventListener("change", () => {
      const d = dateEl.value;
      if (d) {
        clearEl.hidden = false;
        this._zoomToDate(d);
      } else {
        clearEl.hidden = true;
        this._resetZoom();
      }
    });

    clearEl?.addEventListener("click", () => {
      dateEl.value = "";
      clearEl.hidden = true;
      this._resetZoom();
    });

    runBtn?.addEventListener("click", () => this.queryForTicker(state.ticker));

    liveBtn?.addEventListener("click", () => {
      this._liveMode = !this._liveMode;
      liveBtn.classList.toggle("active", this._liveMode);
      // disable/enable date picker
      if (dateEl) dateEl.disabled = this._liveMode;
      this.queryForTicker(state.ticker);
    });

    newsToggle?.addEventListener("click", () => this._openNewsDrawer());
    newsClose?.addEventListener("click",  () => {
      document.getElementById("qa-news-drawer").hidden = true;
    });
  },

  _openNewsDrawer() {
    const drawer  = document.getElementById("qa-news-drawer");
    const content = document.getElementById("qa-news-content");
    if (!this._newsData.length) return;
    content.innerHTML = this._newsData.map((g) => `
      <div class="qa-news-group">
        <div class="qa-news-group-date">
          #${g.rank} ${g.date}
          <span class="qa-news-group-sim">sim ${g.sim.toFixed(3)}</span>
        </div>
        ${g.news_items.length
          ? g.news_items.map((h) => `<div class="qa-news-item">${h}</div>`).join("")
          : `<div class="qa-news-item" style="opacity:0.5;font-style:italic">No headlines recorded for this date.</div>`
        }
      </div>`).join("");
    drawer.hidden = false;
  },

  _zoomToDate(dateStr) {
    const center = new Date(dateStr);
    // Show ±90 days around the selected date
    const start = new Date(center); start.setDate(start.getDate() - 90);
    const end   = new Date(center); end.setDate(end.getDate()   + 90);
    const fmt = (d) => d.toISOString().slice(0, 10);
    Plotly.relayout(el.chart, { "xaxis.range": [fmt(start), fmt(end)] });
  },

  _resetZoom() {
    Plotly.relayout(el.chart, { "xaxis.autorange": true });
  },

  async queryForTicker(ticker) {
    this._inflight = ticker;
    const date = document.getElementById("qa-panel-date")?.value || null;
    const statusEl    = document.getElementById("qa-panel-status");
    const narrativeEl = document.getElementById("qa-panel-narrative");
    const consensusEl = document.getElementById("qa-panel-consensus");
    const cardsEl     = document.getElementById("qa-panel-cards");
    const methodEl    = document.getElementById("qa-panel-method");
    const extEl       = document.getElementById("qa-extended");

    statusEl.className = "qa-panel-status running";
    statusEl.textContent = "running quantum kernel…";
    narrativeEl.innerHTML = `<span class="qa-panel-empty">Computing ZZFeatureMap kernel for ${ticker}${date ? " · " + date : ""}…</span>`;
    if (consensusEl) consensusEl.hidden = true;
    cardsEl.innerHTML = "";
    // Clear only the text span — don't wipe the news-toggle button
    const methodSpan = methodEl?.querySelector(".qa-method-text");
    if (methodSpan) methodSpan.textContent = "";
    if (extEl) extEl.hidden = true;

    try {
      const url = `/api/qatlas?ticker=${ticker}${date && !this._liveMode ? "&date=" + date : ""}${this._liveMode ? "&live=true" : ""}`;
      const data = await fetch(url).then((r) => {
        if (!r.ok) throw new Error(`server error ${r.status}`);
        return r.json();
      });
      if (this._inflight !== ticker) return;

      statusEl.className = "qa-panel-status done";
      statusEl.textContent = `${data.query_date} · ${data.n_candidates.toLocaleString()} candidates`;
      this._renderInline(data);
    } catch (e) {
      if (this._inflight !== ticker) return;
      statusEl.className = "qa-panel-status error";
      statusEl.textContent = "error";
      narrativeEl.innerHTML = `<span class="qa-panel-empty" style="color:var(--red)">Q-Atlas error: ${e.message}</span>`;
    }
  },

  _renderInline(data) {
    const narrativeEl  = document.getElementById("qa-panel-narrative");
    const consensusEl  = document.getElementById("qa-panel-consensus");
    const cardsEl      = document.getElementById("qa-panel-cards");
    const methodEl     = document.getElementById("qa-panel-method");
    const extEl        = document.getElementById("qa-extended");
    const extMethodEl  = document.getElementById("qa-extended-method");
    const newsToggle   = document.getElementById("qa-news-toggle");

    // ── Stash news data for the drawer ──────────────────────────
    this._newsData = data.analogs
      .filter((a) => a.news_items?.length)
      .map((a) => ({ rank: a.rank, date: a.date, sim: a.similarity, news_items: a.news_items }));
    if (newsToggle) newsToggle.hidden = this._newsData.length === 0;

    // ── Gemini narrative ────────────────────────────────────────
    const liveBadge = data.is_live ? `<span class="qa-live-badge">LIVE</span>` : "";
    narrativeEl.innerHTML = "";
    // Live signals strip
    if (data.is_live) {
      const qs = data.query_state;
      const fmt = (v) => v !== undefined ? `${v >= 0 ? "+" : ""}${Number(v).toFixed(3)}` : "n/a";
      const sigDiv = document.createElement("div");
      sigDiv.className = "qa-live-signals";
      sigDiv.innerHTML = `
        <div class="qa-live-signals-label">⚡ Live signals · ${data.query_date}</div>
        <div class="qa-live-signals-kv">
          <span>vol_20d <b>${fmt(qs.vol_20d)}</b></span>
          <span>mom_20d <b>${fmt(qs.mom_20d)}</b></span>
          <span>dist_hi <b>${fmt(qs.dist_to_high)}</b></span>
          <span>news_sent <b>${fmt(qs.news_sent)}</b></span>
        </div>`;
      narrativeEl.appendChild(sigDiv);
      // Today's live headlines
      if (data.live_headlines?.length) {
        const hlDiv = document.createElement("div");
        hlDiv.className = "qa-live-headlines";
        hlDiv.innerHTML = `<div class="qa-live-headlines-label">Today's headlines</div>` +
          data.live_headlines.map((h) => `<div class="qa-live-headline-item">${h}</div>`).join("");
        narrativeEl.appendChild(hlDiv);
      }
      const narDiv = document.createElement("div");
      narDiv.style.cssText = "padding:10px 14px;font-size:12.5px;line-height:1.6";
      narDiv.textContent = data.narrative || "Narrative not available.";
      narrativeEl.appendChild(narDiv);
    } else {
      narrativeEl.textContent = data.narrative || "Narrative not available.";
    }

    // ── Consensus badge ─────────────────────────────────────────
    const fwds = data.analogs.map((a) => a.fwd_5d_pct).filter((v) => v != null);
    const avg  = fwds.length ? fwds.reduce((a, v) => a + v, 0) / fwds.length : 0;
    const nPos = fwds.filter((v) => v > 0).length;
    const dir  = avg > 0.5 ? "BULLISH" : avg < -0.5 ? "BEARISH" : "NEUTRAL";
    consensusEl.hidden = false;
    consensusEl.innerHTML = `
      <div>
        <div class="qa-consensus-direction ${dir.toLowerCase()}">⚛ ${dir}</div>
        <div class="qa-consensus-meta" style="margin-top:2px">${nPos}/${fwds.length} analogs positive</div>
      </div>
      <div style="text-align:right">
        <div style="font-size:18px;font-weight:800;color:${avg >= 0 ? "var(--green)" : "var(--red)"}">
          ${avg >= 0 ? "+" : ""}${avg.toFixed(1)}%
        </div>
        <div class="qa-consensus-meta">avg 5d forward</div>
      </div>`;

    // ── Compact analog cards ─────────────────────────────────────
    const maxSim = Math.max(...data.analogs.map((a) => a.similarity), 0.01);
    const fmt = (v) => v !== undefined ? `${v >= 0 ? "+" : ""}${Number(v).toFixed(3)}` : "n/a";
    cardsEl.innerHTML = data.analogs.map((a) => {
      const s = a.state || {};
      const hasFwd = a.fwd_5d_pct != null;
      const fwdCls = !hasFwd ? "na" : a.fwd_5d_pct >= 0 ? "up" : "down";
      const fwdTxt = hasFwd ? `${a.fwd_5d_pct >= 0 ? "+" : ""}${a.fwd_5d_pct.toFixed(1)}%` : "n/a";
      const simPct = ((a.similarity / maxSim) * 100).toFixed(1);
      const regimeTag = a.regime ? ` <span class="qa-regime-tag">${a.regime}</span>` : "";
      const headline = a.headline && a.headline !== "(no headline available)"
        ? `<div class="qa-panel-card-headline">"${a.headline}"</div>` : "";
      return `
        <div class="qa-panel-card">
          <div class="qa-panel-card-top">
            <span class="qa-panel-card-rank">#${a.rank}</span>
            <span class="qa-panel-card-date">${a.date}${regimeTag}</span>
            <span class="qa-fwd ${fwdCls}" style="font-size:12px">${fwdTxt}</span>
            <div class="qa-panel-card-simbar" title="sim ${a.similarity.toFixed(3)}">
              <div class="qa-panel-card-simfill" style="width:${simPct}%"></div>
            </div>
          </div>
          <div class="qa-panel-card-signals">vol=${fmt(s.vol_20d)} sent=${fmt(s.news_sent)} mom=${fmt(s.mom_20d)} dist=${fmt(s.dist_to_high)}</div>
          ${headline}
        </div>`;
    }).join("");

    // ── Kernel method — update data-method attr, CSS shows it ──
    methodEl.dataset.method = data.method + (data.kernel_time ? ` · ${data.kernel_time}s` : "");
    const methodSpan = methodEl.querySelector(".qa-method-text") ||
      (() => { const s = document.createElement("span"); s.className = "qa-method-text"; methodEl.prepend(s); return s; })();
    methodSpan.textContent = data.method + (data.kernel_time ? ` · ${data.kernel_time}s` : "");

    // ── Extended charts: unhide BEFORE Plotly renders ─────────
    extEl.hidden = false;
    extMethodEl.textContent = data.method;

    const colors = ["#4f7cff", "#3a9bd5", "#27ae60", "#f39c12", "#e74c3c"];
    this._drawRegimeChart(data, colors);
    this._drawRadarChart(data, colors);
    requestAnimationFrame(() => window.dispatchEvent(new Event("resize")));
  },

  _drawRegimeChart(data, colors) {
    const traces = [];

    // Analog traces
    data.analogs.forEach((a, i) => {
      if (!a.window || !a.window.length) return;
      const hasFwd = a.fwd_5d_pct !== null && a.fwd_5d_pct !== undefined;
      const label  = `#${a.rank} ${a.date}${hasFwd ? ` (${a.fwd_5d_pct >= 0 ? "+" : ""}${a.fwd_5d_pct.toFixed(1)}% 5d)` : ""}`;
      traces.push({
        x: a.window.map((p) => p.day),
        y: a.window.map((p) => p.pct),
        mode: "lines",
        name: label,
        line: { color: colors[i], width: 1.8 },
        hovertemplate: `Day %{x}<br>%{y:+.2f}%<extra>${label}</extra>`,
      });
    });

    // Query window trace (dashed white)
    if (data.query_window && data.query_window.length) {
      traces.push({
        x: data.query_window.map((p) => p.day),
        y: data.query_window.map((p) => p.pct),
        mode: "lines",
        name: `Query ${data.query_date}`,
        line: { color: "#e7ebf3", width: 2.2, dash: "dot" },
        hovertemplate: `Day %{x}<br>%{y:+.2f}%<extra>Query ${data.query_date}</extra>`,
      });
    }

    const layout = {
      paper_bgcolor: "transparent",
      plot_bgcolor:  "transparent",
      margin: { t: 6, b: 36, l: 44, r: 10 },
      font:   { color: "#8b93a7", size: 11, family: "-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif" },
      xaxis: {
        title: { text: "Day offset from analog date", font: { size: 10 } },
        zeroline: true, zerolinecolor: "#444", zerolinewidth: 1,
        gridcolor: "#1e222e", tickcolor: "#2a2f3d", linecolor: "#2a2f3d",
        showgrid: true,
      },
      yaxis: {
        title: { text: "Return (%)", font: { size: 10 } },
        zeroline: true, zerolinecolor: "#444", zerolinewidth: 1,
        gridcolor: "#1e222e", tickcolor: "#2a2f3d", linecolor: "#2a2f3d",
        ticksuffix: "%",
      },
      legend: {
        orientation: "h", x: 0, y: -0.22,
        font: { size: 10 }, bgcolor: "transparent",
      },
      shapes: [{
        type: "line", x0: 0, x1: 0, y0: 0, y1: 1,
        xref: "x", yref: "paper",
        line: { color: "#ffffff22", width: 1, dash: "dash" },
      }],
    };

    Plotly.react("qa-chart", traces, layout, {
      displayModeBar: false, responsive: true,
    });
  },

  _drawRadarChart(data, colors) {
    const dims = ["vol_20d", "news_sent", "mom_20d", "dist_to_high"];
    const labels = ["Vol 20d", "News Sentiment", "Momentum 20d", "Dist to High"];

    // Collect all values to find per-dim scale for normalisation
    const allStates = [data.query_state, ...data.analogs.map((a) => a.state)];
    const scales = dims.map((d) =>
      Math.max(...allStates.map((s) => Math.abs(s[d] || 0)), 0.01)
    );

    const norm = (state) => dims.map((d, i) => (state[d] || 0) / scales[i]);

    const traces = [];

    // Analog traces (filled, semi-transparent)
    data.analogs.forEach((a, i) => {
      const vals = norm(a.state);
      const hasFwd = a.fwd_5d_pct !== null && a.fwd_5d_pct !== undefined;
      traces.push({
        type: "scatterpolar",
        r: [...vals, vals[0]],
        theta: [...labels, labels[0]],
        fill: "toself",
        fillcolor: colors[i] + "28",
        line: { color: colors[i], width: 1.6 },
        name: `#${a.rank} ${a.date}${hasFwd ? ` (${a.fwd_5d_pct >= 0 ? "+" : ""}${a.fwd_5d_pct.toFixed(1)}%)` : ""}`,
        hovertemplate: dims.map((d, j) =>
          `${labels[j]}: ${(a.state[d] >= 0 ? "+" : "")}${a.state[d].toFixed(3)}`
        ).join("<br>") + "<extra>" + `#${a.rank} ${a.date}` + "</extra>",
      });
    });

    // Query trace (white dashed, no fill)
    const qVals = norm(data.query_state);
    traces.push({
      type: "scatterpolar",
      r: [...qVals, qVals[0]],
      theta: [...labels, labels[0]],
      fill: "none",
      line: { color: "#e7ebf3", width: 2.4, dash: "dot" },
      name: `Query ${data.query_date}`,
      hovertemplate: dims.map((d, j) =>
        `${labels[j]}: ${(data.query_state[d] >= 0 ? "+" : "")}${data.query_state[d].toFixed(3)}`
      ).join("<br>") + `<extra>Query ${data.query_date}</extra>`,
    });

    const layout = {
      polar: {
        bgcolor: "transparent",
        radialaxis: {
          visible: true,
          range: [-1, 1],
          tickfont: { size: 9, color: "#8b93a7" },
          gridcolor: "#2a2f3d",
          linecolor: "#2a2f3d",
        },
        angularaxis: {
          tickfont: { size: 10, color: "#8b93a7" },
          gridcolor: "#2a2f3d",
          linecolor: "#2a2f3d",
        },
      },
      paper_bgcolor: "transparent",
      plot_bgcolor:  "transparent",
      margin: { t: 10, b: 30, l: 30, r: 30 },
      font:   { color: "#8b93a7", size: 11, family: "-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif" },
      legend: {
        orientation: "h", x: 0, y: -0.12,
        font: { size: 10 }, bgcolor: "transparent",
      },
      showlegend: true,
    };

    Plotly.react("qa-radar", traces, layout, {
      displayModeBar: false, responsive: true,
    });
  },
};
