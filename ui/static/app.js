// Qubitrade dashboard front end.
// Top = ticker chips, bottom = timeframe + forecast. White line = real,
// green line = prediction (+ confidence band). Right panel = quantum-kernel
// analog regimes. Stat strip = today's move, predicted move, hit-rate.

const state = { ticker: "NVDA", timeframe: "3M", forecast: 10, model: "lightgbm" };

const el = {
  tickers: document.getElementById("tickers"),
  timeframes: document.getElementById("timeframes"),
  chart: document.getElementById("chart"),
  skeleton: document.getElementById("skeleton"),
  forecast: document.getElementById("forecast"),
  forecastVal: document.getElementById("forecast-val"),
  model: document.getElementById("model"),
  stubBadge: document.getElementById("stub-badge"),
  stats: document.getElementById("stats"),
  analogs: document.getElementById("analogs"),
  narration: document.getElementById("narration"),
  analogMethod: document.getElementById("analog-method"),
};

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

  refresh();
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
  for (const m of models) {
    const opt = document.createElement("option");
    opt.value = m.key;
    opt.textContent = m.label;
    if (m.desc) opt.title = m.desc;
    if (m.key === state.model) opt.selected = true;
    el.model.appendChild(opt);
  }
}

function selectTicker(t) {
  const changed = t !== state.ticker;
  state.ticker = t;
  document.querySelectorAll(".chip").forEach((c) =>
    c.classList.toggle("active", c.dataset.ticker === t)
  );
  refresh(changed);
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

  // analogs only depend on the ticker — refetch when it changes
  if (reloadAnalogs) loadAnalogs(state.ticker);

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

async function loadAnalogs(ticker) {
  el.narration.textContent = "retrieving analogs…";
  el.analogs.innerHTML = "";
  try {
    const d = await fetch(`/api/analogs?ticker=${ticker}`).then((r) => r.json());
    if (ticker !== state.ticker) return; // superseded
    el.narration.textContent = d.narration || "No analogs available.";
    el.analogMethod.textContent = d.method || "";
    el.analogs.innerHTML = d.analogs
      .map((a) => {
        const cls = a.fwd_5d_pct >= 0 ? "up" : "down";
        return `<div class="analog-item">
          <div class="analog-meta">
            <span class="analog-date">${a.date}</span>
            <span class="analog-tag">${a.regime_tag}</span>
            <span class="analog-sim">similarity ${a.similarity}</span>
          </div>
          <div class="analog-fwd ${cls}">${fmtPct(a.fwd_5d_pct)}<div class="analog-tag" style="text-align:right">5d fwd</div></div>
        </div>`;
      })
      .join("");
  } catch (e) {
    el.narration.textContent = "Couldn't load analogs.";
  }
}

function modelLabel(key) {
  const opt = el.model.querySelector(`option[value="${key}"]`);
  return opt ? opt.textContent : key;
}

function draw(data) {
  el.stubBadge.hidden = !data.prediction.is_stub;
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
