// Q-Atlas dashboard front end.
// Top = ticker chips, bottom = timeframe + forecast. Black line = real,
// green line = prediction (from the backend's predictor plug-in).

const state = {
  ticker: "NVDA",
  timeframe: "3M",
  forecast: 10,
  model: "lightgbm",
};

const el = {
  tickers: document.getElementById("tickers"),
  timeframes: document.getElementById("timeframes"),
  chart: document.getElementById("chart"),
  loading: document.getElementById("loading"),
  forecast: document.getElementById("forecast"),
  forecastVal: document.getElementById("forecast-val"),
  model: document.getElementById("model"),
  stubBadge: document.getElementById("stub-badge"),
};

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
  state.ticker = t;
  document.querySelectorAll(".chip").forEach((c) =>
    c.classList.toggle("active", c.dataset.ticker === t)
  );
  refresh();
}

function selectTimeframe(tf) {
  state.timeframe = tf;
  document.querySelectorAll(".tf").forEach((b) =>
    b.classList.toggle("active", b.dataset.tf === tf)
  );
  refresh();
}

let inflight = 0;
async function refresh() {
  const reqId = ++inflight;
  el.loading.hidden = false;
  try {
    const url = `/api/series?ticker=${state.ticker}&timeframe=${state.timeframe}&forecast_days=${state.forecast}&model=${state.model}`;
    const data = await fetch(url).then((r) => {
      if (!r.ok) throw new Error("request failed");
      return r.json();
    });
    if (reqId !== inflight) return; // a newer request superseded this one
    draw(data);
  } catch (e) {
    if (reqId === inflight) {
      Plotly.purge(el.chart);
      el.chart.innerHTML = `<div style="padding:40px;color:#8b93a7">Couldn't load ${state.ticker}. ${e.message}</div>`;
    }
  } finally {
    if (reqId === inflight) el.loading.hidden = true;
  }
}

function modelLabel(key) {
  const opt = el.model.querySelector(`option[value="${key}"]`);
  return opt ? opt.textContent : key;
}

function draw(data) {
  el.stubBadge.hidden = !data.prediction.is_stub;

  const real = {
    x: data.real.dates,
    y: data.real.close,
    mode: "lines",
    name: "Real",
    line: { color: "#e7ebf3", width: 2.5 },
    hovertemplate: "%{x}<br>$%{y:.2f}<extra>real</extra>",
  };

  const pred = {
    x: data.prediction.pred_dates,
    y: data.prediction.pred_close,
    mode: "lines",
    name: "Prediction",
    line: { color: "#1f9d4d", width: 3 },
    hovertemplate: "%{x}<br>$%{y:.2f}<extra>prediction</extra>",
  };

  // Vertical marker where the prediction is anchored.
  const shapes = [];
  if (data.prediction.split_date) {
    shapes.push({
      type: "line",
      x0: data.prediction.split_date,
      x1: data.prediction.split_date,
      yref: "paper",
      y0: 0,
      y1: 1,
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
      font: { color: "#e7ebf3", size: 15 },
      x: 0.01,
    },
    xaxis: { gridcolor: "#1e222e", showline: false, zeroline: false },
    yaxis: {
      gridcolor: "#1e222e",
      tickprefix: "$",
      showline: false,
      zeroline: false,
    },
    legend: {
      orientation: "h",
      x: 1, xanchor: "right",
      y: 1.12,
      font: { color: "#e7ebf3" },
    },
    shapes,
    showlegend: true,
  };

  Plotly.react(el.chart, [real, pred], layout, {
    responsive: true,
    displayModeBar: false,
  });
}

boot();
