"""
Q-Atlas query engine — shared by FastAPI UI and notebook.

Quantum kernel pipeline:
  1. Classical cosine pre-filter → top-100 candidates
  2. xpyq remote quantum covariance kernel (if XPYQ_KEY set)
     OR local Qiskit ZZFeatureMap fallback
  3. Re-rank → top-K analogs
  4. Gemini narration with plausible news reasoning
"""
import os, re, time, json, warnings
import requests
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
warnings.filterwarnings('ignore')

from qiskit.circuit.library import ZZFeatureMap
try:
    from qiskit_machine_learning.kernels import FidelityStatevectorKernel
    _USE_SV = True
except ImportError:
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    _USE_SV = False

import google.generativeai as genai

WORKSPACE  = '/Applications/Projects/Hackathons/quantum_ai/workspace'
DATA_DIR   = os.path.join(WORKSPACE, 'data')

STATE_FEATURES = ['vol_20d', 'news_sent', 'mom_20d', 'dist_to_high']
N_QUBITS       = 4
REPS           = 2
TOP_K          = 5
XPYQ_BASE      = 'https://xpyq-lib-production.up.railway.app'
XPYQ_PREFILTER = 100   # classical pre-filter size before xpyq rerank

_ALIASES = {
    'nvidia': 'NVDA', 'apple': 'AAPL', 'microsoft': 'MSFT',
    'google': 'GOOGL', 'alphabet': 'GOOGL', 'amazon': 'AMZN',
    'meta': 'META', 'facebook': 'META', 'tesla': 'TSLA',
    'amd': 'AMD', 'intel': 'INTC', 'tsmc': 'TSM',
    'jpmorgan': 'JPM', 'jp morgan': 'JPM', 'bank of america': 'BAC',
    'johnson': 'JNJ', 'pfizer': 'PFE', 'walmart': 'WMT',
    'coca cola': 'KO', 'coke': 'KO', 'pepsi': 'PEP',
    'caterpillar': 'CAT', 'chevron': 'CVX', 'exxon': 'XOM',
    'visa': 'V', 'mastercard': 'MA', 'eli lilly': 'LLY',
    'lilly': 'LLY', 'united health': 'UNH', 'procter': 'PG',
    'boeing': 'BA',
}

# Sector info for news context
_SECTOR = {
    'NVDA': 'semiconductors/AI chips', 'AAPL': 'consumer tech/iPhone',
    'MSFT': 'cloud/enterprise software', 'GOOGL': 'digital advertising/cloud',
    'JPM':  'investment banking/consumer lending', 'BAC': 'retail banking',
    'V':    'payments network', 'MA': 'payments network',
    'JNJ':  'pharmaceuticals/medtech', 'PFE': 'pharmaceuticals',
    'UNH':  'managed care/health insurance', 'LLY': 'GLP-1 weight-loss drugs',
    'WMT':  'big-box retail', 'PG': 'consumer staples/household products',
    'KO':   'beverages/consumer staples', 'PEP': 'beverages/snacks',
    'XOM':  'integrated oil & gas', 'CVX': 'integrated oil & gas',
    'CAT':  'heavy machinery/infrastructure', 'BA': 'commercial aerospace/defense',
}


_COMPANY_NAME = {
    'NVDA': 'NVIDIA', 'AAPL': 'Apple', 'MSFT': 'Microsoft', 'GOOGL': 'Alphabet Google',
    'JPM': 'JPMorgan', 'BAC': 'Bank of America', 'V': 'Visa', 'MA': 'Mastercard',
    'JNJ': 'Johnson Johnson', 'PFE': 'Pfizer', 'UNH': 'UnitedHealth', 'LLY': 'Eli Lilly',
    'WMT': 'Walmart', 'PG': 'Procter Gamble', 'KO': 'Coca-Cola', 'PEP': 'PepsiCo',
    'XOM': 'ExxonMobil', 'CVX': 'Chevron', 'CAT': 'Caterpillar', 'BA': 'Boeing',
}


def _parse_news(news_today: str, market_news: str, max_items: int = 6) -> list[str]:
    """Split pipe-delimited news strings into a clean list of headlines."""
    combined = []
    for raw in (news_today, market_news):
        if raw and str(raw) not in ('nan', 'None', ''):
            combined += [h.strip() for h in str(raw).split('|') if h.strip()]
    seen = set()
    out = []
    for h in combined:
        key = h[:60].lower()
        if key not in seen:
            seen.add(key)
            out.append(h)
        if len(out) >= max_items:
            break
    return out


class QAtlasEngine:
    def __init__(self, workspace_dir=WORKSPACE, gemini_api_key=None):
        self.workspace = workspace_dir
        self.data_dir  = os.path.join(workspace_dir, 'data')

        # Gemini
        api_key = gemini_api_key or os.environ.get('GOOGLE_API_KEY', '')
        if api_key:
            genai.configure(api_key=api_key)
            self._llm_ready = True
        else:
            self._llm_ready = False

        # xpyq
        self._xpyq_key = (os.environ.get('XPYQ_KEY') or
                          os.environ.get('XPYQ_API_KEY') or '')

        # Master dataset
        self.master = pd.read_parquet(os.path.join(self.data_dir, 'master_dataset.parquet'))
        self.master['timestamp'] = pd.to_datetime(self.master['timestamp'])
        self.tickers  = sorted(self.master['ticker'].unique().tolist())
        self.date_min = self.master['timestamp'].min()
        self.date_max = self.master['timestamp'].max()

        self._build_indices()

        # Local Qiskit fallback kernel
        feature_map = ZZFeatureMap(feature_dimension=N_QUBITS, reps=REPS, entanglement='linear')
        self._qiskit_kernel = (
            FidelityStatevectorKernel(feature_map=feature_map) if _USE_SV
            else FidelityQuantumKernel(feature_map=feature_map)
        )
        self._kernel_backend = 'FidelityStatevectorKernel' if _USE_SV else 'FidelityQuantumKernel'

        mode = 'xpyq remote' if self._xpyq_key else f'local {self._kernel_backend}'
        print(f'Q-Atlas engine ready | {len(self.tickers)} tickers | kernel: {mode}')

    # ── Index ─────────────────────────────────────────────────────────────────
    def _build_indices(self):
        self._ticker_dates   = {}
        self._ticker_matrix  = {}
        self._ticker_scalers = {}
        self._ticker_df      = {}
        for ticker in self.tickers:
            df_t = (self.master[self.master['ticker'] == ticker]
                    .sort_values('timestamp').copy())
            state_df = df_t[['timestamp'] + STATE_FEATURES].dropna().reset_index(drop=True)
            if len(state_df) < 50:
                continue
            scaler = StandardScaler()
            mat    = scaler.fit_transform(state_df[STATE_FEATURES].values)
            self._ticker_dates[ticker]   = pd.DatetimeIndex(state_df['timestamp'])
            self._ticker_matrix[ticker]  = mat
            self._ticker_scalers[ticker] = scaler
            self._ticker_df[ticker]      = state_df

    # ── NLP ticker parsing ────────────────────────────────────────────────────
    def parse_ticker(self, text: str):
        for t in self.tickers:
            if re.search(r'\b' + t + r'\b', text, re.IGNORECASE):
                return t, 'exact'
        text_lower = text.lower()
        for alias, ticker in _ALIASES.items():
            if alias in text_lower and ticker in self.tickers:
                return ticker, 'alias'
        return None, None

    # ── xpyq quantum covariance kernel ───────────────────────────────────────
    def _xpyq_kernel(self, query_vec: np.ndarray, cand_matrix: np.ndarray) -> np.ndarray | None:
        """
        Two-stage quantum kernel via xpyq:
          1. Classical cosine pre-filter → top XPYQ_PREFILTER candidates
          2. Submit [query + top-N] to xpyq quantum covariance
          3. Return full-length similarity vector (re-inserting zeros for
             candidates that were filtered out)

        xpyq API: POST /api/v1/compute/runs with Python code string.
        The code uses from_numpy() + quant.covariance() quantum ops.
        Returns K[0, 1:] — query row of the (N+1 × N+1) kernel matrix.
        """
        if not self._xpyq_key:
            return None

        N = len(cand_matrix)
        pre_n = min(XPYQ_PREFILTER, N)

        # Stage 1 — classical cosine pre-filter
        cos_sim = cosine_similarity(query_vec, cand_matrix)[0]
        pre_idx = np.argsort(cos_sim)[::-1][:pre_n]
        pre_matrix = cand_matrix[pre_idx]     # (pre_n, 4)

        # Stage 2 — xpyq quantum covariance on [query + pre_n candidates]
        # X shape: (pre_n+1, 4).  X.T shape: (4, pre_n+1)
        # quant.covariance(X.T) → (pre_n+1, pre_n+1) quantum kernel
        # We emit only K[0, 1:] — the query row, length pre_n
        X_all = np.vstack([query_vec, pre_matrix]).astype(np.float32)
        X_t   = X_all.T          # (4, pre_n+1)

        code = (
            "import numpy as _np\n"
            "import json as _json\n"
            f"X_t = _np.array({X_t.tolist()}, dtype=_np.float32)\n"
            "R   = from_numpy(X_t)\n"
            "K   = quant.covariance(R).numpy()\n"   # (pre_n+1, pre_n+1)
            "sim = K[0, 1:]\n"                       # query vs candidates
            "s_min, s_max = float(sim.min()), float(sim.max())\n"
            "sim_norm = (sim - s_min) / (s_max - s_min + 1e-9)\n"
            'print("DATA_START")\n'
            "print(_json.dumps(sim_norm.tolist()))\n"
            'print("DATA_END")\n'
        )

        H = {"Authorization": f"Bearer {self._xpyq_key}",
             "Content-Type": "application/json"}
        try:
            submit = requests.post(
                f"{XPYQ_BASE}/api/v1/compute/runs", headers=H,
                json={"code": code, "name": "qatlas_kernel"}, timeout=15,
            )
            if submit.status_code not in (200, 202):
                return None
            run_id = submit.json().get("run_id")
            if not run_id:
                return None

            # Give xpyq 30 s to complete; if still queued after 15 s → Qiskit fallback
            XPYQ_TIMEOUT   = 30
            QUEUE_PATIENCE = 15   # bail immediately if stuck in queue this long
            deadline   = time.time() + XPYQ_TIMEOUT
            queued_since: float | None = None
            poll: dict | None = None
            last_status  = ''
            last_print_t = time.time()
            while time.time() < deadline:
                try:
                    r = requests.get(
                        f"{XPYQ_BASE}/api/v1/compute/runs/{run_id}",
                        headers=H, timeout=10,
                    )
                    if r.status_code in (200, 202):
                        poll   = r.json()
                        status = poll.get("status", "?")
                        qp     = poll.get("queue_position", "")
                        pos_str = f" (pos {qp})" if qp else ""
                        now = time.time()
                        if status != last_status or (now - last_print_t) >= 10:
                            print(f"  xpyq: {status}{pos_str}", flush=True)
                            last_status  = status
                            last_print_t = now
                        if status == "queued":
                            if queued_since is None:
                                queued_since = now
                            elif now - queued_since > QUEUE_PATIENCE:
                                print("  xpyq: queue wait exceeded — falling back to Qiskit", flush=True)
                                return None
                        else:
                            queued_since = None  # reset if it moves to running
                        if isinstance(poll, dict) and poll.get("status") in (
                            "completed", "failed", "timed_out", "cancelled", "error"
                        ):
                            break
                except Exception:
                    pass
                time.sleep(2)

            if not isinstance(poll, dict) or poll.get("status") != "completed":
                print(f"  xpyq: gave up (status={poll.get('status') if poll else 'none'}) — falling back to Qiskit", flush=True)
                return None
            stdout = poll.get("stdout", "")
            if "DATA_START" not in stdout or "DATA_END" not in stdout:
                return None

            payload  = stdout.split("DATA_START")[1].split("DATA_END")[0].strip()
            xpyq_sim = np.array(json.loads(payload), dtype=np.float32)  # (pre_n,)

        except Exception:
            return None

        # Expand back to full N-length vector (unseen candidates get 0)
        full_sim = np.zeros(N, dtype=np.float32)
        full_sim[pre_idx] = xpyq_sim
        return full_sim

    # ── Core query ────────────────────────────────────────────────────────────
    def query(self, ticker: str, date=None, top_k: int = TOP_K,
              exclude_window: int = 30):
        """
        Quantum KNN for one ticker.  Returns (result_dict, error_str).

        Kernel priority:
          1. xpyq remote quantum covariance   (if XPYQ_KEY set)
          2. local Qiskit ZZFeatureMap         (fallback)
        """
        if ticker not in self._ticker_dates:
            return None, f'{ticker} not in dataset. Available: {self.tickers}'

        dates  = self._ticker_dates[ticker]
        matrix = self._ticker_matrix[ticker]
        scaler = self._ticker_scalers[ticker]
        df_t   = self._ticker_df[ticker]

        # Resolve date
        if date is None:
            query_ts = dates[-1]
        else:
            query_ts = pd.Timestamp(date)
            if query_ts not in dates:
                idx = int(np.argmin(np.abs(dates - query_ts)))
                query_ts = dates[idx]

        q_pos     = np.where(dates == query_ts)[0][0]
        query_raw = df_t.iloc[q_pos][STATE_FEATURES].values.astype(float).reshape(1, -1)
        query_vec = scaler.transform(query_raw)

        lo   = query_ts - pd.Timedelta(days=exclude_window)
        hi   = query_ts + pd.Timedelta(days=exclude_window)
        mask = (dates < lo) | (dates > hi)

        cand_dates  = dates[mask]
        cand_matrix = matrix[mask]

        if len(cand_matrix) == 0:
            return None, 'No candidates after exclusion window'

        # ── Quantum kernel ────────────────────────────────────────
        t0 = time.time()
        kernel_used = 'xpyq'

        k_sim = self._xpyq_kernel(query_vec, cand_matrix)

        if k_sim is None:
            # Fallback to local Qiskit
            kernel_used = self._kernel_backend
            k_sim = self._qiskit_kernel.evaluate(
                x_vec=query_vec, y_vec=cand_matrix
            )[0]

        elapsed = time.time() - t0

        # Top-K analogs
        top_idx = np.argsort(k_sim)[::-1][:top_k]
        analogs = []
        for rank, idx in enumerate(top_idx):
            analog_ts = cand_dates[idx]
            ctx_rows  = self.master[
                (self.master['ticker'] == ticker) &
                (self.master['timestamp'] == analog_ts)
            ]
            ctx = {}
            if len(ctx_rows):
                row = ctx_rows.iloc[0]
                ctx['return_1d']     = float(row.get('return_1d')     or 0)
                ctx['fwd_return_5d'] = float(row.get('fwd_return_5d') or 0)
                ctx['news_sent']     = float(row.get('news_sent')      or 0)
                ctx['regime']        = str(row.get('regime', 'unknown'))
                ctx['price']         = float(row.get('price', 0)       or 0)
                ctx['state']         = {f: float(row.get(f) or 0) for f in STATE_FEATURES}
                ctx['news_items'] = _parse_news(
                    row.get('news_today', ''), row.get('market_news_today', '')
                )
                ctx['headline'] = ctx['news_items'][0] if ctx['news_items'] else '(no headline available)'

            analogs.append({
                'rank':       rank + 1,
                'date':       analog_ts,
                'similarity': float(k_sim[idx]),
                'context':    ctx,
            })

        q_row = self.master[
            (self.master['ticker'] == ticker) &
            (self.master['timestamp'] == query_ts)
        ]
        query_state = {}
        if len(q_row):
            r = q_row.iloc[0]
            query_state = {f: float(r.get(f) or 0) for f in STATE_FEATURES}
            query_state['price']     = float(r.get('price', 0)     or 0)
            query_state['return_1d'] = float(r.get('return_1d', 0) or 0)
            query_state['regime']    = str(r.get('regime', 'unknown'))

        return {
            'ticker':       ticker,
            'query_date':   query_ts,
            'query_state':  query_state,
            'analogs':      analogs,
            'n_candidates': len(cand_matrix),
            'kernel_time':  elapsed,
            'kernel_used':  kernel_used,
        }, None

    # ── Live signal computation ───────────────────────────────────────────────
    def _live_news_sentiment(self, ticker: str) -> tuple[float, list[str]]:
        """Fetch today's headlines via NewsAPI; return (sentiment_score, headlines)."""
        api_key = os.environ.get('NEWS_API_KEY', '')
        headlines = []
        if not api_key:
            return 0.0, headlines
        try:
            company = _COMPANY_NAME.get(ticker, ticker)
            r = requests.get(
                'https://newsapi.org/v2/everything',
                params={'q': company, 'sortBy': 'publishedAt', 'pageSize': 8,
                        'language': 'en', 'apiKey': api_key},
                timeout=8,
            )
            if r.status_code != 200:
                return 0.0, headlines
            articles = r.json().get('articles', [])
            pos = {'surge', 'gain', 'rise', 'beat', 'record', 'high', 'growth',
                   'strong', 'rally', 'upgrade', 'buy', 'bullish', 'profit', 'exceed'}
            neg = {'fall', 'drop', 'loss', 'miss', 'low', 'weak', 'decline',
                   'downgrade', 'sell', 'crash', 'risk', 'bearish', 'warn', 'cut'}
            scores = []
            for a in articles:
                title = (a.get('title') or '')
                desc  = (a.get('description') or '')
                text  = (title + ' ' + desc).lower()
                if title:
                    headlines.append(title[:120])
                words = set(text.split())
                p = len(words & pos)
                n = len(words & neg)
                if p + n:
                    scores.append((p - n) / (p + n))
            return (float(np.mean(scores)) if scores else 0.0), headlines[:6]
        except Exception:
            return 0.0, headlines

    def compute_live_state(self, ticker: str) -> dict:
        """Compute current state vector from live yfinance + NewsAPI."""
        import yfinance as yf
        raw = yf.download(ticker, period='2y', interval='1d',
                          progress=False, auto_adjust=True)
        close = raw['Close'].squeeze()
        rets  = close.pct_change()
        vol_20d      = float(rets.rolling(20).std().iloc[-1])
        mom_20d      = float(close.pct_change(20).iloc[-1])
        high_252     = float(close.rolling(252).max().iloc[-1])
        dist_to_high = float(close.iloc[-1] / high_252 - 1) if high_252 > 0 else 0.0
        news_sent, live_headlines = self._live_news_sentiment(ticker)
        return {
            'vol_20d':      vol_20d,
            'news_sent':    news_sent,
            'mom_20d':      mom_20d,
            'dist_to_high': dist_to_high,
            'price':        float(close.iloc[-1]),
            'live_date':    close.index[-1].strftime('%Y-%m-%d'),
            'live_headlines': live_headlines,
        }

    def query_live(self, ticker: str, top_k: int = TOP_K):
        """Run quantum KNN using today's live signals as the query vector."""
        if ticker not in self._ticker_dates:
            return None, f'{ticker} not in dataset'

        live = self.compute_live_state(ticker)
        state_raw = np.array(
            [live[f] for f in STATE_FEATURES], dtype=float
        ).reshape(1, -1)
        query_vec = self._ticker_scalers[ticker].transform(state_raw)

        cand_dates  = self._ticker_dates[ticker]
        cand_matrix = self._ticker_matrix[ticker]

        t0 = time.time()
        kernel_used = 'xpyq'
        k_sim = self._xpyq_kernel(query_vec, cand_matrix)
        if k_sim is None:
            kernel_used = self._kernel_backend
            k_sim = self._qiskit_kernel.evaluate(x_vec=query_vec, y_vec=cand_matrix)[0]
        elapsed = time.time() - t0

        top_idx = np.argsort(k_sim)[::-1][:top_k]
        analogs = []
        for rank, idx in enumerate(top_idx):
            analog_ts = cand_dates[idx]
            ctx_rows  = self.master[
                (self.master['ticker'] == ticker) &
                (self.master['timestamp'] == analog_ts)
            ]
            ctx = {}
            if len(ctx_rows):
                row = ctx_rows.iloc[0]
                ctx['return_1d']     = float(row.get('return_1d')     or 0)
                ctx['fwd_return_5d'] = float(row.get('fwd_return_5d') or 0)
                ctx['news_sent']     = float(row.get('news_sent')      or 0)
                ctx['regime']        = str(row.get('regime', 'unknown'))
                ctx['price']         = float(row.get('price', 0)       or 0)
                ctx['state']         = {f: float(row.get(f) or 0) for f in STATE_FEATURES}
                ctx['news_items']    = _parse_news(
                    row.get('news_today', ''), row.get('market_news_today', '')
                )
                ctx['headline']      = ctx['news_items'][0] if ctx['news_items'] else '(no headline available)'
            analogs.append({
                'rank':       rank + 1,
                'date':       analog_ts,
                'similarity': float(k_sim[idx]),
                'context':    ctx,
            })

        query_state = {f: live[f] for f in STATE_FEATURES}
        query_state.update({'price': live['price'], 'regime': 'live (today)'})

        return {
            'ticker':           ticker,
            'query_date':       live['live_date'],
            'query_state':      query_state,
            'analogs':          analogs,
            'n_candidates':     len(cand_matrix),
            'kernel_time':      round(elapsed, 2),
            'kernel_used':      kernel_used,
            'is_live':          True,
            'live_headlines':   live['live_headlines'],
        }, None

    # ── Gemini narration with news reasoning ──────────────────────────────────
    def narrate(self, result: dict, original_query: str = '') -> str:
        if not self._llm_ready:
            return self._template_narrative(result)

        ticker  = result['ticker']
        qdate   = pd.Timestamp(result['query_date']).strftime('%Y-%m-%d')
        qs      = result['query_state']
        analogs = result['analogs']
        sector  = _SECTOR.get(ticker, 'equities')

        # Collect analog headlines for news context
        analog_lines = []
        headlines    = []
        for a in analogs:
            ctx  = a['context']
            line = (
                f"#{a['rank']} {pd.Timestamp(a['date']).strftime('%Y-%m-%d')}  "
                f"sim={a['similarity']:.4f}  "
                f"1d={ctx.get('return_1d',0)*100:+.2f}%  "
                f"fwd5d={ctx.get('fwd_return_5d',0)*100:+.2f}%"
            )
            h = ctx.get('headline', '')
            if h and h != '(no headline available)':
                line    += f'  | "{h[:100]}"'
                headlines.append(h[:100])
            analog_lines.append(line)

        fwds    = [a['context'].get('fwd_return_5d', 0) for a in analogs]
        avg_fwd = np.mean(fwds) * 100
        n_pos   = sum(1 for f in fwds if f > 0)

        # Regime interpretation
        vol_z  = qs.get('vol_20d', 0)
        sent   = qs.get('news_sent', 0)
        mom    = qs.get('mom_20d', 0)
        dist_h = qs.get('dist_to_high', 0)

        regime_desc = []
        if vol_z > 0.5:   regime_desc.append('elevated volatility')
        elif vol_z < -0.5: regime_desc.append('compressed volatility')
        if sent > 0.1:    regime_desc.append('positive news sentiment')
        elif sent < -0.1:  regime_desc.append('negative news sentiment')
        if mom > 0.03:    regime_desc.append('strong upward momentum')
        elif mom < -0.03:  regime_desc.append('downward momentum')
        if dist_h > -0.02: regime_desc.append('near 52-week high')
        elif dist_h < -0.15: regime_desc.append('far below 52-week high')
        regime_str = ', '.join(regime_desc) if regime_desc else 'neutral regime'

        kernel_note = (
            f"xpyq remote quantum covariance kernel (pre-filtered {XPYQ_PREFILTER} → top-{len(analogs)})"
            if result.get('kernel_used') == 'xpyq'
            else f"local ZZFeatureMap quantum kernel (4 qubits, reps=2)"
        )

        prompt = f"""You are Q-Atlas, an institutional quantum market intelligence system.

QUERY: "{original_query or f'Why is {ticker} moving?'}"
TICKER: {ticker} — sector: {sector}
DATE: {qdate}
KERNEL: {kernel_note}

CURRENT MARKET STATE for {ticker}:
  vol_20d={qs.get('vol_20d',0):.4f}  news_sent={qs.get('news_sent',0):.4f}
  mom_20d={qs.get('mom_20d',0):.4f}  dist_to_high={qs.get('dist_to_high',0):.4f}
  price={qs.get('price',0):.2f}  regime={qs.get('regime','unknown')}
  Regime interpretation: {regime_str}

TOP-{len(analogs)} QUANTUM ANALOG REGIMES (most structurally similar historical dates):
{chr(10).join(analog_lines)}

QUANTUM CONSENSUS: {n_pos}/{len(analogs)} analogs positive · avg 5d fwd = {avg_fwd:+.2f}%

Respond in this exact structure (4 sentences total):

1. [WHY] Based on the current regime ({regime_str}) and {ticker}'s sector ({sector}), \
name 1-2 plausible news catalysts that typically drive this type of move \
(e.g. earnings beat/miss, macro data, rate decision, product launch, sector rotation). \
Reference specific analog dates and what drove moves then.

2. [ANALOGS] Name the top 2 most similar analog dates with their actual return numbers. \
What event or dynamic made those dates structurally similar to today?

3. [SIGNAL] State the quantum consensus direction (BULLISH / BEARISH / NEUTRAL) \
with confidence (e.g. "4/5 analogs positive").

4. [POSITION] End with exactly: \
"Stack held [{ticker}, <1-2 sector peers>]; {avg_fwd:+.1f}% avg over 5 days."

Be sharp, specific, and use real numbers. No hedging."""

        for model_name in ['gemini-2.5-flash', 'gemini-1.5-flash', 'gemini-1.5-pro']:
            try:
                model    = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                return response.text
            except Exception:
                continue

        return self._template_narrative(result)

    def _template_narrative(self, result: dict) -> str:
        ticker    = result['ticker']
        analogs   = result['analogs']
        fwds      = [a['context'].get('fwd_return_5d', 0) for a in analogs]
        avg_fwd   = np.mean(fwds) * 100
        n_pos     = sum(1 for f in fwds if f > 0)
        top       = analogs[0]
        direction = 'BULLISH' if avg_fwd > 0.5 else ('BEARISH' if avg_fwd < -0.5 else 'NEUTRAL')
        return (
            f"Quantum memory retrieved {len(analogs)} analogs for {ticker}. "
            f"Closest: {pd.Timestamp(top['date']).strftime('%Y-%m-%d')} "
            f"(sim={top['similarity']:.4f}, fwd5d={top['context'].get('fwd_return_5d',0)*100:+.2f}%). "
            f"{n_pos}/{len(analogs)} analogs positive. Avg fwd 5d: {avg_fwd:+.2f}%. "
            f"Quantum consensus: {direction}."
        )
