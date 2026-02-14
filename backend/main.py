from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# --------- Config ----------
HOLDINGS_CSV = os.getenv("HOLDINGS_CSV", "./holdings.csv")
DEFAULT_LOOKBACK_YEARS = int(os.getenv("LOOKBACK_YEARS", "5"))

# Scenario shifts (annualized) added to estimated mean returns
SCENARIO_DELTA = {
    "bear": -0.04,
    "base": 0.00,
    "bull": +0.04,
}

# Optional hard overrides if a symbol is not the Yahoo Finance ticker
YF_OVERRIDES = {
    # "SYMBOL": "YAHOO_TICKER",
}

# Some symbols in your portfolio may be unlisted/unavailable on Yahoo Finance.
# You can provide manual assumptions for those through the API (cagr/vol).
# ---------------------------------

app = FastAPI(title="Portfolio Estimator API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # lock down in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------- Pydantic models ----------
class Holding(BaseModel):
    instrument: str
    qty: float
    avg_cost: float
    ltp: float
    invested: float
    cur_val: float
    weight_pct: float


class HoldingsResponse(BaseModel):
    total_value: float
    holdings: List[Holding]


class MarketStatusItem(BaseModel):
    instrument: str
    yahoo_ticker: Optional[str]
    available: bool
    reason: Optional[str] = None


class MarketStatusResponse(BaseModel):
    lookback_years: int
    items: List[MarketStatusItem]


class ManualAssumption(BaseModel):
    # Annualized expected return and volatility (e.g., 0.12 = 12%)
    cagr: float = Field(..., ge=-0.5, le=0.6)
    vol: float = Field(..., ge=0.0, le=1.5)


class SimulateRequest(BaseModel):
    years: int = Field(10, ge=1, le=50)
    n_sims: int = Field(2000, ge=200, le=50000)
    scenario: str = Field("base")
    # weights_by_year[year][instrument] = weight percent
    weights_by_year: Dict[int, Dict[str, float]]
    # for symbols with missing market data
    manual_assumptions: Dict[str, ManualAssumption] = Field(default_factory=dict)
    # set to True if you want to include per-stock median values in response (heavier payload)
    include_stock_breakdown: bool = False


class Percentiles(BaseModel):
    p10: float
    p50: float
    p90: float


class SimulateYearResult(BaseModel):
    year: int
    portfolio: Percentiles
    stocks_p50: Optional[Dict[str, float]] = None


class SimulateResponse(BaseModel):
    base_currency: str = "INR"
    years: int
    n_sims: int
    scenario: str
    initial_value: float
    results: List[SimulateYearResult]
    notes: List[str]


# --------- Helpers ----------
def _load_holdings() -> Tuple[pd.DataFrame, float]:
    df = pd.read_csv(HOLDINGS_CSV)
    # drop any unnamed cols
    df = df.loc[:, ~df.columns.str.lower().str.startswith("unnamed")]
    # normalize expected columns (Kite export)
    rename = {
        "Instrument": "instrument",
        "Qty.": "qty",
        "Avg. cost": "avg_cost",
        "LTP": "ltp",
        "Invested": "invested",
        "Cur. val": "cur_val",
    }
    df = df.rename(columns=rename)
    for c in ["qty", "avg_cost", "ltp", "invested", "cur_val"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    total = float(df["cur_val"].sum())
    df["weight_pct"] = np.where(total > 0, (df["cur_val"] / total) * 100.0, 0.0)
    return df, total


def _to_yahoo_ticker(instrument: str) -> Optional[str]:
    # Basic NSE mapping: SYMBOL -> SYMBOL.NS
    if instrument in YF_OVERRIDES:
        return YF_OVERRIDES[instrument]
    # Heuristic: if already has suffix, return as is
    if "." in instrument:
        return instrument
    # Some symbols are not available (e.g. unlisted). We'll try anyway and mark missing if fails.
    return f"{instrument}.NS"


def _fetch_price_history(yahoo_tickers: List[str], lookback_years: int) -> pd.DataFrame:
    """
    Returns dataframe of Adj Close prices (daily) for tickers.
    """
    period = f"{lookback_years}y"
    data = yf.download(
        tickers=" ".join(yahoo_tickers),
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=True,
    )

    # yfinance returns:
    # - if one ticker: columns like ['Open','High',...,'Adj Close']
    # - if many: column multi-index with top-level field names
    if isinstance(data.columns, pd.MultiIndex):
        adj = data["Adj Close"].copy()
    else:
        # single ticker
        adj = data[["Adj Close"]].copy()
        adj.columns = [yahoo_tickers[0]]
    adj = adj.dropna(how="all")
    return adj


def _estimate_mu_cov(adj_close: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Estimate annualized mean returns and covariance from daily log returns.
    """
    rets = np.log(adj_close / adj_close.shift(1)).dropna(how="any")
    mu_daily = rets.mean()
    cov_daily = rets.cov()

    # 252 trading days
    mu_annual = mu_daily * 252.0
    cov_annual = cov_daily * 252.0
    return mu_annual, cov_annual


def _normalize_weights(w: Dict[str, float]) -> Dict[str, float]:
    keys = list(w.keys())
    s = sum(max(0.0, float(w[k])) for k in keys)
    if s <= 0:
        eq = 100.0 / max(1, len(keys))
        return {k: eq for k in keys}
    return {k: max(0.0, float(w[k])) / s * 100.0 for k in keys}


def _rebalance(values: np.ndarray, w_pct: np.ndarray) -> np.ndarray:
    total = values.sum(axis=1, keepdims=True)
    return total * (w_pct.reshape(1, -1) / 100.0)


# --------- Endpoints ----------
@app.get("/holdings", response_model=HoldingsResponse)
def get_holdings() -> HoldingsResponse:
    df, total = _load_holdings()
    holdings = [
        Holding(
            instrument=row["instrument"],
            qty=float(row["qty"]),
            avg_cost=float(row["avg_cost"]),
            ltp=float(row["ltp"]),
            invested=float(row["invested"]),
            cur_val=float(row["cur_val"]),
            weight_pct=float(row["weight_pct"]),
        )
        for _, row in df.iterrows()
    ]
    return HoldingsResponse(total_value=total, holdings=holdings)


@app.get("/marketdata/status", response_model=MarketStatusResponse)
def marketdata_status(lookback_years: int = DEFAULT_LOOKBACK_YEARS) -> MarketStatusResponse:
    df, _ = _load_holdings()
    instruments = df["instrument"].tolist()

    items: List[MarketStatusItem] = []
    yahoo = [_to_yahoo_ticker(i) for i in instruments]

    # Try fetching in one batch; mark missing tickers if their column is absent or all-NaN
    try:
        adj = _fetch_price_history([t for t in yahoo if t is not None], lookback_years=lookback_years)
    except Exception as e:
        # If batch fails, mark all as unknown
        for ins, t in zip(instruments, yahoo):
            items.append(MarketStatusItem(instrument=ins, yahoo_ticker=t, available=False, reason=str(e)))
        return MarketStatusResponse(lookback_years=lookback_years, items=items)

    for ins, t in zip(instruments, yahoo):
        if t is None:
            items.append(MarketStatusItem(instrument=ins, yahoo_ticker=None, available=False, reason="No mapping"))
            continue
        if t not in adj.columns:
            items.append(MarketStatusItem(instrument=ins, yahoo_ticker=t, available=False, reason="No column returned"))
            continue
        col = adj[t]
        if col.dropna().empty:
            items.append(MarketStatusItem(instrument=ins, yahoo_ticker=t, available=False, reason="All-NaN series"))
        else:
            items.append(MarketStatusItem(instrument=ins, yahoo_ticker=t, available=True))

    return MarketStatusResponse(lookback_years=lookback_years, items=items)


@app.post("/simulate", response_model=SimulateResponse)
def simulate(req: SimulateRequest) -> SimulateResponse:
    scenario_key = (req.scenario or "base").lower().strip()
    delta = SCENARIO_DELTA.get(scenario_key, 0.0)

    df, total = _load_holdings()
    instruments = df["instrument"].tolist()
    initial_value = float(total)

    # map instruments -> yahoo
    instr_to_yf = {ins: _to_yahoo_ticker(ins) for ins in instruments}
    yf_tickers = [t for t in instr_to_yf.values() if t is not None]

    notes: List[str] = []
    lookback_years = DEFAULT_LOOKBACK_YEARS

    # Fetch market data & estimate mu/cov for available tickers
    adj = _fetch_price_history(yf_tickers, lookback_years=lookback_years)
    mu_yf, cov_yf = _estimate_mu_cov(adj)

    # Build final universe in the same order as instruments
    # For missing tickers, use manual assumptions (or fallback defaults)
    mu = []
    vol = []
    available_mask = []
    for ins in instruments:
        yf_t = instr_to_yf[ins]
        if yf_t in mu_yf.index and yf_t in cov_yf.index:
            m = float(mu_yf.loc[yf_t]) + float(delta)
            v = float(np.sqrt(cov_yf.loc[yf_t, yf_t]))
            mu.append(m)
            vol.append(max(1e-9, v))
            available_mask.append(True)
        else:
            # manual assumption needed
            if ins in req.manual_assumptions:
                m = float(req.manual_assumptions[ins].cagr) + float(delta)
                v = float(req.manual_assumptions[ins].vol)
                mu.append(m)
                vol.append(max(1e-9, v))
                available_mask.append(False)
                notes.append(f"Used manual assumptions for {ins} (missing market data).")
            else:
                # conservative fallback
                mu.append(0.08 + float(delta))
                vol.append(0.25)
                available_mask.append(False)
                notes.append(f"Used fallback assumptions for {ins} (missing market data). Add manual_assumptions to improve.")
    mu = np.array(mu, dtype=float)
    vol = np.array(vol, dtype=float)

    # Covariance:
    # Start with empirical cov among available symbols, then add diagonal-only for missing.
    n = len(instruments)
    cov = np.zeros((n, n), dtype=float)

    # Fill available-available blocks using mapped tickers
    idx_av = [i for i, ok in enumerate(available_mask) if ok]
    ins_av = [instruments[i] for i in idx_av]
    yf_av = [instr_to_yf[ins] for ins in ins_av]

    if len(idx_av) >= 2:
        cov_block = cov_yf.loc[yf_av, yf_av].to_numpy()
        # apply scenario shift only to mean; keep cov as is
        for a_i, i in enumerate(idx_av):
            for a_j, j in enumerate(idx_av):
                cov[i, j] = float(cov_block[a_i, a_j])

    # Put diagonal variances for all (ensures PSD-ish)
    for i in range(n):
        if cov[i, i] <= 0:
            cov[i, i] = float(vol[i] ** 2)

    # If any missing symbols, keep them uncorrelated (0 off-diagonal) for simplicity
    missing = [instruments[i] for i, ok in enumerate(available_mask) if not ok]
    if missing:
        notes.append("Missing symbols are treated as uncorrelated with others unless you extend the model.")

    # Simulation setup
    years = req.years
    n_sims = req.n_sims

    # Initial holdings values from current weights
    w0 = _normalize_weights(req.weights_by_year.get(0, {ins: float(df.loc[df['instrument']==ins,'weight_pct'].values[0]) for ins in instruments}))
    w0_arr = np.array([w0.get(ins, 0.0) for ins in instruments], dtype=float)
    w0_arr = np.array(list(_normalize_weights({ins: w for ins, w in zip(instruments, w0_arr)}).values()), dtype=float)

    values = np.repeat((initial_value * (w0_arr / 100.0)).reshape(1, -1), repeats=n_sims, axis=0)

    # We'll draw annual returns using multivariate normal on log-returns approximation:
    # r ~ N(mu, cov) for arithmetic returns (approx). For more realism use lognormal; this is a pragmatic estimator.
    rng = np.random.default_rng(42)

    results: List[SimulateYearResult] = []

    # Year 0 percentiles
    port0 = values.sum(axis=1)
    results.append(SimulateYearResult(
        year=0,
        portfolio=Percentiles(
            p10=float(np.percentile(port0, 10)),
            p50=float(np.percentile(port0, 50)),
            p90=float(np.percentile(port0, 90)),
        ),
        stocks_p50={ins: float(np.percentile(values[:, i], 50)) for i, ins in enumerate(instruments)} if req.include_stock_breakdown else None
    ))

    for y in range(1, years + 1):
        wy = _normalize_weights(req.weights_by_year.get(y, req.weights_by_year.get(y - 1, w0)))
        w_arr = np.array([wy.get(ins, 0.0) for ins in instruments], dtype=float)
        w_arr = np.array(list(_normalize_weights({ins: w for ins, w in zip(instruments, w_arr)}).values()), dtype=float)

        # Rebalance at start of year
        values = _rebalance(values, w_arr)

        # Draw returns and apply
        r = rng.multivariate_normal(mean=mu, cov=cov, size=n_sims)
        # clamp extreme arithmetic returns
        r = np.clip(r, -0.90, 2.00)
        values = values * (1.0 + r)

        port = values.sum(axis=1)
        results.append(SimulateYearResult(
            year=y,
            portfolio=Percentiles(
                p10=float(np.percentile(port, 10)),
                p50=float(np.percentile(port, 50)),
                p90=float(np.percentile(port, 90)),
            ),
            stocks_p50={ins: float(np.percentile(values[:, i], 50)) for i, ins in enumerate(instruments)} if req.include_stock_breakdown else None
        ))

    return SimulateResponse(
        years=years,
        n_sims=n_sims,
        scenario=scenario_key,
        initial_value=initial_value,
        results=results,
        notes=[
            f"Lookback used for market estimates: {lookback_years} years (daily data).",
            "This is an estimator based on historical mean/vol/cov + scenario shift; not a prediction.",
            "Rebalancing is applied at the start of each year using your weights_by_year.",
            *notes
        ],
    )
