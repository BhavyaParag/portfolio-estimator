# Option 1: Historical-data backend + GUI

## What you get
- **FastAPI backend** that:
  - reads your `holdings.csv` (Kite holdings export)
  - pulls **historical prices** from Yahoo Finance via `yfinance`
  - estimates annualized **mean returns & covariance**
  - runs a **Monte Carlo** simulation for the next N years
  - supports **year-by-year rebalancing** (your slider changes per year)

- **React GUI** that:
  - loads holdings from backend
  - lets you set year-wise weights (sliders)
  - runs simulation and plots 10/50/90 percentile bands

> Important: This is an **estimator**, not a guarantee or “prediction”.

---

## 1) Backend setup (FastAPI)

### Install dependencies
```bash
pip install fastapi uvicorn yfinance pandas numpy
```

### Put your holdings file
Copy your `holdings.csv` into the backend folder (same folder as `main.py`).

### Run
```bash
cd backend
uvicorn main:app --reload --port 8000
```

### Test
Open:
- `http://127.0.0.1:8000/holdings`
- `http://127.0.0.1:8000/marketdata/status`

---

## 2) Frontend setup (React)

Install recharts:
```bash
npm i recharts
```

Run your dev server normally (Vite/CRA/etc).

If frontend runs on port 5173, it can call backend on 8000 (CORS enabled).

---

## 3) Missing tickers (important)
Some instruments may be **unlisted** or not available on Yahoo Finance (example: `TATACAP`, `GROWW`).
For those, the API will use:
- your provided `manual_assumptions` (recommended), else
- fallback defaults

You can edit assumptions in the GUI or hardcode them.

---

## 4) Security note
CORS is open (`allow_origins=["*"]`) for ease. Lock it down for production.
