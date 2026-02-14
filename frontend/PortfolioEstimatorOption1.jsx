import React, { useEffect, useMemo, useState } from "react";
import {
  ResponsiveContainer,
  LineChart, Line,
  BarChart, Bar,
  XAxis, YAxis, Tooltip, Legend, CartesianGrid,
} from "recharts";

/**
 * Frontend GUI that talks to the FastAPI backend:
 * - GET /holdings
 * - POST /simulate
 *
 * Shows:
 * - sliders to set weights by year
 * - simulation percentile bands (p10/p50/p90)
 * - optional stacked bars for p50 by stock (enable include_stock_breakdown)
 */

const API = "http://127.0.0.1:8000";

function clamp(n, a, b) {
  return Math.max(a, Math.min(b, n));
}

function formatINR(x) {
  const v = Number.isFinite(x) ? x : 0;
  return v.toLocaleString("en-IN", { maximumFractionDigits: 0 });
}

function normalizeWeights(weightsObj) {
  const keys = Object.keys(weightsObj);
  const sum = keys.reduce((acc, k) => acc + (weightsObj[k] ?? 0), 0);
  if (sum <= 0) {
    const eq = 100 / keys.length;
    const out = {};
    keys.forEach((k) => (out[k] = eq));
    return out;
  }
  const out = {};
  keys.forEach((k) => (out[k] = (Math.max(0, weightsObj[k]) / sum) * 100));
  return out;
}

export default function PortfolioEstimatorOption1() {
  const [holdings, setHoldings] = useState([]);
  const [totalValue, setTotalValue] = useState(0);

  const [activeYear, setActiveYear] = useState(0);
  const [years, setYears] = useState(10);
  const [nSims, setNSims] = useState(3000);
  const [scenario, setScenario] = useState("base");

  // weightsByYear[year][instrument]=pct
  const [weightsByYear, setWeightsByYear] = useState({});
  const [manualAssumptions, setManualAssumptions] = useState({}); // {SYMBOL:{cagr,vol}}

  const [simResult, setSimResult] = useState(null);
  const [notes, setNotes] = useState([]);

  useEffect(() => {
    (async () => {
      const res = await fetch(`${API}/holdings`);
      const data = await res.json();
      setHoldings(data.holdings);
      setTotalValue(data.total_value);

      const year0 = {};
      data.holdings.forEach((h) => (year0[h.instrument] = h.weight_pct));
      const all = {};
      for (let y = 0; y <= 10; y++) all[y] = { ...year0 };
      setWeightsByYear(all);
    })();
  }, []);

  const instruments = useMemo(() => holdings.map((h) => h.instrument), [holdings]);
  const activeWeights = weightsByYear[activeYear] || {};

  const setWeight = (sym, val) => {
    setWeightsByYear((prev) => {
      const next = { ...prev };
      const w = { ...(next[activeYear] || {}) };
      w[sym] = clamp(val, 0, 100);
      next[activeYear] = normalizeWeights(w);
      return next;
    });
  };

  const copyWeightsToFuture = () => {
    setWeightsByYear((prev) => {
      const next = { ...prev };
      for (let y = activeYear + 1; y <= years; y++) {
        next[y] = { ...(prev[activeYear] || {}) };
      }
      return next;
    });
  };

  const runSimulation = async () => {
    const payload = {
      years,
      n_sims: nSims,
      scenario,
      weights_by_year: weightsByYear,
      manual_assumptions: manualAssumptions,
      include_stock_breakdown: true,
    };
    const res = await fetch(`${API}/simulate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    setSimResult(data);
    setNotes(data.notes || []);
  };

  const bandData = useMemo(() => {
    if (!simResult?.results) return [];
    return simResult.results.map((r) => ({
      year: `Y${r.year}`,
      p10: r.portfolio.p10,
      p50: r.portfolio.p50,
      p90: r.portfolio.p90,
    }));
  }, [simResult]);

  const stockP50Data = useMemo(() => {
    if (!simResult?.results) return [];
    return simResult.results.map((r) => {
      const row = { year: `Y${r.year}` };
      const sp = r.stocks_p50 || {};
      Object.keys(sp).forEach((k) => (row[k] = sp[k]));
      return row;
    });
  }, [simResult]);

  // simple editor for missing/unlisted tickers:
  const setAssumption = (sym, field, value) => {
    setManualAssumptions((prev) => {
      const next = { ...prev };
      const cur = next[sym] || { cagr: 0.10, vol: 0.25 };
      next[sym] = { ...cur, [field]: value };
      return next;
    });
  };

  return (
    <div style={{ padding: 16, fontFamily: "system-ui, Arial" }}>
      <h2 style={{ margin: "0 0 6px" }}>Portfolio Estimator (Option 1: Historical + Monte Carlo)</h2>
      <div style={{ color: "#555", marginBottom: 12 }}>
        Backend: FastAPI + yfinance. This is an estimator, not a prediction.
      </div>

      <div style={{ display: "flex", gap: 12, flexWrap: "wrap", marginBottom: 12 }}>
        <div style={{ padding: 10, border: "1px solid #ddd", borderRadius: 10 }}>
          <div style={{ color: "#666", fontSize: 12 }}>Current total value</div>
          <div style={{ fontSize: 18, fontWeight: 700 }}>₹ {formatINR(totalValue)}</div>
        </div>

        <label style={{ display: "grid", gap: 6 }}>
          <span>Years</span>
          <input type="number" value={years} min={1} max={30}
                 onChange={(e)=>setYears(Number(e.target.value||10))}
                 style={{ padding: 8, width: 120 }} />
        </label>

        <label style={{ display: "grid", gap: 6 }}>
          <span>Simulations</span>
          <input type="number" value={nSims} min={200} max={50000}
                 onChange={(e)=>setNSims(Number(e.target.value||3000))}
                 style={{ padding: 8, width: 140 }} />
        </label>

        <label style={{ display: "grid", gap: 6 }}>
          <span>Scenario</span>
          <select value={scenario} onChange={(e)=>setScenario(e.target.value)}
                  style={{ padding: 8, width: 140 }}>
            <option value="bear">Bear</option>
            <option value="base">Base</option>
            <option value="bull">Bull</option>
          </select>
        </label>

        <button onClick={runSimulation} style={{ padding: "10px 14px", cursor: "pointer" }}>
          Run simulation
        </button>
      </div>

      <div style={{ display: "flex", gap: 12, flexWrap: "wrap", alignItems: "center" }}>
        <div>
          <b>Edit allocations for year:</b>{" "}
          <select
            value={activeYear}
            onChange={(e) => setActiveYear(Number(e.target.value))}
            style={{ padding: 6, marginLeft: 6 }}
          >
            {Array.from({ length: years + 1 }, (_, i) => (
              <option key={i} value={i}>
                Year {i}
              </option>
            ))}
          </select>
        </div>

        <button onClick={copyWeightsToFuture} style={{ padding: "8px 10px", cursor: "pointer" }}>
          Copy this year’s weights to all future years
        </button>
      </div>

      <div style={{ marginTop: 10, display: "grid", gridTemplateColumns: "1.1fr 0.9fr", gap: 12 }}>
        <div style={{ border: "1px solid #eee", borderRadius: 10, padding: 12 }}>
          <h3 style={{ marginTop: 0 }}>Allocation sliders (Year {activeYear})</h3>
          {instruments.map((sym) => (
            <div key={sym} style={{ marginBottom: 10 }}>
              <div style={{ display: "flex", justifyContent: "space-between" }}>
                <span><b>{sym}</b></span>
                <span>{(activeWeights?.[sym] ?? 0).toFixed(2)}%</span>
              </div>
              <input
                type="range"
                min="0"
                max="100"
                value={activeWeights?.[sym] ?? 0}
                onChange={(e) => setWeight(sym, Number(e.target.value))}
                style={{ width: "100%" }}
              />
            </div>
          ))}
          <div style={{ fontSize: 12, color: "#666" }}>* Sliders auto-normalize to 100%.</div>
        </div>

        <div style={{ border: "1px solid #eee", borderRadius: 10, padding: 12 }}>
          <h3 style={{ marginTop: 0 }}>Manual assumptions (only if needed)</h3>
          <div style={{ color: "#666", fontSize: 12, marginBottom: 8 }}>
            Use for instruments not found in market data (e.g., unlisted). CAGR & Vol are annualized.
          </div>
          {instruments.map((sym) => {
            const a = manualAssumptions[sym] || { cagr: 0.10, vol: 0.25 };
            return (
              <div key={sym} style={{ display: "grid", gridTemplateColumns: "1fr 90px 90px", gap: 8, marginBottom: 8, alignItems:"center" }}>
                <div><b>{sym}</b></div>
                <input type="number" step="0.01" value={a.cagr}
                  onChange={(e)=>setAssumption(sym,"cagr",Number(e.target.value))}
                  style={{ padding: 6 }} />
                <input type="number" step="0.01" value={a.vol}
                  onChange={(e)=>setAssumption(sym,"vol",Number(e.target.value))}
                  style={{ padding: 6 }} />
              </div>
            );
          })}
        </div>
      </div>

      <div style={{ marginTop: 14, border: "1px solid #eee", borderRadius: 10, padding: 12 }}>
        <h3 style={{ marginTop: 0 }}>Portfolio value bands (₹): p10 / p50 / p90</h3>
        <div style={{ height: 320 }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={bandData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="year" />
              <YAxis tickFormatter={(v) => `₹${formatINR(v)}`} />
              <Tooltip formatter={(v)=>`₹ ${formatINR(v)}`} />
              <Legend />
              <Line type="monotone" dataKey="p10" />
              <Line type="monotone" dataKey="p50" />
              <Line type="monotone" dataKey="p90" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div style={{ marginTop: 14, border: "1px solid #eee", borderRadius: 10, padding: 12 }}>
        <h3 style={{ marginTop: 0 }}>Median valuation by stock (stacked, p50)</h3>
        <div style={{ height: 420 }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={stockP50Data}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="year" />
              <YAxis tickFormatter={(v) => `₹${formatINR(v)}`} />
              <Tooltip formatter={(v)=>`₹ ${formatINR(v)}`} />
              <Legend />
              {instruments.map((sym) => (
                <Bar key={sym} dataKey={sym} stackId="a" />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {notes?.length ? (
        <div style={{ marginTop: 12, padding: 12, border: "1px solid #eee", borderRadius: 10, color: "#555" }}>
          <h3 style={{ marginTop: 0 }}>Notes</h3>
          <ul>
            {notes.map((n, i) => <li key={i}>{n}</li>)}
          </ul>
        </div>
      ) : null}
    </div>
  );
}
