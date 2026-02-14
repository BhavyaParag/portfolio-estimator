# Deploy-ready Portfolio Estimator (Option 1)

Prepared for:
- Backend on **Render** (FastAPI + yfinance + Monte Carlo)
- Frontend on **Vercel** (Vite + React + Recharts)

## A) Deploy Backend to Render (gives you an API link)

1. Upload this project to a GitHub repo.
2. Render → New → **Blueprint** → select your repo.
3. Render reads `render.yaml` and deploys automatically.
4. After deploy, your API link will look like:
   `https://portfolio-estimator-api.onrender.com`

Test:
- `/docs`
- `/holdings`

## B) Deploy Frontend to Vercel (gives you the Chrome link)

1. Vercel → New Project → import the same repo.
2. **Root Directory**: `frontend/vite_app`
3. Framework: **Vite**
4. Add Environment Variable:
   - `VITE_API_URL` = your Render API URL (no trailing slash)
5. Deploy.

Your final public link will be like:
`https://portfolio-estimator-ui.vercel.app`

## Local run (optional)

Backend:
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Frontend:
```bash
cd frontend/vite_app
npm install
npm run dev
```

Open `http://localhost:5173`
